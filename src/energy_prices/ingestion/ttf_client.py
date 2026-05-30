"""TTF gas benchmark client (MVP) backed by yfinance.

The Dutch TTF is the European wholesale gas benchmark. For the MVP we use the
free Yahoo Finance front-month natural-gas TTF futures continuous contract
(`TTF=F`) as a daily proxy, mapped onto our canonical ``PriceObservation`` shape.

This is a *proxy*: Yahoo quotes the ICE TTF front-month future (EUR/MWh), which
tracks but does not equal the day-ahead TTF index. It is good enough as an
exogenous gas driver and a sanity benchmark until a proper TTF/PSV feed (e.g.
GME-GAS or ICE) is wired in. The daily ``Close`` is stamped at 00:00 UTC of its
trading day with a DAILY (1440-minute) resolution.

Public surface:
    TtfClient.fetch(start, end) -> list[dict]   # PriceObservation rows
    ingest(session, start, end) -> int          # fetch + upsert, audited
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from typing import Any

import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session

from energy_prices.config.enums import Market, Resolution
from energy_prices.storage.repositories import IngestionRepository, PriceRepository

logger = logging.getLogger(__name__)

#: Source tag stored on every row and on the ingestion audit log.
SOURCE = "yfinance"

#: Yahoo Finance ticker for the ICE Dutch TTF natural-gas front-month future.
TTF_TICKER = "TTF=F"


def _to_utc_midnight(value: Any) -> dt.datetime | None:
    """Coerce a pandas/py date-like index value to 00:00:00 UTC of that day.

    yfinance returns a tz-naive (or sometimes tz-aware) ``DatetimeIndex`` of
    trading days. We anchor each daily observation to the start of its calendar
    day in UTC so it aligns with the rest of the time-series store.
    """
    ts = pd.Timestamp(value)
    if ts is pd.NaT or pd.isna(ts):
        return None
    # Normalize to the calendar date, dropping any intraday/time-of-day part,
    # then attach UTC. We only care about the date for a DAILY observation.
    date = ts.date()
    return dt.datetime(date.year, date.month, date.day, tzinfo=dt.UTC)


def _coerce_price(value: Any) -> float | None:
    """Return a finite float price, or None for NaN/inf/non-numeric values."""
    if value is None:
        return None
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(price) or math.isinf(price):
        return None
    return price


def _extract_close(df: pd.DataFrame) -> pd.Series | None:
    """Pull a 1-D ``Close`` Series out of a (possibly MultiIndex) yfinance frame.

    yfinance schema quirks handled here:
      * Single-ticker downloads usually have flat columns (``Close``, ...).
      * With ``group_by="ticker"`` or multi-ticker calls, columns are a
        ``MultiIndex`` like ``("Close", "TTF=F")`` or ``("TTF=F", "Close")``.
      * Some versions surface ``Adj Close`` only — we still prefer raw ``Close``
        (we pass ``auto_adjust=False``), falling back to ``Adj Close``.
    Returns None if no usable close column is present.
    """
    if df is None or df.empty:
        return None

    columns = df.columns
    if isinstance(columns, pd.MultiIndex):
        # Find the level that carries the OHLC field names, then slice "Close".
        for field in ("Close", "Adj Close"):
            for level in range(columns.nlevels):
                level_values = columns.get_level_values(level)
                if field in level_values:
                    sub = df.xs(field, axis=1, level=level)
                    # ``sub`` is a DataFrame (one column per ticker); take the
                    # first column as our single TTF series.
                    if isinstance(sub, pd.DataFrame):
                        if sub.shape[1] == 0:
                            continue
                        return sub.iloc[:, 0]
                    return sub
        return None

    # Flat columns.
    for field in ("Close", "Adj Close"):
        if field in columns:
            series = df[field]
            if isinstance(series, pd.DataFrame):  # duplicate column names
                series = series.iloc[:, 0]
            return series
    return None


class TtfClient:
    """Fetch daily TTF front-month closes from Yahoo Finance.

    Stateless and cheap to construct. Network access happens only in
    :meth:`fetch`. Parameters mirror ``yfinance.download`` knobs we care about.
    """

    def __init__(self, ticker: str = TTF_TICKER) -> None:
        self.ticker = ticker

    def fetch(self, start: dt.date, end: dt.date) -> list[dict[str, Any]]:
        """Return PriceObservation-shaped dicts for [start, end] (inclusive).

        yfinance treats the ``end`` argument as *exclusive*, so we add one day
        to include the requested ``end`` trading day. Empty results, NaN closes
        and odd column layouts are handled defensively and never raise.
        """
        if start > end:
            logger.warning("TTF fetch called with start %s after end %s; nothing to do.",
                           start, end)
            return []

        # yfinance's `end` is exclusive — extend by a day to include `end`.
        yf_end = end + dt.timedelta(days=1)
        try:
            raw = yf.download(
                self.ticker,
                start=start.isoformat(),
                end=yf_end.isoformat(),
                interval="1d",
                auto_adjust=False,
                progress=False,
                actions=False,
            )
        except Exception as exc:  # network / yfinance internal errors
            logger.warning("yfinance download for %s failed: %s", self.ticker, exc)
            return []

        if raw is None or getattr(raw, "empty", True):
            logger.info("yfinance returned no rows for %s in %s..%s.",
                        self.ticker, start, end)
            return []

        close = _extract_close(raw)
        if close is None or close.empty:
            logger.warning("No usable Close column in yfinance data for %s.", self.ticker)
            return []

        rows: list[dict[str, Any]] = []
        skipped = 0
        for index_value, raw_price in close.items():
            delivery_start = _to_utc_midnight(index_value)
            price = _coerce_price(raw_price)
            if delivery_start is None or price is None:
                skipped += 1
                continue
            rows.append(
                {
                    "market": Market.TTF.value,
                    "zone": None,
                    "delivery_start": delivery_start,
                    "resolution_minutes": Resolution.DAILY.value,
                    "price": price,
                    "source": SOURCE,
                    "unit": "EUR/MWh",
                    "currency": "EUR",
                }
            )

        if skipped:
            logger.debug("Skipped %d TTF rows with missing date/price.", skipped)
        logger.info("Fetched %d TTF daily observations from %s.", len(rows), SOURCE)
        return rows


def ingest(session: Session, start: dt.date, end: dt.date) -> int:
    """Fetch TTF daily closes for [start, end] and upsert them.

    Records the run in the ingestion audit log and returns the number of rows
    upserted. Failures are logged to the audit log and re-raised so the caller
    (and ``session_scope``) can roll back.
    """
    now = dt.datetime.now(dt.UTC)
    audit = IngestionRepository(session)
    run = audit.start(SOURCE, now)
    try:
        rows = TtfClient().fetch(start, end)
        count = PriceRepository(session).upsert(rows)
        audit.finish(
            run,
            status="success",
            finished_at=dt.datetime.now(dt.UTC),
            rows=count,
            message=f"TTF {Market.TTF.value} {start}..{end}: {count} rows",
        )
        return count
    except Exception as exc:
        logger.exception("TTF ingestion failed for %s..%s", start, end)
        audit.finish(
            run,
            status="failed",
            finished_at=dt.datetime.now(dt.UTC),
            rows=0,
            message=str(exc),
        )
        raise
