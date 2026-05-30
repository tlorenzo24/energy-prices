"""ENTSO-E ingestion client — the primary free electricity data source.

Fetches Italian day-ahead (MGP) zonal prices and the exogenous day-ahead
forecasts (load, wind+solar) that drive forecast accuracy, then reconstructs a
PUN-like volume-weighted index from the 7 zonal prices.

Relies on the ``entsoe-py`` library (>=0.8.0), whose Transparency-Platform
endpoints transparently return 60-minute data before 2025-10-01 and 15-minute
data on/after the SDAC quarter-hour go-live, so the resolution is *inferred*
from each returned series rather than assumed.

All timestamps are normalised to timezone-aware UTC; prices are in EUR/MWh.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import pandas as pd
from entsoe import EntsoePandasClient
from requests import HTTPError
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from energy_prices.config.enums import (
    ENTSOE_ZONE_CODE,
    MARKET_ZONES,
    PUN_ZONE_WEIGHTS,
    Market,
    Zone,
)
from energy_prices.config.settings import Settings, get_settings
from energy_prices.storage.repositories import (
    ExogenousRepository,
    PriceRepository,
)

logger = logging.getLogger(__name__)

SOURCE = "entsoe"

# entsoe-py expects tz-aware timestamps; it localises results to this tz.
_QUERY_TZ = "Europe/Brussels"
# ENTSO-E rejects query windows wider than ~1 year; chunk below that.
_MAX_WINDOW = dt.timedelta(days=366)
# Exogenous series names (stable identifiers used by ExogenousRepository).
_LOAD_FORECAST = "load_forecast"
_WIND_SOLAR_FORECAST = "wind_solar_forecast"

# Transient network/server errors worth retrying.
_TRANSIENT = (HTTPError, RequestsConnectionError, RequestsTimeout)

_retry = retry(
    reraise=True,
    retry=retry_if_exception_type(_TRANSIENT),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)


def _infer_resolution_minutes(index: pd.DatetimeIndex, default: int = 60) -> int:
    """Infer the market time unit (15 or 60 min) from a datetime index spacing."""
    if index is None or len(index) < 2:
        return default
    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        return default
    seconds = deltas.dt.total_seconds().min()
    minutes = int(round(seconds / 60.0))
    return minutes if minutes > 0 else default


def _to_utc_index(index: pd.Index) -> pd.DatetimeIndex:
    """Return a timezone-aware UTC DatetimeIndex (localising naive input)."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize(dt.UTC)
    else:
        idx = idx.tz_convert("UTC")
    return idx


def _day_bounds(start: dt.date, end: dt.date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Build the inclusive [start 00:00, end+1 00:00) tz-aware query window."""
    start_ts = pd.Timestamp(start, tz=_QUERY_TZ)
    # end is treated as an inclusive delivery day -> query up to the next midnight.
    end_ts = pd.Timestamp(end + dt.timedelta(days=1), tz=_QUERY_TZ)
    return start_ts, end_ts


def _iter_chunks(
    start: pd.Timestamp, end: pd.Timestamp
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split [start, end) into <=1-year windows to respect the ENTSO-E cap."""
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor < end:
        nxt = min(cursor + _MAX_WINDOW, end)
        chunks.append((cursor, nxt))
        cursor = nxt
    return chunks


class EntsoeClient:
    """Thin wrapper over ``EntsoePandasClient`` for the Italian day-ahead market."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        token = self._settings.entsoe_api_token
        if not token:
            raise ValueError(
                "ENTSO-E API token missing; set ENERGY_ENTSOE_API_TOKEN "
                "(settings.has_entsoe is False)."
            )
        self._client = EntsoePandasClient(api_key=token)

    # --- low-level retried queries -------------------------------------------------

    @_retry
    def _query_prices(
        self, code: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series:
        return self._client.query_day_ahead_prices(code, start=start, end=end)

    @_retry
    def _query_load_forecast(
        self, code: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame | pd.Series:
        return self._client.query_load_forecast(code, start=start, end=end)

    @_retry
    def _query_wind_solar(
        self, code: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        return self._client.query_wind_and_solar_forecast(code, start=start, end=end)

    # --- high-level fetchers -------------------------------------------------------

    def fetch_prices(self, start: dt.date, end: dt.date) -> list[dict[str, Any]]:
        """Fetch day-ahead zonal prices for all 7 market zones.

        Returns a list of PriceObservation-shaped dicts (one per zone/interval).
        """
        rows: list[dict[str, Any]] = []
        win_start, win_end = _day_bounds(start, end)
        for zone in MARKET_ZONES:
            code = ENTSOE_ZONE_CODE[zone]
            series = self._collect_price_series(code, win_start, win_end)
            if series is None or series.empty:
                logger.warning("No ENTSO-E prices returned for zone %s", zone.value)
                continue
            rows.extend(self._price_rows(zone, series))
        return rows

    def _collect_price_series(
        self, code: str, win_start: pd.Timestamp, win_end: pd.Timestamp
    ) -> pd.Series | None:
        parts: list[pd.Series] = []
        for c_start, c_end in _iter_chunks(win_start, win_end):
            try:
                part = self._query_prices(code, c_start, c_end)
            except Exception as exc:  # noqa: BLE001 - log & continue per chunk
                logger.warning(
                    "ENTSO-E price query failed for %s [%s..%s]: %s",
                    code,
                    c_start.date(),
                    c_end.date(),
                    exc,
                )
                continue
            if part is not None and not part.empty:
                parts.append(part)
        if not parts:
            return None
        combined = pd.concat(parts)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        return combined

    @staticmethod
    def _price_rows(zone: Zone, series: pd.Series) -> list[dict[str, Any]]:
        idx_utc = _to_utc_index(series.index)
        res = _infer_resolution_minutes(idx_utc)
        rows: list[dict[str, Any]] = []
        for ts, value in zip(idx_utc, series.to_numpy(), strict=False):
            if pd.isna(value):
                continue
            rows.append(
                {
                    "market": Market.ELEC_DAYAHEAD.value,
                    "zone": zone.value,
                    "delivery_start": ts.to_pydatetime(),
                    "resolution_minutes": res,
                    "price": float(value),
                    "source": SOURCE,
                }
            )
        return rows

    def fetch_exogenous(self, start: dt.date, end: dt.date) -> list[dict[str, Any]]:
        """Best-effort fetch of load and wind+solar day-ahead forecasts per zone.

        Each driver is wrapped independently: a failure for one zone/series is
        logged and skipped so it never blocks price ingestion.
        """
        rows: list[dict[str, Any]] = []
        win_start, win_end = _day_bounds(start, end)
        for zone in MARKET_ZONES:
            code = ENTSOE_ZONE_CODE[zone]
            rows.extend(self._fetch_load(zone, code, win_start, win_end))
            rows.extend(self._fetch_wind_solar(zone, code, win_start, win_end))
        return rows

    def _fetch_load(
        self, zone: Zone, code: str, win_start: pd.Timestamp, win_end: pd.Timestamp
    ) -> list[dict[str, Any]]:
        series = self._collect_exog(
            self._query_load_forecast, code, win_start, win_end, label="load_forecast"
        )
        if series is None or series.empty:
            return []
        return self._exog_rows(zone, series, _LOAD_FORECAST, unit="MW")

    def _fetch_wind_solar(
        self, zone: Zone, code: str, win_start: pd.Timestamp, win_end: pd.Timestamp
    ) -> list[dict[str, Any]]:
        df = self._collect_exog(
            self._query_wind_solar,
            code,
            win_start,
            win_end,
            label="wind_solar_forecast",
        )
        if df is None or (hasattr(df, "empty") and df.empty):
            return []
        series = self._sum_wind_solar(df)
        if series is None or series.empty:
            return []
        return self._exog_rows(zone, series, _WIND_SOLAR_FORECAST, unit="MW")

    @staticmethod
    def _sum_wind_solar(df: pd.DataFrame | pd.Series) -> pd.Series | None:
        """Sum wind + solar columns into a single generation-forecast series."""
        if isinstance(df, pd.Series):
            return df
        if df.empty:
            return None
        # query_wind_and_solar_forecast yields columns like
        # 'Solar', 'Wind Onshore', 'Wind Offshore'.
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            return None
        return numeric.sum(axis=1, skipna=True)

    def _collect_exog(
        self,
        query_fn,
        code: str,
        win_start: pd.Timestamp,
        win_end: pd.Timestamp,
        *,
        label: str,
    ):
        parts = []
        for c_start, c_end in _iter_chunks(win_start, win_end):
            try:
                part = query_fn(code, c_start, c_end)
            except Exception as exc:  # noqa: BLE001 - best-effort exogenous fetch
                logger.warning(
                    "ENTSO-E %s query failed for %s [%s..%s]: %s",
                    label,
                    code,
                    c_start.date(),
                    c_end.date(),
                    exc,
                )
                continue
            if part is not None and len(part) > 0:
                parts.append(part)
        if not parts:
            return None
        combined = pd.concat(parts)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        return combined

    @staticmethod
    def _exog_rows(
        zone: Zone, series: pd.Series, name: str, *, unit: str
    ) -> list[dict[str, Any]]:
        idx_utc = _to_utc_index(series.index)
        res = _infer_resolution_minutes(idx_utc)
        rows: list[dict[str, Any]] = []
        for ts, value in zip(idx_utc, series.to_numpy(), strict=False):
            if pd.isna(value):
                continue
            rows.append(
                {
                    "series": name,
                    "zone": zone.value,
                    "valid_start": ts.to_pydatetime(),
                    "resolution_minutes": res,
                    "value": float(value),
                    "source": SOURCE,
                    "unit": unit,
                }
            )
        return rows

    @staticmethod
    def reconstruct_pun(price_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build an approximate PUN-like index from the zonal price rows.

        Volume-weighted average using PUN_ZONE_WEIGHTS. This is only a proxy for
        the official GME PUN Index and is tagged with zone=Zone.PUN / source=entsoe
        so it is clearly distinguishable from any GME-sourced PUN.
        """
        if not price_rows:
            return []
        df = pd.DataFrame(price_rows)
        # Keep only physical zones with a known weight.
        df = df[df["zone"].isin({z.value for z in MARKET_ZONES})]
        if df.empty:
            return []
        weights = {z.value: PUN_ZONE_WEIGHTS[z] for z in MARKET_ZONES}
        df = df.assign(_w=df["zone"].map(weights))
        df = df.dropna(subset=["_w", "price"])
        df["_wp"] = df["price"] * df["_w"]

        grouped = df.groupby("delivery_start")
        agg = grouped.agg(
            wp_sum=("_wp", "sum"),
            w_sum=("_w", "sum"),
            resolution_minutes=("resolution_minutes", "max"),
        )
        agg = agg[agg["w_sum"] > 0]
        if agg.empty:
            return []
        agg["pun"] = agg["wp_sum"] / agg["w_sum"]

        rows: list[dict[str, Any]] = []
        for delivery_start, row in agg.iterrows():
            rows.append(
                {
                    "market": Market.ELEC_DAYAHEAD.value,
                    "zone": Zone.PUN.value,
                    "delivery_start": delivery_start,
                    "resolution_minutes": int(row["resolution_minutes"]),
                    "price": float(row["pun"]),
                    "source": SOURCE,
                }
            )
        return rows


def ingest(session, start: dt.date, end: dt.date) -> int:
    """Fetch ENTSO-E prices + exogenous forecasts and upsert them.

    Builds repositories from ``session``, ingests zonal day-ahead prices, an
    approximate PUN index, and the load / wind+solar day-ahead forecasts.
    Returns the total number of rows upserted. If no ENTSO-E token is
    configured, logs a warning and returns 0 without touching the database.
    """
    settings = get_settings()
    if not settings.has_entsoe:
        logger.warning(
            "ENTSO-E token not configured (settings.has_entsoe is False); "
            "skipping ENTSO-E ingestion."
        )
        return 0

    client = EntsoeClient(settings)
    price_repo = PriceRepository(session)
    exog_repo = ExogenousRepository(session)

    total = 0

    price_rows = client.fetch_prices(start, end)
    if price_rows:
        total += price_repo.upsert(price_rows)

    pun_rows = client.reconstruct_pun(price_rows)
    if pun_rows:
        total += price_repo.upsert(pun_rows)

    try:
        exog_rows = client.fetch_exogenous(start, end)
    except Exception as exc:  # noqa: BLE001 - exogenous is best-effort
        logger.warning("ENTSO-E exogenous ingestion failed: %s", exc)
        exog_rows = []
    if exog_rows:
        total += exog_repo.upsert(exog_rows)

    logger.info(
        "ENTSO-E ingestion complete: %d price rows, %d PUN rows, %d exogenous rows "
        "(total %d) for %s..%s",
        len(price_rows),
        len(pun_rows),
        len(exog_rows),
        total,
        start,
        end,
    )
    return total
