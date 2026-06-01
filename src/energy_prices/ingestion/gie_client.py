"""GIE AGSI+ gas-storage ingestion client (Italy).

Pulls daily aggregated gas-storage data for Italy from the GIE AGSI+ Transparency
Platform and maps it onto exogenous driver series for the forecasting models.
A free API key is required (request it at https://agsi.gie.eu/account); supply it
via ``settings.agsi_api_key`` (env ``ENERGY_AGSI_API_KEY``).

API reference
-------------
- Base URL: ``https://agsi.gie.eu/api``.
- Auth: every request sends the header ``x-key: <api_key>``.
- Country query: ``GET /?country=IT&from=YYYY-MM-DD&to=YYYY-MM-DD&page=N&size=S``
  returns the aggregated country (national) storage time series for Italy.
- The response is a paged JSON envelope::

      {
        "last_page": <int>,    # total number of pages for the query
        "page": <int>,         # current 1-based page
        "size": <int>,         # page size requested
        "data": [ {<daily record>}, ... ]
      }

JSON fields consumed from each ``data`` record
----------------------------------------------
- ``gasDayStart``  -> the gas day, ``"YYYY-MM-DD"``. Interpreted as 00:00 UTC and
  used as ``valid_start`` (resolution 1440 min / daily).
- ``full``         -> storage fullness as a percentage (0-100). Mapped to series
  ``gas_storage_pct`` (unit ``"%"``).
- ``withdrawal``   -> daily withdrawal in GWh/d.
- ``injection``    -> daily injection in GWh/d.
  When both are present, the net withdrawal ``withdrawal - injection`` is mapped to
  series ``gas_storage_net_withdrawal`` (unit ``"GWh/d"``), positive when storage is
  being drawn down (winter) and negative when it is being filled (summer).

All numeric fields arrive as strings (or ``"-"`` / empty when unavailable) and are
parsed defensively; records with an unparseable fullness are skipped.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from energy_prices.config import get_settings
from energy_prices.storage.repositories import ExogenousRepository, IngestionRepository

logger = logging.getLogger(__name__)

SOURCE = "agsi"

API_BASE_URL = "https://agsi.gie.eu/api"
COUNTRY_CODE = "IT"
PAGE_SIZE = 300  # AGSI+ maximum page size
REQUEST_TIMEOUT = 30  # seconds
MAX_PAGES = 200  # hard safety cap to avoid runaway pagination

SERIES_STORAGE_PCT = "gas_storage_pct"
SERIES_NET_WITHDRAWAL = "gas_storage_net_withdrawal"
RESOLUTION_MINUTES = 1440  # daily

def _is_retryable(exc: BaseException) -> bool:
    """Retry transient network errors and HTTP 429/5xx; fail fast on other 4xx."""
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        resp = getattr(exc, "response", None)
        if resp is None:
            return True  # no response attached -> treat as transient
        return resp.status_code == 429 or resp.status_code >= 500
    return False


class GieClient:
    """Thin HTTP client for the GIE AGSI+ country storage endpoint."""

    def __init__(self, api_key: str, base_url: str = API_BASE_URL) -> None:
        if not api_key:
            raise ValueError("GieClient requires a non-empty AGSI+ api_key.")
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update(
            {
                "x-key": api_key,
                "Accept": "application/json",
                "User-Agent": "energy-prices/0.1 (+agsi)",
            }
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception(_is_retryable),
    )
    def _get_page(self, start: dt.date, end: dt.date, page: int) -> dict[str, Any]:
        """Fetch a single page of the Italy storage series."""
        params: dict[str, Any] = {
            "country": COUNTRY_CODE,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "page": page,
            "size": PAGE_SIZE,
        }
        response = self._session.get(
            f"{self._base_url}/", params=params, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected AGSI+ payload type: {type(payload)!r}")
        return payload

    def iter_records(self, start: dt.date, end: dt.date) -> list[dict[str, Any]]:
        """Return all daily storage records for Italy in [start, end], paginated."""
        records: list[dict[str, Any]] = []
        page = 1
        while page <= MAX_PAGES:
            payload = self._get_page(start, end, page)
            data = payload.get("data") or []
            if isinstance(data, dict):  # defensive: some envelopes nest under data
                data = data.get("data") or []
            records.extend(r for r in data if isinstance(r, dict))

            last_page = _to_int(payload.get("last_page"), default=page)
            if not data or page >= last_page:
                break
            page += 1
        else:  # pragma: no cover - only when MAX_PAGES is exceeded
            logger.warning("AGSI+ pagination hit MAX_PAGES=%d; results may be truncated.", MAX_PAGES)
        return records

    def close(self) -> None:
        self._session.close()


def _to_float(value: Any) -> float | None:
    """Parse an AGSI+ numeric string; return None for missing/sentinel values."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if text in ("", "-", "N/A", "n/a", "null", "None"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(value: Any, default: int) -> int:
    parsed = _to_float(value)
    return int(parsed) if parsed is not None else default


def _gas_day_start_utc(value: Any) -> dt.datetime | None:
    """Convert a ``gasDayStart`` date string to a tz-aware UTC midnight datetime."""
    if not value:
        return None
    text = str(value).strip()
    try:
        day = dt.date.fromisoformat(text[:10])
    except ValueError:
        return None
    return dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC)


def _records_to_observations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map raw AGSI+ records to ExogenousObservation upsert dicts."""
    observations: list[dict[str, Any]] = []
    for rec in records:
        valid_start = _gas_day_start_utc(rec.get("gasDayStart"))
        if valid_start is None:
            continue

        full_pct = _to_float(rec.get("full"))
        if full_pct is not None:
            observations.append(
                {
                    "series": SERIES_STORAGE_PCT,
                    "zone": None,
                    "valid_start": valid_start,
                    "resolution_minutes": RESOLUTION_MINUTES,
                    "value": full_pct,
                    "unit": "%",
                    "source": SOURCE,
                }
            )

        withdrawal = _to_float(rec.get("withdrawal"))
        injection = _to_float(rec.get("injection"))
        if withdrawal is not None and injection is not None:
            observations.append(
                {
                    "series": SERIES_NET_WITHDRAWAL,
                    "zone": None,
                    "valid_start": valid_start,
                    "resolution_minutes": RESOLUTION_MINUTES,
                    "value": withdrawal - injection,
                    "unit": "GWh/d",
                    "source": SOURCE,
                }
            )
    return observations


def ingest(session: Any, start: dt.date, end: dt.date) -> int:
    """Ingest GIE AGSI+ Italian gas-storage data for [start, end] into exogenous series.

    Parameters
    ----------
    session:
        An open SQLAlchemy Session; the caller owns its transaction/commit.
    start, end:
        Inclusive gas-day date range to fetch.

    Returns
    -------
    int
        Number of exogenous observation rows upserted (0 if no key configured or
        nothing was returned).
    """
    settings = get_settings()
    api_key = settings.agsi_api_key
    if not api_key:
        logger.warning(
            "AGSI+ ingestion skipped: settings.agsi_api_key is not set "
            "(set ENERGY_AGSI_API_KEY). Returning 0 rows."
        )
        return 0

    if end < start:
        logger.warning("AGSI+ ingest called with end (%s) before start (%s); swapping.", end, start)
        start, end = end, start

    exog_repo = ExogenousRepository(session)
    ingestion_repo = IngestionRepository(session)
    run = ingestion_repo.start(SOURCE, dt.datetime.now(dt.UTC))

    client = GieClient(api_key)
    try:
        records = client.iter_records(start, end)
        observations = _records_to_observations(records)
        rows = exog_repo.upsert(observations)
        ingestion_repo.finish(
            run,
            status="success",
            finished_at=dt.datetime.now(dt.UTC),
            rows=rows,
            message=f"AGSI+ IT storage {start}..{end}: {len(records)} records -> {rows} rows",
        )
        logger.info(
            "AGSI+ ingestion complete: %s..%s, %d records, %d exogenous rows.",
            start,
            end,
            len(records),
            rows,
        )
        return rows
    except Exception as exc:
        ingestion_repo.finish(
            run,
            status="error",
            finished_at=dt.datetime.now(dt.UTC),
            rows=0,
            message=f"AGSI+ ingestion failed: {exc}",
        )
        logger.exception("AGSI+ ingestion failed for %s..%s", start, end)
        raise
    finally:
        client.close()
