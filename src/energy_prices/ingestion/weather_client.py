"""Open-Meteo weather client: population-weighted Italian temperature + degree days.

Temperature is a primary driver of gas (heating) and, in summer, power (cooling)
demand. This client builds a national, population-weighted daily mean 2 m air
temperature for Italy and derives heating/cooling degree days (HDD/CDD), storing
them as exogenous series for the demand-aware forecasting models.

Source: the FREE Open-Meteo **Historical Forecast API**
(https://historical-forecast-api.open-meteo.com/v1/forecast). We deliberately use
the *historical forecast* archive (the past runs of the operational forecast
model) rather than ERA5 reanalysis: at prediction time the model only ever had
access to a forecast, never the reanalysed truth, so training on forecasts keeps
the feature leak-safe and consistent with what is available for future horizons.

No API key is required. NOTE: the Open-Meteo free tier is for **non-commercial
use** only; a commercial subscription / self-hosted instance is required for
production commercial use.

Exogenous series produced (zone=None, daily, valid_start = day 00:00 UTC):
    - ``temp_pop_it`` : population-weighted daily mean temperature (degC)
    - ``hdd``         : heating degree days, max(0, 18 - tavg) (degC-day)
    - ``cdd``         : cooling degree days, max(0, tavg - 21) (degC-day)
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Any

import requests
from sqlalchemy.orm import Session

from energy_prices.storage.repositories import (
    ExogenousRepository,
    IngestionRepository,
)

logger = logging.getLogger(__name__)

SOURCE = "open-meteo"
_API_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
_RESOLUTION_MINUTES = 1440  # daily
_TIMEOUT = 60  # seconds

# Heating / cooling degree-day base temperatures (degC), standard EU conventions.
HDD_BASE = 18.0
CDD_BASE = 21.0


@dataclass(frozen=True)
class City:
    """A reference city with coordinates and a rough demand (population) weight."""

    name: str
    latitude: float
    longitude: float
    weight: float


# A handful of major Italian cities spanning the peninsula, with rough population
# weights (normalised below). Not exhaustive — a deliberately cheap proxy for the
# national population-weighted temperature, good enough as a demand driver.
CITIES: tuple[City, ...] = (
    City("Milano", 45.46, 9.19, 0.30),
    City("Roma", 41.90, 12.50, 0.30),
    City("Napoli", 40.85, 14.27, 0.18),
    City("Torino", 45.07, 7.69, 0.12),
    City("Palermo", 38.12, 13.36, 0.10),
)


def _normalised_weights(cities: tuple[City, ...]) -> dict[str, float]:
    total = sum(c.weight for c in cities)
    if total <= 0:
        raise ValueError("City weights must sum to a positive value.")
    return {c.name: c.weight / total for c in cities}


def _fetch_city_daily_mean(
    city: City, start: dt.date, end: dt.date, session: requests.Session
) -> dict[dt.date, float]:
    """Fetch daily mean 2 m temperature (degC) for one city, keyed by date.

    Returns an empty dict if the API yields no usable data for the window.
    """
    params: dict[str, Any] = {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "temperature_2m_mean",
        "timezone": "UTC",
    }
    response = session.get(_API_URL, params=params, timeout=_TIMEOUT)
    response.raise_for_status()
    payload = response.json()

    daily = payload.get("daily") or {}
    days = daily.get("time") or []
    temps = daily.get("temperature_2m_mean") or []

    result: dict[dt.date, float] = {}
    for day_str, temp in zip(days, temps):
        if temp is None:
            continue
        try:
            day = dt.date.fromisoformat(day_str)
        except (TypeError, ValueError):
            logger.warning("Skipping unparseable date %r for %s", day_str, city.name)
            continue
        result[day] = float(temp)
    return result


def _population_weighted(
    per_city: dict[str, dict[dt.date, float]], weights: dict[str, float]
) -> dict[dt.date, float]:
    """Combine per-city daily means into a population-weighted national mean.

    For each day, weights are renormalised over the cities that actually reported
    data, so a single missing city does not bias the average.
    """
    all_days: set[dt.date] = set()
    for series in per_city.values():
        all_days.update(series.keys())

    weighted: dict[dt.date, float] = {}
    for day in sorted(all_days):
        num = 0.0
        denom = 0.0
        for name, series in per_city.items():
            value = series.get(day)
            if value is None:
                continue
            w = weights[name]
            num += w * value
            denom += w
        if denom > 0:
            weighted[day] = num / denom
    return weighted


def _midnight_utc(day: dt.date) -> dt.datetime:
    return dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC)


def _build_rows(weighted: dict[dt.date, float]) -> list[dict[str, object]]:
    """Build ExogenousRepository rows for temp_pop_it, hdd and cdd."""
    rows: list[dict[str, object]] = []
    for day in sorted(weighted):
        tavg = weighted[day]
        hdd = max(0.0, HDD_BASE - tavg)
        cdd = max(0.0, tavg - CDD_BASE)
        valid_start = _midnight_utc(day)
        rows.append(
            {
                "series": "temp_pop_it",
                "zone": None,
                "valid_start": valid_start,
                "resolution_minutes": _RESOLUTION_MINUTES,
                "value": tavg,
                "source": SOURCE,
                "unit": "degC",
            }
        )
        rows.append(
            {
                "series": "hdd",
                "zone": None,
                "valid_start": valid_start,
                "resolution_minutes": _RESOLUTION_MINUTES,
                "value": hdd,
                "source": SOURCE,
                "unit": "degC-day",
            }
        )
        rows.append(
            {
                "series": "cdd",
                "zone": None,
                "valid_start": valid_start,
                "resolution_minutes": _RESOLUTION_MINUTES,
                "value": cdd,
                "source": SOURCE,
                "unit": "degC-day",
            }
        )
    return rows


def ingest(session: Session, start: dt.date, end: dt.date) -> int:
    """Ingest population-weighted Italian temperature, HDD and CDD into the DB.

    Fetches daily mean 2 m temperature for the reference cities from the
    Open-Meteo Historical Forecast API over ``[start, end]`` (inclusive),
    computes the population-weighted national mean and the derived degree days,
    and upserts them as exogenous series via :class:`ExogenousRepository`.

    Returns the number of exogenous rows upserted (3 series per day with data).
    """
    if end < start:
        raise ValueError(f"end ({end}) must not precede start ({start}).")

    ingestion_repo = IngestionRepository(session)
    run = ingestion_repo.start(SOURCE, dt.datetime.now(dt.UTC))

    weights = _normalised_weights(CITIES)
    try:
        per_city: dict[str, dict[dt.date, float]] = {}
        with requests.Session() as http:
            for city in CITIES:
                try:
                    per_city[city.name] = _fetch_city_daily_mean(
                        city, start, end, http
                    )
                except requests.RequestException as exc:
                    logger.warning(
                        "Open-Meteo fetch failed for %s (%s); skipping city.",
                        city.name,
                        exc,
                    )
                    per_city[city.name] = {}

        weighted = _population_weighted(per_city, weights)
        rows = _build_rows(weighted)

        repo = ExogenousRepository(session)
        n_rows = repo.upsert(rows)

        ingestion_repo.finish(
            run,
            status="success",
            finished_at=dt.datetime.now(dt.UTC),
            rows=n_rows,
            message=(
                f"{len(weighted)} days {start}..{end}; "
                f"series temp_pop_it/hdd/cdd from {SOURCE}"
            ),
        )
        logger.info(
            "Ingested %d weather rows (%d days) from %s for %s..%s.",
            n_rows,
            len(weighted),
            SOURCE,
            start,
            end,
        )
        return n_rows
    except Exception as exc:
        ingestion_repo.finish(
            run,
            status="error",
            finished_at=dt.datetime.now(dt.UTC),
            rows=0,
            message=str(exc),
        )
        logger.exception("Weather ingestion from %s failed.", SOURCE)
        raise
