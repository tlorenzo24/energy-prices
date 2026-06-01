"""Repositories: the only sanctioned way to read/write prices & forecasts.

Ingestion clients call PriceRepository.upsert(...); the forecast runner calls
ForecastRepository.save(...); the dashboard reads via get_* methods that return
tidy pandas DataFrames (UTC-indexed). Models/dashboard never touch the ORM directly.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from energy_prices.storage.models import (
    ExogenousObservation,
    Forecast,
    IngestionRun,
    PriceObservation,
)

logger = logging.getLogger(__name__)


def _utc(value: dt.datetime | None) -> dt.datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.UTC)
    return value.astimezone(dt.UTC)


# National/zone-less series (gas, TTF, country-level exogenous) are stored with
# this non-null sentinel instead of NULL, so the unique indexes (which include
# `zone`) actually fire on re-ingest. SQL treats NULL != NULL, which would
# otherwise let an ON CONFLICT clause miss and duplicate rows pile up.
_NO_ZONE = ""


def _zone(zone: str | None) -> str:
    """Normalize a zone (None -> the no-zone sentinel) for storage and filtering."""
    return zone if zone else _NO_ZONE


# Max rows per INSERT statement. Chunked so (rows * columns) stays well under
# SQLite's bound-parameter limit (SQLITE_MAX_VARIABLE_NUMBER) on every version.
_UPSERT_CHUNK = 400


def _upsert(session: Session, table, rows: list[dict[str, Any]], index_elements: list[str],
            update_cols: list[str]) -> int:
    """Dialect-aware bulk upsert (SQLite & Postgres), chunked. Returns rows sent."""
    if not rows:
        return 0
    # Collapse within-batch duplicates on the conflict key (keep last). Postgres
    # ON CONFLICT DO UPDATE raises CardinalityViolation if one INSERT touches the
    # same conflict row twice; SQLite silently keeps the last. Dedup here so both
    # dialects behave identically and we never hit the prod-only failure.
    deduped: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        deduped[tuple(row.get(col) for col in index_elements)] = row
    rows = list(deduped.values())
    dialect = session.bind.dialect.name  # type: ignore[union-attr]
    insert = pg_insert if dialect == "postgresql" else sqlite_insert
    total = 0
    for i in range(0, len(rows), _UPSERT_CHUNK):
        chunk = rows[i : i + _UPSERT_CHUNK]
        stmt = insert(table).values(chunk)
        stmt = stmt.on_conflict_do_update(
            index_elements=index_elements,
            set_={c: getattr(stmt.excluded, c) for c in update_cols},
        )
        session.execute(stmt)
        total += len(chunk)
    return total


class PriceRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(self, observations: list[dict[str, Any]]) -> int:
        """Insert/update price observations.

        Each dict needs: market, zone, delivery_start (UTC datetime),
        resolution_minutes, price, source. Optional: currency, unit.
        """
        clean: list[dict[str, Any]] = []
        for obs in observations:
            row = dict(obs)
            row["delivery_start"] = _utc(row["delivery_start"])
            row["zone"] = _zone(row.get("zone"))
            row.setdefault("currency", "EUR")
            row.setdefault("unit", "EUR/MWh")
            clean.append(row)
        return _upsert(
            self.session,
            PriceObservation.__table__,
            clean,
            index_elements=["market", "zone", "delivery_start", "source"],
            update_cols=["price", "resolution_minutes", "currency", "unit"],
        )

    def get_prices(
        self,
        market: str,
        zone: str | None = None,
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        """Return prices as a DataFrame indexed by delivery_start (UTC)."""
        stmt = select(PriceObservation).where(
            PriceObservation.market == market,
            PriceObservation.zone == _zone(zone),
        )
        if source is not None:
            stmt = stmt.where(PriceObservation.source == source)
        if start is not None:
            stmt = stmt.where(PriceObservation.delivery_start >= _utc(start))
        if end is not None:
            stmt = stmt.where(PriceObservation.delivery_start <= _utc(end))
        # Secondary sort on ingested_at/id so duplicate timestamps resolve
        # deterministically and downstream `duplicated(keep="last")` keeps the
        # freshest row regardless of physical scan order (Postgres/Timescale).
        stmt = stmt.order_by(
            PriceObservation.delivery_start,
            PriceObservation.ingested_at,
            PriceObservation.id,
        )
        rows = self.session.execute(stmt).scalars().all()
        df = pd.DataFrame(
            [
                {
                    "delivery_start": _utc(r.delivery_start),
                    "market": r.market,
                    "zone": r.zone or None,
                    "resolution_minutes": r.resolution_minutes,
                    "price": r.price,
                    "unit": r.unit,
                    "source": r.source,
                }
                for r in rows
            ]
        )
        if not df.empty:
            df["delivery_start"] = pd.to_datetime(df["delivery_start"], utc=True)
            df = df.set_index("delivery_start")
        return df

    def latest_delivery(self, market: str, zone: str | None = None) -> dt.datetime | None:
        stmt = select(PriceObservation.delivery_start).where(
            PriceObservation.market == market,
            PriceObservation.zone == _zone(zone),
        )
        stmt = stmt.order_by(PriceObservation.delivery_start.desc()).limit(1)
        result = self.session.execute(stmt).scalar_one_or_none()
        return _utc(result)

    def distinct_zones(self, market: str) -> list[str]:
        stmt = (
            select(PriceObservation.zone)
            .where(PriceObservation.market == market)
            .distinct()
        )
        # Drop both None and the "" no-zone sentinel: only real zones are returned.
        return [z for (z,) in self.session.execute(stmt).all() if z]


class ExogenousRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(self, observations: list[dict[str, Any]]) -> int:
        """Insert/update exogenous series values.

        Each dict needs: series, zone (or None), valid_start (UTC),
        resolution_minutes, value, source. Optional: unit.
        """
        clean: list[dict[str, Any]] = []
        for obs in observations:
            row = dict(obs)
            row["valid_start"] = _utc(row["valid_start"])
            row["zone"] = _zone(row.get("zone"))
            row.setdefault("unit", None)
            clean.append(row)
        return _upsert(
            self.session,
            ExogenousObservation.__table__,
            clean,
            index_elements=["series", "zone", "valid_start", "source"],
            update_cols=["value", "resolution_minutes", "unit"],
        )

    def get_series(
        self,
        series: str,
        zone: str | None = None,
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
    ) -> pd.Series:
        """Return one exogenous series as a UTC-indexed pandas Series named `series`."""
        stmt = select(ExogenousObservation).where(
            ExogenousObservation.series == series,
            ExogenousObservation.zone == _zone(zone),
        )
        if start is not None:
            stmt = stmt.where(ExogenousObservation.valid_start >= _utc(start))
        if end is not None:
            stmt = stmt.where(ExogenousObservation.valid_start <= _utc(end))
        stmt = stmt.order_by(
            ExogenousObservation.valid_start,
            ExogenousObservation.ingested_at,
            ExogenousObservation.id,
        )
        rows = self.session.execute(stmt).scalars().all()
        if not rows:
            return pd.Series(dtype=float, name=series)
        idx = pd.to_datetime([_utc(r.valid_start) for r in rows], utc=True)
        return pd.Series([r.value for r in rows], index=idx, name=series)


class ForecastRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def save(self, forecasts: list[dict[str, Any]]) -> int:
        """Upsert forecast rows (one per quantile).

        Each dict needs: run_at, market, zone, target_start, resolution_minutes,
        model_name, quantile, value. Optional: model_version, unit.
        """
        clean: list[dict[str, Any]] = []
        for fc in forecasts:
            row = dict(fc)
            row["run_at"] = _utc(row["run_at"])
            row["target_start"] = _utc(row["target_start"])
            row["zone"] = _zone(row.get("zone"))
            row.setdefault("model_version", "0.1.0")
            row.setdefault("unit", "EUR/MWh")
            row.setdefault("quantile", 0.5)
            clean.append(row)
        return _upsert(
            self.session,
            Forecast.__table__,
            clean,
            index_elements=["run_at", "market", "zone", "target_start", "model_name", "quantile"],
            update_cols=["value", "resolution_minutes", "model_version", "unit"],
        )

    def latest_run_at(
        self, market: str, zone: str | None = None, model_name: str | None = None
    ) -> dt.datetime | None:
        stmt = select(Forecast.run_at).where(
            Forecast.market == market,
            Forecast.zone == _zone(zone),
        )
        if model_name is not None:
            stmt = stmt.where(Forecast.model_name == model_name)
        stmt = stmt.order_by(Forecast.run_at.desc()).limit(1)
        return _utc(self.session.execute(stmt).scalar_one_or_none())

    def get_forecasts(
        self,
        market: str,
        zone: str | None = None,
        model_name: str | None = None,
        run_at: dt.datetime | None = None,
        latest: bool = True,
    ) -> pd.DataFrame:
        """Return forecasts pivoted wide: index target_start (UTC), columns 'q0.1'..'q0.9'.

        If run_at is None and latest=True, uses the most recent run for the filter.
        """
        zone_db = _zone(zone)
        if run_at is None and latest:
            run_at = self.latest_run_at(market, zone, model_name)
            if run_at is None:
                return pd.DataFrame()

        # When no model is pinned, resolve the single model that owns this run so
        # the pivot never blends quantiles from two models sharing a run_at.
        if model_name is None and run_at is not None:
            owners = [
                m
                for (m,) in self.session.execute(
                    select(Forecast.model_name)
                    .where(
                        Forecast.market == market,
                        Forecast.zone == zone_db,
                        Forecast.run_at == _utc(run_at),
                    )
                    .distinct()
                ).all()
                if m
            ]
            if len(owners) > 1:
                model_name = self.session.execute(
                    select(Forecast.model_name)
                    .where(
                        Forecast.market == market,
                        Forecast.zone == zone_db,
                        Forecast.run_at == _utc(run_at),
                    )
                    .order_by(Forecast.id.desc())
                    .limit(1)
                ).scalar_one_or_none()
                logger.warning(
                    "Multiple models %s at run_at=%s for market=%s zone=%s; "
                    "selecting most recent (%s).",
                    owners, run_at, market, zone, model_name,
                )
            elif owners:
                model_name = owners[0]

        stmt = select(Forecast).where(
            Forecast.market == market,
            Forecast.zone == zone_db,
        )
        if model_name is not None:
            stmt = stmt.where(Forecast.model_name == model_name)
        if run_at is not None:
            stmt = stmt.where(Forecast.run_at == _utc(run_at))
        stmt = stmt.order_by(Forecast.target_start)
        rows = self.session.execute(stmt).scalars().all()
        if not rows:
            return pd.DataFrame()

        long = pd.DataFrame(
            [
                {
                    "target_start": _utc(r.target_start),
                    "model_name": r.model_name,
                    "quantile": r.quantile,
                    "value": r.value,
                }
                for r in rows
            ]
        )
        long["target_start"] = pd.to_datetime(long["target_start"], utc=True)
        wide = long.pivot_table(
            index="target_start", columns="quantile", values="value", aggfunc="last"
        )
        wide.columns = [f"q{c:g}" for c in wide.columns]
        return wide.sort_index()


class IngestionRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def start(self, source: str, started_at: dt.datetime) -> IngestionRun:
        run = IngestionRun(source=source, started_at=_utc(started_at), status="running")
        self.session.add(run)
        self.session.flush()
        return run

    def finish(
        self,
        run: IngestionRun,
        status: str,
        finished_at: dt.datetime,
        rows: int = 0,
        message: str | None = None,
    ) -> None:
        run.status = status
        run.finished_at = _utc(finished_at)
        run.rows_ingested = rows
        run.message = (message or "")[:512]
        self.session.add(run)
