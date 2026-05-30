"""One-shot migration: copy all data from the local SQLite DB into Postgres+Timescale.

For the shared (colleagues) deployment we move from the zero-setup SQLite file to
the Postgres+TimescaleDB system-of-record. This script copies every table
(price_observations, forecasts, exogenous_observations, ingestion_runs) row for
row, in batches, into a freshly initialised Postgres schema.

Idempotent-ish: the destination is created with the same UNIQUE constraints, and
inserts use ON CONFLICT DO NOTHING (Postgres), so re-running skips duplicates.

Usage (from repo root, venv active):

    # dest defaults to the docker-compose Postgres; source to ./data/energy_prices.db
    python scripts/migrate_sqlite_to_postgres.py \
        --source sqlite:///./data/energy_prices.db \
        --dest postgresql+psycopg2://energy:energy@localhost:5432/energy

Requires the `postgres` extra: pip install -e ".[postgres]".
"""

from __future__ import annotations

import argparse
import logging

from sqlalchemy import create_engine, insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

# Import via the package so truststore TLS + settings load identically.
from energy_prices.storage.models import (
    Base,
    ExogenousObservation,
    Forecast,
    IngestionRun,
    PriceObservation,
)

logger = logging.getLogger("migrate")

# Copy order is irrelevant (no FKs between tables) but kept stable for clear logs.
TABLES = [PriceObservation, Forecast, ExogenousObservation, IngestionRun]
_BATCH = 5_000


def _rows(engine, model):
    """Yield all rows of a table as plain dicts (column -> value)."""
    cols = [c.name for c in model.__table__.columns]
    with engine.connect() as conn:
        for row in conn.execute(select(model.__table__)).mappings():
            yield {c: row[c] for c in cols}


def _copy_table(src_engine, dst_engine, model, is_postgres_dst: bool) -> int:
    table = model.__table__
    batch: list[dict] = []
    total = 0

    def flush(rows: list[dict]) -> int:
        if not rows:
            return 0
        # Skip the autoincrement PK so the destination assigns its own ids.
        payload = [{k: v for k, v in r.items() if k != "id"} for r in rows]
        stmt = (
            pg_insert(table).on_conflict_do_nothing()
            if is_postgres_dst
            else insert(table)
        )
        with dst_engine.begin() as conn:
            conn.execute(stmt, payload)
        return len(payload)

    for row in _rows(src_engine, model):
        batch.append(row)
        if len(batch) >= _BATCH:
            total += flush(batch)
            batch = []
            logger.info("  %s: %d rows…", table.name, total)
    total += flush(batch)
    logger.info("%s: %d rows copied.", table.name, total)
    return total


def migrate(source_url: str, dest_url: str) -> dict[str, int]:
    src = create_engine(source_url, future=True)
    dst = create_engine(dest_url, future=True, pool_pre_ping=True)
    is_pg = dest_url.startswith("postgresql")

    logger.info("Creating destination schema at %s", dest_url)
    Base.metadata.create_all(dst)
    if is_pg:
        # Best-effort Timescale hypertables on the destination.
        try:
            from energy_prices.storage.db import _enable_timescale

            _enable_timescale(dst)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Timescale setup skipped: %s", exc)

    results: dict[str, int] = {}
    for model in TABLES:
        results[model.__tablename__] = _copy_table(src, dst, model, is_pg)
    logger.info("Migration complete: %s", results)
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Migrate SQLite -> Postgres+Timescale.")
    parser.add_argument(
        "--source", default="sqlite:///./data/energy_prices.db", help="Source SQLAlchemy URL."
    )
    parser.add_argument(
        "--dest",
        default="postgresql+psycopg2://energy:energy@localhost:5432/energy",
        help="Destination Postgres URL.",
    )
    args = parser.parse_args()
    migrate(args.source, args.dest)


if __name__ == "__main__":
    main()
