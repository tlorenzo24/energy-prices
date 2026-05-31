"""Postgres + TimescaleDB integration tests (skipped unless on a Postgres URL).

These guard the prod-only path that SQLite can never exercise:
* the composite-PK widening + ``create_hypertable`` succeed (so Timescale
  partitioning/compression actually materialises), and
* the dialect-aware ``pg_insert`` on-conflict upsert de-dups correctly.

Locally they no-op (SQLite). In CI the workflow provides a real
``timescale/timescaledb`` service and sets ENERGY_DATABASE_URL to it.
"""

from __future__ import annotations

import datetime as dt
import os

import pytest

_PG = os.environ.get("ENERGY_DATABASE_URL", "").startswith("postgresql")
pytestmark = pytest.mark.skipif(not _PG, reason="requires a Postgres+Timescale ENERGY_DATABASE_URL")


def _reset_caches():
    from energy_prices.config import settings as settings_mod
    from energy_prices.storage import db as db_mod

    for fn in (settings_mod.get_settings, db_mod.get_engine, db_mod._session_factory):
        fn.cache_clear()


@pytest.fixture
def pg_db():
    """Initialise the schema (tables + hypertables) on the configured Postgres."""
    _reset_caches()
    from energy_prices.storage import db as db_mod

    db_mod.init_db()
    yield
    _reset_caches()


def test_timescale_hypertables_created(pg_db):
    """The PK widening lets create_hypertable succeed for both time-series tables."""
    from energy_prices.storage.db import get_engine, timescale_hypertable_count

    assert timescale_hypertable_count(get_engine()) == 2


def test_postgres_upsert_dedup_and_distinct_sources(pg_db):
    """pg_insert on-conflict updates in place (non-null zone) and keeps sources apart."""
    from energy_prices.storage.db import session_scope
    from energy_prices.storage.repositories import PriceRepository

    ds = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)

    def row(price, source="gme"):
        return {
            "market": "elec_dayahead", "zone": "PUN", "delivery_start": ds,
            "resolution_minutes": 60, "price": price, "source": source,
        }

    with session_scope() as s:
        repo = PriceRepository(s)
        repo.upsert([row(10.0)])
        repo.upsert([row(20.0)])          # same key -> update
        repo.upsert([row(11.0, "entsoe")])  # distinct source -> coexists
    with session_scope() as s:
        gme = PriceRepository(s).get_prices("elec_dayahead", zone="PUN", source="gme")
        ent = PriceRepository(s).get_prices("elec_dayahead", zone="PUN", source="entsoe")
    assert len(gme) == 1 and float(gme["price"].iloc[0]) == 20.0
    assert len(ent) == 1 and float(ent["price"].iloc[0]) == 11.0
