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
# The dedicated CI job sets ENERGY_REQUIRE_PG=1: there a missing/mistyped URL must
# FAIL (not silently skip), so the job can't go green having validated nothing.
_REQUIRE_PG = os.environ.get("ENERGY_REQUIRE_PG") == "1"
_skip_pg = pytest.mark.skipif(not _PG, reason="requires a Postgres+Timescale ENERGY_DATABASE_URL")


@pytest.mark.skipif(not _REQUIRE_PG, reason="only enforced in the dedicated Postgres CI job")
def test_ci_database_url_is_postgres():
    """Fails (not skips) if the Timescale URL is missing/mistyped in the PG CI job."""
    url = os.environ.get("ENERGY_DATABASE_URL", "")
    assert url.startswith("postgresql"), f"expected a Postgres URL, got {url!r}"


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


@_skip_pg
def test_timescale_hypertables_created(pg_db):
    """The PK widening lets create_hypertable succeed for both time-series tables."""
    from energy_prices.storage.db import get_engine, timescale_hypertable_count

    assert timescale_hypertable_count(get_engine()) == 2


@_skip_pg
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


@_skip_pg
def test_postgres_null_zone_dedup(pg_db):
    """National/gas series (zone=None) dedup on the real Postgres dialect too (the "" sentinel)."""
    from energy_prices.storage.db import session_scope
    from energy_prices.storage.repositories import PriceRepository

    ds = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)

    def row(price):
        return {
            "market": "gas_dayahead", "zone": None, "delivery_start": ds,
            "resolution_minutes": 1440, "price": price, "source": "ttf",
        }

    with session_scope() as s:
        repo = PriceRepository(s)
        repo.upsert([row(30.0)])
        repo.upsert([row(40.0)])  # same NULL-zone key -> update, not duplicate
    with session_scope() as s:
        df = PriceRepository(s).get_prices("gas_dayahead")
    assert len(df) == 1 and float(df["price"].iloc[0]) == 40.0
