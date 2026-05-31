"""Storage upsert/dedup + 15-min electricity end-to-end tests.

Covers the repository on-conflict path (idempotency, chunking, multi-source
coexistence) and the project's flagship 2025-reform feature — the 15-minute PUN
resolution — which the (hourly) demo seed never exercises.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Throwaway SQLite DB with a freshly created schema (caches reset)."""
    monkeypatch.setenv("ENERGY_DATABASE_URL", f"sqlite:///{(tmp_path / 't.db').as_posix()}")
    monkeypatch.setenv("ENERGY_DEMO_MODE", "true")
    from energy_prices.config import settings as settings_mod
    from energy_prices.storage import db as db_mod

    for fn in (settings_mod.get_settings, db_mod.get_engine, db_mod._session_factory):
        fn.cache_clear()
    db_mod.init_db()
    yield
    for fn in (settings_mod.get_settings, db_mod.get_engine, db_mod._session_factory):
        fn.cache_clear()


def _price_row(zone, ds, price, source="s", market="elec_dayahead"):
    return {
        "market": market, "zone": zone, "delivery_start": ds,
        "resolution_minutes": 60, "price": price, "source": source,
    }


# --- Repository upsert / dedup ---------------------------------------------
def test_price_upsert_idempotent_last_write_wins(tmp_db):
    """Re-upserting the same (market,zone,delivery_start,source) updates in place."""
    from energy_prices.storage.db import session_scope
    from energy_prices.storage.repositories import PriceRepository

    ds = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    with session_scope() as s:
        repo = PriceRepository(s)
        repo.upsert([_price_row("PUN", ds, 10.0)])
        repo.upsert([_price_row("PUN", ds, 20.0)])  # same key -> update
    with session_scope() as s:
        df = PriceRepository(s).get_prices("elec_dayahead", zone="PUN")
    assert len(df) == 1
    assert float(df["price"].iloc[0]) == 20.0


def test_price_upsert_distinct_sources_coexist(tmp_db):
    """Same delivery_start from two sources are kept as distinct rows."""
    from energy_prices.storage.db import session_scope
    from energy_prices.storage.repositories import PriceRepository

    ds = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    with session_scope() as s:
        repo = PriceRepository(s)
        repo.upsert([_price_row("PUN", ds, 10.0, source="gme")])
        repo.upsert([_price_row("PUN", ds, 11.0, source="entsoe")])
    with session_scope() as s:
        gme = PriceRepository(s).get_prices("elec_dayahead", zone="PUN", source="gme")
        ent = PriceRepository(s).get_prices("elec_dayahead", zone="PUN", source="entsoe")
    assert len(gme) == 1 and len(ent) == 1
    assert float(gme["price"].iloc[0]) == 10.0 and float(ent["price"].iloc[0]) == 11.0


def test_price_upsert_chunks_over_400_all_persist(tmp_db):
    """The 400-row chunking persists every row of a larger batch."""
    from energy_prices.storage.db import session_scope
    from energy_prices.storage.repositories import PriceRepository

    base = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    rows = [_price_row("PUN", base + dt.timedelta(hours=h), 50.0 + h) for h in range(500)]
    with session_scope() as s:
        PriceRepository(s).upsert(rows)
    with session_scope() as s:
        df = PriceRepository(s).get_prices("elec_dayahead", zone="PUN")
    assert len(df) == 500


def test_forecast_upsert_idempotent(tmp_db):
    """ForecastRepository.save de-dups on its full natural key (non-null zone)."""
    from energy_prices.storage.db import session_scope
    from energy_prices.storage.repositories import ForecastRepository

    run = dt.datetime(2026, 5, 30, tzinfo=dt.UTC)
    target = dt.datetime(2026, 5, 31, tzinfo=dt.UTC)

    def row(value):
        return {
            "run_at": run, "market": "elec_dayahead", "zone": "PUN",
            "target_start": target, "resolution_minutes": 60,
            "model_name": "m", "quantile": 0.5, "value": value,
        }

    with session_scope() as s:
        repo = ForecastRepository(s)
        repo.save([row(100.0)])
        repo.save([row(150.0)])
    with session_scope() as s:
        fc = ForecastRepository(s).get_forecasts("elec_dayahead", zone="PUN")
    assert len(fc) == 1 and float(fc["q0.5"].iloc[0]) == 150.0


@pytest.mark.xfail(
    reason="KNOWN BUG: NULL zones are treated as distinct in the unique index "
    "(both SQLite and Postgres), so gas/TTF (zone=None) re-ingestion duplicates "
    "rather than updates. Reads de-dup via keep='last', so forecasts stay correct, "
    "but the table bloats. Fix needs a zone sentinel or a COALESCE(zone,'') unique "
    "index — a schema decision flagged for review.",
    strict=True,
)
def test_price_upsert_null_zone_should_dedup(tmp_db):
    from energy_prices.storage.db import session_scope
    from energy_prices.storage.repositories import PriceRepository

    ds = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    with session_scope() as s:
        repo = PriceRepository(s)
        repo.upsert([_price_row(None, ds, 10.0, market="gas_dayahead")])
        repo.upsert([_price_row(None, ds, 20.0, market="gas_dayahead")])
    with session_scope() as s:
        df = PriceRepository(s).get_prices("gas_dayahead")
    assert len(df) == 1  # fails today: two NULL-zone rows coexist


# --- 15-minute electricity end-to-end (the 2025 reform feature) ------------
def test_15min_electricity_end_to_end(tmp_db):
    """Seed 15-min PUN, run the forecast, and assert a 96-step/day, 15-min forecast."""
    import sqlalchemy as sa

    from energy_prices.config import Market, Zone
    from energy_prices.forecasting.runner import run_forecasts
    from energy_prices.storage.db import session_scope
    from energy_prices.storage.models import Forecast
    from energy_prices.storage.repositories import ForecastRepository, PriceRepository

    # 20 days of 15-min PUN, ending yesterday (so "next day" horizon is well-defined).
    end = pd.Timestamp(dt.datetime.now(dt.UTC).date(), tz="UTC")
    idx = pd.date_range(end - pd.Timedelta(days=20), end, freq="15min", inclusive="left", tz="UTC")
    shape = 110 + 25 * np.sin(np.arange(len(idx)) / 96.0 * 2 * np.pi)  # daily cycle
    rows = [
        {
            "market": Market.ELEC_DAYAHEAD.value, "zone": Zone.PUN.value,
            "delivery_start": ts.to_pydatetime(), "resolution_minutes": 15,
            "price": float(p), "source": "test",
        }
        for ts, p in zip(idx, shape)
    ]
    with session_scope() as s:
        PriceRepository(s).upsert(rows)

    saved = run_forecasts(Market.ELEC_DAYAHEAD.value, Zone.PUN.value)
    assert saved > 0

    with session_scope() as s:
        fc = ForecastRepository(s).get_forecasts(Market.ELEC_DAYAHEAD.value, zone=Zone.PUN.value)
        res = s.execute(
            sa.select(Forecast.resolution_minutes)
            .where(Forecast.market == Market.ELEC_DAYAHEAD.value, Forecast.zone == Zone.PUN.value)
            .limit(1)
        ).scalar_one()
    assert res == 15  # forecast emitted on the 15-min grid
    assert len(fc) == 96  # one full delivery day = 96 quarter-hours
