"""Scheduler tests: the production entry point's contracts.

Covers the bits the daily Docker job depends on but that had no test evidence:
backfill chunk arithmetic, per-stage failure isolation in daily_job, the
weather opt-in gating of "all", and the first-boot demo bootstrap.
"""

from __future__ import annotations

import datetime as dt

import pytest

from energy_prices.ingestion import scheduler


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
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


# --- source resolution / weather opt-in -------------------------------------
def test_resolve_sources_excludes_weather_by_default(tmp_db):
    # _hermetic_env leaves ENERGY_ENABLE_WEATHER unset -> default False.
    sources = scheduler._resolve_sources("all")
    assert "weather" not in sources
    assert {"entsoe", "gme", "ttf", "gie"} <= set(sources)


def test_resolve_sources_includes_weather_when_enabled(tmp_db, monkeypatch):
    monkeypatch.setenv("ENERGY_ENABLE_WEATHER", "true")
    from energy_prices.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    assert "weather" in scheduler._resolve_sources("all")


def test_resolve_sources_explicit_weather_is_honoured(tmp_db):
    assert scheduler._resolve_sources("weather") == ["weather"]


def test_resolve_sources_unknown_raises(tmp_db):
    with pytest.raises(ValueError):
        scheduler._resolve_sources("nope")


# --- backfill chunk arithmetic ----------------------------------------------
def test_backfill_chunks_are_contiguous_and_inclusive(monkeypatch):
    """Chunks must tile [start, end] with no gaps/overlaps; each <= chunk_days."""
    calls: list[tuple[dt.date, dt.date]] = []

    def fake_ingest(source, start, end, skip_gas=False):
        calls.append((start, end))
        return {source: 1}

    monkeypatch.setattr(scheduler, "run_ingestion", fake_ingest)
    start, end = dt.date(2024, 1, 1), dt.date(2024, 4, 10)  # 101 days
    scheduler.run_backfill(source="ttf", start=start, end=end, chunk_days=30)

    assert calls[0][0] == start
    assert calls[-1][1] == end
    for (_, prev_end), (next_start, _) in zip(calls, calls[1:]):
        assert next_start == prev_end + dt.timedelta(days=1)  # contiguous, no overlap
    for c_start, c_end in calls:
        assert 0 <= (c_end - c_start).days <= 29  # within chunk_days


def test_backfill_rejects_start_after_end():
    with pytest.raises(ValueError):
        scheduler.run_backfill(source="ttf", start=dt.date(2024, 5, 1), end=dt.date(2024, 1, 1))


# --- daily_job stage isolation ----------------------------------------------
def test_daily_job_survives_ingestion_failure(tmp_db, monkeypatch):
    """One stage raising must not abort the whole daily job."""
    def boom(*a, **k):
        raise RuntimeError("ingest exploded")

    monkeypatch.setattr(scheduler, "run_ingestion", boom)
    summary = scheduler.daily_job(notify=False)
    assert summary["ingested"] == {}  # failure isolated to an empty result
    assert "finished_at" in summary and "forecast_rows" in summary


# --- first-boot bootstrap ----------------------------------------------------
def test_bootstrap_seeds_when_empty_and_demo(tmp_db):
    from energy_prices.config import Market, Zone
    from energy_prices.storage.db import session_scope
    from energy_prices.storage.repositories import PriceRepository

    scheduler._bootstrap_if_empty()
    with session_scope() as s:
        assert PriceRepository(s).latest_delivery(
            Market.ELEC_DAYAHEAD.value, zone=Zone.PUN.value
        ) is not None


def test_bootstrap_noop_when_demo_off(tmp_db, monkeypatch):
    monkeypatch.setenv("ENERGY_DEMO_MODE", "false")
    from energy_prices.config import Market, Zone
    from energy_prices.config import settings as settings_mod
    from energy_prices.storage.db import session_scope
    from energy_prices.storage.repositories import PriceRepository

    settings_mod.get_settings.cache_clear()
    scheduler._bootstrap_if_empty()
    with session_scope() as s:
        assert PriceRepository(s).latest_delivery(
            Market.ELEC_DAYAHEAD.value, zone=Zone.PUN.value
        ) is None
