"""CLI tests — the documented primary entry point had zero coverage.

Exercises command wiring, the market-alias map, and the input-validation
boundaries (bad market/zone/date) that should fail cleanly rather than with a
raw traceback or a misleading green "Saved 0 forecast rows.".
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from energy_prices.cli import _MARKET_ALIASES, app

runner = CliRunner()


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("ENERGY_DATABASE_URL", f"sqlite:///{(tmp_path / 't.db').as_posix()}")
    monkeypatch.setenv("ENERGY_DEMO_MODE", "true")
    from energy_prices.config import settings as settings_mod
    from energy_prices.storage import db as db_mod

    for fn in (settings_mod.get_settings, db_mod.get_engine, db_mod._session_factory):
        fn.cache_clear()
    yield
    for fn in (settings_mod.get_settings, db_mod.get_engine, db_mod._session_factory):
        fn.cache_clear()


def _text(result) -> str:
    txt = result.output or ""
    try:
        txt += result.stderr or ""
    except (ValueError, AttributeError):
        pass
    return txt


# --- help wiring ------------------------------------------------------------
@pytest.mark.parametrize("argv", [["--help"], ["forecast", "--help"], ["backtest", "--help"],
                                  ["ingest", "--help"], ["scheduler", "--help"]])
def test_help_exits_zero(argv):
    assert runner.invoke(app, argv).exit_code == 0


# --- market aliases ---------------------------------------------------------
def test_market_aliases_resolve():
    from energy_prices.config import Market

    assert _MARKET_ALIASES["elec"] == Market.ELEC_DAYAHEAD.value
    assert _MARKET_ALIASES["electricity"] == Market.ELEC_DAYAHEAD.value
    assert _MARKET_ALIASES["gas"] == Market.GAS_DAYAHEAD.value
    assert _MARKET_ALIASES["ttf"] == Market.TTF.value


# --- input-validation boundaries -------------------------------------------
def test_forecast_unknown_market_rejected(tmp_db):
    result = runner.invoke(app, ["forecast", "--market", "bogus"])
    assert result.exit_code != 0
    assert "unknown market" in _text(result).lower()


def test_forecast_unknown_zone_rejected(tmp_db):
    result = runner.invoke(app, ["forecast", "--market", "elec", "--zone", "NORDD"])
    assert result.exit_code != 0
    assert "unknown zone" in _text(result).lower()


def test_forecast_zone_on_gas_rejected(tmp_db):
    result = runner.invoke(app, ["forecast", "--market", "gas", "--zone", "NORD"])
    assert result.exit_code != 0


def test_ingest_bad_date_rejected(tmp_db):
    result = runner.invoke(app, ["ingest", "--start", "not-a-date"])
    assert result.exit_code != 0
    assert "invalid date" in _text(result).lower()


# --- end-to-end happy path (SQLite + demo) ---------------------------------
def test_init_seed_forecast_roundtrip(tmp_db):
    assert runner.invoke(app, ["init-db"]).exit_code == 0
    seeded = runner.invoke(app, ["seed-demo", "--days", "40"])
    assert seeded.exit_code == 0
    fc = runner.invoke(app, ["forecast", "--market", "elec", "--zone", "PUN"])
    assert fc.exit_code == 0
    assert "Saved" in _text(fc)
