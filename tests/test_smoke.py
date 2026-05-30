"""Smoke / regression tests for the energy-prices MVP.

Covers: evaluation metrics, the leak-safe feature builder (incl. the daily-data
regression where a collapsed rolling window must not wipe every row), model
fit/predict producing non-crossing quantiles, and the full seed -> forecast
round-trip against a throwaway SQLite database.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_prices.models.base import DEFAULT_QUANTILES

_Q_COLS = [f"q{q}" for q in DEFAULT_QUANTILES]


# --- Pure: evaluation metrics ----------------------------------------------
def test_point_and_interval_metrics():
    from energy_prices.forecasting import evaluation as ev

    y = pd.Series([10.0, 12.0, 11.0, 13.0])
    perfect = y.copy()
    naive = pd.Series([9.0, 9.0, 9.0, 9.0])

    assert ev.mae(y, perfect) == 0.0
    assert ev.rmae(y, perfect, naive) == 0.0  # perfect model beats naive
    # Pinball at q=0.5 is half the MAE -> 0 for a perfect forecast.
    assert ev.pinball_loss(y, perfect, 0.5) == pytest.approx(0.0)
    # Coverage: all actuals inside [lo, hi] -> 1.0.
    cov = ev.coverage(pd.Series([1.0, 2.0, 3.0]),
                      pd.Series([0.0, 0.0, 0.0]),
                      pd.Series([5.0, 5.0, 5.0]))
    assert cov == 1.0


def test_diebold_mariano_runs():
    from energy_prices.forecasting import evaluation as ev

    rng = np.random.default_rng(0)
    err_a = rng.normal(0, 1, 200)
    err_b = rng.normal(0, 3, 200)  # B clearly worse
    stat, p = ev.diebold_mariano(err_a, err_b)
    assert np.isfinite(stat) and 0.0 <= p <= 1.0


# --- Pure: feature builder --------------------------------------------------
def test_feature_frame_hourly_nonempty():
    from energy_prices.features.build import build_feature_frame

    idx = pd.date_range("2024-01-01", periods=24 * 30, freq="h", tz="UTC")
    y = pd.Series(np.sin(np.arange(len(idx)) / 24.0) * 10 + 100, index=idx, name="price")
    frame = build_feature_frame(y)
    assert not frame.empty
    assert "hour_sin" in frame.columns and "lag_24h" in frame.columns


def test_feature_frame_daily_does_not_collapse():
    """Regression: on daily data a 24h rolling window collapses to 1 period and
    its std is all-NaN; that column must be dropped, not wipe every row."""
    from energy_prices.features.build import build_feature_frame

    didx = pd.date_range("2024-01-01", periods=120, freq="D", tz="UTC")
    yd = pd.Series(np.arange(120, dtype=float) + 50.0, index=didx, name="price")
    frame = build_feature_frame(yd)
    assert not frame.empty  # would be empty before the all-NaN-column fix


# --- Pure: models produce non-crossing quantiles ----------------------------
@pytest.mark.parametrize("model_path,cls_name", [
    ("energy_prices.models.baseline", "SeasonalNaiveForecaster"),
    ("energy_prices.models.lgbm", "LightGBMForecaster"),
    ("energy_prices.models.lear", "LearForecaster"),
])
def test_model_quantiles_sorted(model_path, cls_name):
    import importlib

    cls = getattr(importlib.import_module(model_path), cls_name)
    idx = pd.date_range("2024-01-01", periods=24 * 45, freq="h", tz="UTC")
    rng = np.random.default_rng(1)
    y = pd.Series(
        np.sin(np.arange(len(idx)) / 24.0 * 2 * np.pi) * 15 + 100
        + rng.normal(0, 3, len(idx)),
        index=idx, name="price",
    )
    horizon = pd.date_range(idx[-1] + pd.Timedelta(hours=1), periods=24, freq="h", tz="UTC")

    result = cls().fit_predict(y, horizon)
    q = result.quantiles
    assert list(q.index) == list(horizon)
    vals = q[_Q_COLS].to_numpy(dtype=float)
    # Each row's quantiles must be non-decreasing across levels.
    assert np.all(np.diff(vals, axis=1) >= -1e-6)


# --- Integration: seed -> forecast round-trip on a throwaway DB -------------
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


def test_seed_then_forecast_roundtrip(tmp_db):
    from energy_prices.config import Market, Zone
    from energy_prices.forecasting.runner import run_forecasts
    from energy_prices.ingestion.demo import seed_demo
    from energy_prices.storage.db import session_scope
    from energy_prices.storage.repositories import ForecastRepository, PriceRepository

    with session_scope() as s:
        rows = seed_demo(s, days=40)
    assert rows > 0

    with session_scope() as s:
        prices = PriceRepository(s).get_prices(Market.ELEC_DAYAHEAD.value, zone=Zone.PUN.value)
    assert not prices.empty

    saved = run_forecasts(Market.ELEC_DAYAHEAD.value, Zone.PUN.value)
    assert saved > 0

    with session_scope() as s:
        fc = ForecastRepository(s).get_forecasts(Market.ELEC_DAYAHEAD.value, zone=Zone.PUN.value)
    assert not fc.empty
    assert "q0.5" in fc.columns
    # Median within a plausible band for the synthetic ~110 EUR/MWh series.
    assert 0.0 < fc["q0.5"].median() < 400.0


def test_gas_forecast_uses_ensemble(tmp_db):
    """Regression: gas ensemble must load SarimaxForecaster from gas_sarimax."""
    from energy_prices.config import Market
    from energy_prices.forecasting.runner import run_forecasts
    from energy_prices.ingestion.demo import seed_demo
    from energy_prices.storage.db import session_scope
    from energy_prices.storage.repositories import ForecastRepository

    with session_scope() as s:
        seed_demo(s, days=200)
    saved = run_forecasts(Market.GAS_DAYAHEAD.value)
    assert saved > 0
    with session_scope() as s:
        run_at = ForecastRepository(s).latest_run_at(Market.GAS_DAYAHEAD.value)
        fc = ForecastRepository(s).get_forecasts(Market.GAS_DAYAHEAD.value, run_at=run_at)
    assert not fc.empty


# --- Calibration (CQR) ------------------------------------------------------
def test_calibrated_forecaster_runs_and_calibrates():
    from energy_prices.models.baseline import SeasonalNaiveForecaster
    from energy_prices.models.calibration import CalibratedForecaster

    idx = pd.date_range("2024-01-01", periods=24 * 60, freq="h", tz="UTC")
    rng = np.random.default_rng(3)
    y = pd.Series(
        np.sin(np.arange(len(idx)) / 24.0 * 2 * np.pi) * 15 + 100 + rng.normal(0, 5, len(idx)),
        index=idx, name="price",
    )
    horizon = pd.date_range(idx[-1] + pd.Timedelta(hours=1), periods=24, freq="h", tz="UTC")

    cal = CalibratedForecaster(SeasonalNaiveForecaster(), cal_fraction=0.25)
    res = cal.fit_predict(y, horizon)
    assert res.model_name.endswith("+cqr")
    assert cal._offset_by_level  # calibration actually ran
    vals = res.quantiles[_Q_COLS].to_numpy(dtype=float)
    assert np.all(np.diff(vals, axis=1) >= -1e-6)  # non-crossing after widening


# --- Alerts -----------------------------------------------------------------
def test_alerts_trigger_and_clear(tmp_db):
    import datetime as dt

    from energy_prices.alerts import AlertRule, evaluate_alerts
    from energy_prices.config import Market
    from energy_prices.storage.db import session_scope
    from energy_prices.storage.repositories import ForecastRepository

    run = dt.datetime(2026, 5, 30, tzinfo=dt.UTC)
    target = dt.datetime(2026, 5, 31, tzinfo=dt.UTC)
    with session_scope() as s:
        ForecastRepository(s).save([{
            "run_at": run, "market": Market.GAS_DAYAHEAD.value, "zone": None,
            "target_start": target, "resolution_minutes": 1440,
            "model_name": "test", "quantile": 0.5, "value": 80.0,
        }])

    hit_rule = AlertRule(Market.GAS_DAYAHEAD.value, None, 60.0, "above", 0.5)
    miss_rule = AlertRule(Market.GAS_DAYAHEAD.value, None, 100.0, "above", 0.5)
    with session_scope() as s:
        hit = evaluate_alerts(s, [hit_rule])
        miss = evaluate_alerts(s, [miss_rule])
    assert len(hit) == 1 and hit[0]["worst_value"] == 80.0
    assert miss == []


# --- Notifications ----------------------------------------------------------
def test_dispatch_stub_and_payload_is_json_serializable():
    """With no channel configured, dispatch is a stub but builds a clean payload."""
    import datetime as dt
    import json

    from energy_prices.notifications import _json_default, build_payload, dispatch_alerts

    alerts = [{
        "rule": "PUN q0.9 above 200", "market": "elec_dayahead", "zone": "PUN",
        "worst_value": 250.0, "worst_target": dt.datetime(2026, 5, 31, 19, tzinfo=dt.UTC),
        "n_crossings": 3, "run_at": dt.datetime(2026, 5, 30, tzinfo=dt.UTC),
        "raised_at": dt.datetime(2026, 5, 30, 12, tzinfo=dt.UTC),
    }]
    payload = build_payload(alerts)
    assert payload["n_alerts"] == 1
    # Datetimes must round-trip through JSON without error.
    s = json.dumps(payload, default=_json_default)
    assert "2026-05-31" in s

    result = dispatch_alerts(alerts)  # no webhook/SMTP env -> stub
    assert result["skipped"] is True
    assert result["delivered"] == 0


def test_dispatch_empty_is_noop():
    from energy_prices.notifications import dispatch_alerts

    result = dispatch_alerts([])
    assert result["delivered"] == 0 and result["skipped"] is False
