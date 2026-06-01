"""Tests for the evaluation/backtest harness that drove model-selection decisions.

walk_forward, crps_sample, crps_from_quantiles, avg_pinball, _interval_bounds and
_seasonal_naive had zero coverage despite producing the headline numbers in the
model docstrings (LightGBM vs ensemble DM p; gas PSV-basis rMAE). These lock in
the leak-safe rolling-origin grid, the metric formulas, and the aggregate's
robustness to degenerate (single-quantile) windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_prices.forecasting import evaluation as ev
from energy_prices.models.base import Forecaster, ForecastResult


class _StubForecaster(Forecaster):
    """Emits a constant per-column value over a fixed set of quantile columns."""

    name, version = "stub", "0"

    def __init__(self, qcols: list[str]) -> None:
        self._qcols = qcols
        self._mu = 0.0

    def fit(self, y, exog=None):
        self._mu = float(y.mean()) if len(y) else 0.0
        return self

    def predict(self, idx, exog_future=None, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9)):
        df = pd.DataFrame({c: self._mu for c in self._qcols}, index=idx)
        return ForecastResult(df, self.name, self.version)


def _hourly(n: int) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(11)
    return pd.Series(100 + rng.normal(0, 5, n), index=idx, name="price")


# --- walk_forward grid + leak-safety ---------------------------------------
def test_walk_forward_grid_and_leak_safety():
    from energy_prices.models.baseline import SeasonalNaiveForecaster

    y = _hourly(24 * 30)
    res = ev.walk_forward(y, SeasonalNaiveForecaster, horizon=24, step=24, n_windows=5)

    assert len(res["windows"]) == 5
    origins = [pd.Timestamp(w["origin"]) for w in res["windows"]]
    # Origins advance by exactly `step` (24h) periods.
    for prev, nxt in zip(origins, origins[1:]):
        assert nxt - prev == pd.Timedelta(hours=24)
    # Leak-safety: every forecast target_start is at/after its window's origin
    # (the model only saw history strictly before the origin).
    preds = res["predictions"]
    for w_idx, origin in enumerate(origins):
        window_idx = preds[preds["window"] == w_idx].index
        assert window_idx.min() >= origin


def test_walk_forward_handles_missing_nominal_coverage():
    """A window emitting <2 quantiles (nominal=None) must not crash the aggregate (#29)."""
    counter = {"i": 0}

    def factory():
        i = counter["i"]
        counter["i"] += 1
        # First window: a full interval; later windows: median only (nominal=None).
        cols = ["q0.1", "q0.5", "q0.9"] if i == 0 else ["q0.5"]
        return _StubForecaster(cols)

    y = _hourly(24 * 20)
    res = ev.walk_forward(y, factory, horizon=24, step=24, n_windows=4)
    agg = res["aggregate"]
    # Only the first window contributes a nominal coverage; no TypeError raised.
    assert agg["nominal_coverage"] == pytest.approx(0.8)
    assert np.isfinite(agg["mae"])


# --- CRPS / pinball / interval helpers --------------------------------------
def test_crps_sample_matches_normal_closed_form():
    rng = np.random.default_rng(0)
    samples = rng.normal(0.0, 3.0, 200_000)
    # CRPS of N(0,sigma) at y=0 = sigma*(sqrt(2/pi) - 1/sqrt(pi)) = 0.701 for sigma=3.
    assert ev.crps_sample(0.0, samples) == pytest.approx(0.701, abs=0.02)


def test_crps_from_quantiles_single_is_twice_pinball():
    y = pd.Series([10.0, 12.0, 11.0, 9.0])
    qdf = pd.DataFrame({"q0.5": [10.0, 10.0, 10.0, 10.0]}, index=y.index)
    expected = 2.0 * ev.pinball_loss(y, qdf["q0.5"], 0.5)
    assert ev.crps_from_quantiles(y, qdf) == pytest.approx(expected)


def test_crps_from_quantiles_multi_is_finite_nonneg():
    y = pd.Series([10.0, 12.0, 11.0, 9.0])
    qdf = pd.DataFrame(
        {"q0.1": [8, 9, 8, 7], "q0.5": [10, 11, 10, 9], "q0.9": [12, 13, 12, 11]},
        index=y.index, dtype="float64",
    )
    val = ev.crps_from_quantiles(y, qdf)
    assert np.isfinite(val) and val >= 0.0


def test_interval_bounds_extremes_and_degenerate():
    idx = pd.RangeIndex(3)
    qdf = pd.DataFrame(
        {"q0.1": [1, 1, 1], "q0.25": [2, 2, 2], "q0.5": [3, 3, 3],
         "q0.75": [4, 4, 4], "q0.9": [5, 5, 5]}, index=idx, dtype="float64",
    )
    lower, upper, nominal = ev._interval_bounds(qdf)
    assert list(lower) == [1, 1, 1]
    assert list(upper) == [5, 5, 5]
    assert nominal == pytest.approx(0.8)
    # Fewer than two quantile columns -> no interval.
    assert ev._interval_bounds(qdf[["q0.5"]]) == (None, None, None)


def test_seasonal_naive_uses_same_weekday_last_week():
    idx = pd.date_range("2024-01-01", periods=24 * 14, freq="h", tz="UTC")
    y = pd.Series(np.arange(len(idx), dtype="float64"), index=idx, name="price")
    cut = 24 * 10
    y_hist = y.iloc[:cut]
    horizon = y.index[cut : cut + 6]
    sn = ev._seasonal_naive(y_hist, horizon)
    # Each value equals the observation exactly 7 days earlier.
    for ts in horizon:
        assert sn.loc[ts] == pytest.approx(float(y_hist.loc[ts - pd.Timedelta(days=7)]))
