"""Forecast evaluation metrics and a rolling-origin walk-forward backtest.

All functions are pure (numpy / pandas / scipy only) and do no DB access.
Timestamps are assumed UTC and timezone-aware; prices are EUR/MWh.

Headline conventions:
* Point accuracy -> rMAE (MAE / naive MAE). We avoid MAPE because electricity
  prices cross/equal zero, which makes percentage errors blow up; rMAE is
  scale-free and < 1 means "beats the naive benchmark".
* Probabilistic accuracy -> averaged pinball loss over quantile levels, an
  approximate CRPS from those levels, and empirical interval coverage.
* Model comparison -> Diebold-Mariano test on per-step error series.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Union

import numpy as np
import pandas as pd
from scipy import stats

from energy_prices.models.base import DEFAULT_QUANTILES, Forecaster

logger = logging.getLogger(__name__)

# Accepted 1-D numeric inputs for the metric functions.
ArrayLike = Union[pd.Series, np.ndarray, list]

__all__ = [
    "mae", "rmae", "pinball_loss", "avg_pinball", "crps_sample",
    "crps_from_quantiles", "coverage", "diebold_mariano", "walk_forward",
]


# --- Internal alignment helpers --------------------------------------------- #
def _to_array(values: ArrayLike) -> np.ndarray:
    """Coerce to a 1-D float ndarray."""
    if isinstance(values, pd.Series):
        arr = values.to_numpy(dtype="float64")
    else:
        arr = np.asarray(values, dtype="float64")
    return arr.ravel()


def _align(*arrays: ArrayLike) -> list[np.ndarray]:
    """Align N inputs to common, mutually-finite rows.

    Series with unique indices are inner-joined on the index; otherwise (e.g.
    duplicate labels in pooled backtest predictions) inputs are aligned
    positionally and truncated. Rows non-finite in any input are dropped.
    """
    series = [x for x in arrays if isinstance(x, pd.Series)]
    joinable = (
        bool(arrays) and len(series) == len(arrays)
        and all(s.index.is_unique for s in series)
    )
    if joinable:
        # `joinable` guarantees every element is a pd.Series; mypy can't narrow it.
        cols = [x.rename(i) for i, x in enumerate(arrays)]  # type: ignore[union-attr]
        joined = pd.concat(cols, axis=1, join="inner")
        out = [joined[i].to_numpy(dtype="float64") for i in range(len(arrays))]
    else:
        out = [_to_array(x) for x in arrays]
        n = min((a.size for a in out), default=0)
        out = [a[:n] for a in out]
    if not out:
        return out
    mask = np.ones(out[0].shape, dtype=bool)
    for a in out:
        mask &= np.isfinite(a)
    return [a[mask] for a in out]


def _align_pair(a: ArrayLike, b: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Align two inputs to common, mutually-finite rows (see ``_align``)."""
    arr_a, arr_b = _align(a, b)
    return arr_a, arr_b


# --- Point metrics ---------------------------------------------------------- #
def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error between actuals and predictions."""
    a, p = _align_pair(y_true, y_pred)
    if a.size == 0:
        return float("nan")
    return float(np.mean(np.abs(a - p)))


def rmae(y_true: ArrayLike, y_pred: ArrayLike, y_naive: ArrayLike) -> float:
    """Relative MAE: MAE(pred) / MAE(naive) — the headline point metric.

    `y_naive` is a benchmark's prediction (e.g. seasonal-naive "same hour, last
    week") over the same timestamps. < 1 beats the benchmark; 1.0 is parity.
    Returns +inf if the naive MAE is zero while the model errs, else nan.
    """
    model_mae = mae(y_true, y_pred)
    naive_mae = mae(y_true, y_naive)
    if not np.isfinite(naive_mae):
        return float("nan")
    if naive_mae == 0.0:
        return float("inf") if model_mae > 0.0 else float("nan")
    return float(model_mae / naive_mae)


# --- Probabilistic metrics -------------------------------------------------- #
def pinball_loss(y_true: ArrayLike, q_pred: ArrayLike, quantile: float) -> float:
    """Average pinball (quantile) loss at a single quantile level.

    L_q(y, f) = max(q * (y - f), (q - 1) * (y - f)).
    Lower is better; for q = 0.5 this equals half the MAE.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile!r}")
    a, f = _align_pair(y_true, q_pred)
    if a.size == 0:
        return float("nan")
    diff = a - f
    loss = np.maximum(quantile * diff, (quantile - 1.0) * diff)
    return float(np.mean(loss))


def _quantile_from_column(col: object) -> float | None:
    """Parse a numeric quantile level from a column label like 'q0.1' or 0.1."""
    if isinstance(col, (int, float)) and not isinstance(col, bool):
        return float(col)
    text = str(col).strip().lstrip("qQ")
    try:
        return float(text)
    except ValueError:
        return None


def avg_pinball(y_true: ArrayLike, quantiles_df: pd.DataFrame) -> float:
    """Mean pinball loss averaged over every quantile column of a wide frame.

    `quantiles_df` is a wide quantile frame (cols 'q0.1', 'q0.5', ... as produced
    by ForecastResult.quantiles); pandas inputs are aligned on common index.
    """
    if quantiles_df is None or quantiles_df.shape[1] == 0:
        return float("nan")

    losses: list[float] = []
    for col in quantiles_df.columns:
        q = _quantile_from_column(col)
        if q is None or not 0.0 < q < 1.0:
            logger.debug("Skipping non-quantile column %r in avg_pinball", col)
            continue
        losses.append(pinball_loss(y_true, quantiles_df[col], q))

    losses = [v for v in losses if np.isfinite(v)]
    if not losses:
        return float("nan")
    return float(np.mean(losses))


def crps_sample(
    y_true: ArrayLike | float, samples: pd.DataFrame | np.ndarray | list
) -> float:
    """Approximate CRPS from a Monte-Carlo ensemble of forecast samples.

    Energy-form estimator CRPS = E|X - y| - 0.5 * E|X - X'|, averaged over
    observations. `samples` is shaped (n_obs, n_samples); a 1-D vector is treated
    as one observation. Returns the mean CRPS across observations (lower better).
    """
    sample_arr = np.asarray(samples, dtype="float64")
    if sample_arr.ndim == 1:
        sample_arr = sample_arr.reshape(1, -1)
    if sample_arr.ndim != 2 or sample_arr.shape[1] == 0:
        return float("nan")

    y_arr = _to_array(y_true)
    if y_arr.size == 1 and sample_arr.shape[0] != 1:
        y_arr = np.repeat(y_arr, sample_arr.shape[0])
    if y_arr.size != sample_arr.shape[0]:
        raise ValueError(
            f"y_true length {y_arr.size} != samples rows {sample_arr.shape[0]}"
        )

    per_obs: list[float] = []
    for y_i, row in zip(y_arr, sample_arr):
        x = np.sort(row[np.isfinite(row)])
        n = x.size
        if n == 0 or not np.isfinite(y_i):
            continue
        term1 = np.mean(np.abs(x - y_i))
        # E|X - X'| = (2/n^2) * sum_i (2i - n - 1) * x_(i)  (Gini mean difference)
        if n == 1:
            term2 = 0.0
        else:
            idx = np.arange(1, n + 1)
            term2 = (2.0 / (n * n)) * float(np.sum((2 * idx - n - 1) * x))
        per_obs.append(float(term1 - 0.5 * term2))

    if not per_obs:
        return float("nan")
    return float(np.mean(per_obs))


def crps_from_quantiles(y_true: ArrayLike, quantiles_df: pd.DataFrame) -> float:
    """Approximate CRPS from a discrete set of predictive quantiles.

    CRPS = 2 * integral over (0,1) of the pinball loss; with a finite grid we
    approximate the integral by the trapezoidal rule over the sorted levels (the
    standard quantile-decomposition estimator). Companion to `avg_pinball`.
    """
    if quantiles_df is None or quantiles_df.shape[1] == 0:
        return float("nan")

    # Collect (level, per-observation pinball loss) over valid quantile columns.
    levels: list[float] = []
    pinballs: list[float] = []
    for col in quantiles_df.columns:
        q = _quantile_from_column(col)
        if q is None or not 0.0 < q < 1.0:
            continue
        loss = pinball_loss(y_true, quantiles_df[col], q)
        if np.isfinite(loss):
            levels.append(q)
            pinballs.append(loss)

    if not levels:
        return float("nan")

    order = np.argsort(levels)
    lv = np.asarray(levels, dtype="float64")[order]
    pb = np.asarray(pinballs, dtype="float64")[order]

    if lv.size == 1:
        # Single quantile: integral of a step ~ 2 * pinball (best available est.).
        return float(2.0 * pb[0])

    # CRPS = 2 * int_0^1 pinball(q) dq ; trapezoid on the available grid (no
    # extrapolation beyond [min level, max level]). Prefer np.trapezoid (numpy>=2)
    # and fall back to the legacy np.trapz name on older numpy.
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
    integral = float(trapezoid(pb, lv))
    return float(2.0 * integral)


def coverage(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Empirical coverage: fraction of actuals inside [lower, upper].

    Compare to the interval's nominal level (an 80% PI from q0.1/q0.9 should
    cover ~0.80). Bounds are sorted per-observation, so the result is invariant
    to which argument is the larger limit.
    """
    y, lo, hi = _align(y_true, lower, upper)
    if y.size == 0:
        return float("nan")

    low = np.minimum(lo, hi)
    high = np.maximum(lo, hi)
    inside = (y >= low) & (y <= high)
    return float(np.mean(inside))


# --- Forecast comparison ---------------------------------------------------- #
def diebold_mariano(
    errors_a: ArrayLike, errors_b: ArrayLike, h: int = 1
) -> tuple[float, float]:
    """Diebold-Mariano test comparing two competing forecast error series.

    Inputs are the signed errors of models A and B over the same timestamps.
    Uses absolute-error loss, a Newey-West (Bartlett) long-run variance with
    truncation lag h-1, and the Harvey-Leybourne-Newbold small-sample
    correction. Returns ``(stat, p_value)`` from a two-sided Student-t test with
    n-1 dof. A negative stat means A has lower loss (A is more accurate).
    Returns ``(nan, nan)`` if the differential has no variance or n < 2.
    """
    if h < 1:
        raise ValueError(f"h must be >= 1, got {h}")

    a, b = _align_pair(errors_a, errors_b)
    n = a.size
    if n < 2:
        return (float("nan"), float("nan"))

    # Loss differential under absolute-error loss, then a Newey-West (Bartlett)
    # long-run variance using autocovariances up to lag h-1.
    d = np.abs(a) - np.abs(b)
    d_bar = float(np.mean(d))
    dc = d - d_bar
    lrv = float(np.mean(dc * dc))
    for lag in range(1, min(h, n)):
        weight = 1.0 - lag / float(h)
        lrv += 2.0 * weight * float(np.mean(dc[lag:] * dc[:-lag]))
    if lrv <= 0.0 or not np.isfinite(lrv):
        return (float("nan"), float("nan"))

    dm_stat = d_bar / np.sqrt(lrv / n)
    # Harvey, Leybourne & Newbold (1997) small-sample correction.
    correction = max((n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n, 0.0)
    hln_stat = dm_stat * np.sqrt(correction) if correction > 0 else dm_stat
    p_value = 2.0 * float(stats.t.cdf(-abs(hln_stat), df=n - 1))
    return (float(hln_stat), float(p_value))


# --- Walk-forward backtest -------------------------------------------------- #
def _seasonal_naive(y_hist: pd.Series, horizon_index: pd.DatetimeIndex) -> pd.Series:
    """Seasonal-naive benchmark (rMAE denominator): the last observation at or
    before target-minus-7-days; empty history falls back to nan / final value.
    """
    week = pd.Timedelta(days=7)
    fallback = float(y_hist.iloc[-1]) if len(y_hist) else float("nan")
    values: list[float] = []
    for ts in horizon_index:
        prior = y_hist.loc[: ts - week]  # last observation at or before ts - 7d
        values.append(float(prior.iloc[-1]) if len(prior) else fallback)
    return pd.Series(values, index=horizon_index, dtype="float64")


def _interval_bounds(
    quantiles_df: pd.DataFrame,
) -> tuple[pd.Series | None, pd.Series | None, float | None]:
    """Widest quantile pair as a coverage interval: (lower, upper, nominal=
    q_high-q_low) from the extreme valid columns, or (None, None, None) if < 2.
    """
    pairs: list[tuple[float, object]] = []
    for col in quantiles_df.columns:
        q = _quantile_from_column(col)
        if q is not None and 0.0 < q < 1.0:
            pairs.append((q, col))
    if len(pairs) < 2:
        return (None, None, None)
    pairs.sort(key=lambda t: t[0])
    q_low, col_low = pairs[0]
    q_high, col_high = pairs[-1]
    return (quantiles_df[col_low], quantiles_df[col_high], q_high - q_low)


def walk_forward(
    y: pd.Series,
    model_factory: Callable[[], Forecaster],
    horizon: int,
    step: int,
    n_windows: int,
    exog: pd.DataFrame | None = None,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> dict:  # noqa: PLR0913 - explicit backtest knobs are clearer than a config obj
    """Rolling-origin (daily-recalibration) walk-forward backtest.

    For each of `n_windows` origins the model is rebuilt via `model_factory()`,
    fit on all history strictly before the origin, and asked to forecast the next
    `horizon` periods; origins advance by `step` periods. This mimics operational
    day-ahead recalibration where a fresh model is trained every gate.

    Params: y (UTC DatetimeIndex history, sorted), model_factory (zero-arg ->
    fresh unfitted Forecaster), horizon (periods/window), step (periods between
    origins, usually == horizon), n_windows (most recent windows used), exog
    (optional, aligned to y and covering the horizon), quantiles.

    Returns a dict with keys:
        ``windows``     -> per-window dicts (window, origin, n, rmae, avg_pinball,
                           coverage, nominal_coverage, mae, model_name, version).
        ``predictions`` -> long-form DataFrame (target_start index): y_true,
                           y_pred, y_naive, window, quantile cols, pi_lower/upper.
        ``aggregate``   -> pooled metrics (rmae, avg_pinball, coverage,
                           nominal_coverage, mae, n, n_windows).
        ``config``      -> echo of horizon / step / n_windows / quantiles.
    """
    if not isinstance(y.index, pd.DatetimeIndex):
        raise TypeError("y must be indexed by a pandas DatetimeIndex")
    if horizon < 1 or step < 1 or n_windows < 1:
        raise ValueError("horizon, step and n_windows must all be >= 1")

    y = y[~y.index.duplicated(keep="last")].sort_index()
    n_obs = len(y)

    config = {
        "horizon": horizon, "step": step,
        "n_windows": n_windows, "quantiles": tuple(quantiles),
    }
    nan = float("nan")
    empty = {
        "windows": [],
        "predictions": pd.DataFrame(),
        "aggregate": {
            "rmae": nan, "avg_pinball": nan, "coverage": nan,
            "nominal_coverage": nan, "mae": nan, "n": 0, "n_windows": 0,
        },
        "config": config,
    }

    # The last origin starts so that its horizon ends at the final observation.
    last_origin = n_obs - horizon
    if last_origin <= 0:
        logger.warning(
            "walk_forward: not enough data (%d obs) for horizon=%d", n_obs, horizon
        )
        return empty

    # Origins for the most-recent n_windows, advancing backwards by `step`.
    origins = [last_origin - k * step for k in range(n_windows)]
    origins = sorted(o for o in origins if o > 0)
    if not origins:
        logger.warning("walk_forward: no valid origins for the requested window grid")
        return empty

    window_records: list[dict] = []
    pred_frames: list[pd.DataFrame] = []

    for w_idx, origin in enumerate(origins):
        y_train = y.iloc[:origin]
        horizon_index = y.index[origin : origin + horizon]
        if len(y_train) < 2 or len(horizon_index) == 0:
            continue
        y_actual = y.iloc[origin : origin + horizon]

        exog_train = None
        exog_future = None
        if exog is not None and not exog.empty:
            exog_train = exog.reindex(y_train.index)
            exog_future = exog.reindex(horizon_index)

        try:
            result = model_factory().fit_predict(
                y_train, horizon_index, exog=exog_train,
                exog_future=exog_future, quantiles=quantiles,
            )
        except Exception as exc:  # noqa: BLE001 - keep the backtest robust
            logger.warning(
                "walk_forward: model failed on window %d (origin=%s): %s",
                w_idx, y.index[origin], exc,
            )
            continue

        qdf = result.quantiles.reindex(horizon_index)
        point = result.point.reindex(horizon_index)
        naive = _seasonal_naive(y_train, horizon_index)
        lower, upper, nominal = _interval_bounds(qdf)

        # Assemble this window's long-form prediction frame.
        frame = pd.DataFrame(index=horizon_index)
        frame["y_true"] = y_actual.to_numpy(dtype="float64")
        frame["y_pred"] = point.to_numpy(dtype="float64")
        frame["y_naive"] = naive.to_numpy(dtype="float64")
        frame["window"] = w_idx
        for col in qdf.columns:
            frame[str(col)] = qdf[col].to_numpy(dtype="float64")
        if lower is not None and upper is not None:
            frame["pi_lower"] = lower.to_numpy(dtype="float64")
            frame["pi_upper"] = upper.to_numpy(dtype="float64")

        win_cov = (
            coverage(frame["y_true"], frame["pi_lower"], frame["pi_upper"])
            if "pi_lower" in frame else float("nan")
        )
        window_records.append({
            "window": w_idx,
            "origin": y.index[origin].to_pydatetime(),
            "n": int(len(frame)),
            "rmae": rmae(frame["y_true"], frame["y_pred"], frame["y_naive"]),
            "avg_pinball": avg_pinball(frame["y_true"], qdf),
            "coverage": win_cov,
            "nominal_coverage": nominal,
            "mae": mae(frame["y_true"], frame["y_pred"]),
            "model_name": result.model_name,
            "model_version": result.model_version,
        })
        pred_frames.append(frame)

    if not pred_frames:
        logger.warning("walk_forward: every window failed; returning empty result")
        return empty

    predictions = pd.concat(pred_frames, axis=0).sort_index()

    q_cols = [c for c in predictions.columns if _quantile_from_column(c) is not None]
    pooled_qdf = predictions[q_cols] if q_cols else pd.DataFrame(index=predictions.index)
    # Filter out None BEFORE averaging: np.nanmean([0.8, None]) raises TypeError,
    # which would abort the whole backtest if any window emitted <2 quantiles.
    _noms = [r["nominal_coverage"] for r in window_records if r["nominal_coverage"] is not None]
    pooled_nominal = float(np.mean(_noms)) if _noms else None
    pooled_cov = (
        coverage(predictions["y_true"], predictions["pi_lower"], predictions["pi_upper"])
        if "pi_lower" in predictions else float("nan")
    )
    aggregate = {
        "rmae": rmae(predictions["y_true"], predictions["y_pred"], predictions["y_naive"]),
        "avg_pinball": avg_pinball(predictions["y_true"], pooled_qdf),
        "coverage": pooled_cov,
        "nominal_coverage": pooled_nominal,
        "mae": mae(predictions["y_true"], predictions["y_pred"]),
        "n": int(len(predictions)),
        "n_windows": len(pred_frames),
    }

    return {
        "windows": window_records,
        "predictions": predictions,
        "aggregate": aggregate,
        "config": config,
    }
