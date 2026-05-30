"""Seasonal-naive baseline forecaster.

The point forecast for a target timestamp is simply the observed value from the
same time one *week* ago (lag 168h for hourly data), falling back to one *day*
ago (lag 24h) and finally to the last observed value when history is too short.

Probabilistic forecasts are produced by an empirical residual approach: we
compute in-sample residuals ``y - seasonal_naive(y)`` over the training history,
take their empirical quantiles, and add those (constant) offsets to the point
forecast. This yields a cheap-but-honest baseline distribution that is calibrated
to the model's own historical errors.

The implementation is resolution-aware: lags are expressed in *steps* derived
from the median spacing of the training index, so it works for hourly electricity
(168/24 step lags) and daily gas (7/1 step lags) alike. It depends only on
pandas/numpy core deps and is robust to short or irregular histories.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .base import DEFAULT_QUANTILES, Forecaster, ForecastResult

logger = logging.getLogger(__name__)

# One week / one day expressed as time offsets; converted to integer step counts
# using the inferred sampling frequency of the training history.
_WEEK = pd.Timedelta(days=7)
_DAY = pd.Timedelta(days=1)


class SeasonalNaiveForecaster(Forecaster):
    """Same-hour-last-week naive forecaster with empirical residual quantiles."""

    name: str = "seasonal_naive"
    version: str = "0.1.0"

    def __init__(self) -> None:
        self._y: pd.Series | None = None
        self._step: pd.Timedelta | None = None
        self._week_steps: int = 0
        self._day_steps: int = 0
        # Empirical residual distribution keyed by quantile level (filled in fit()).
        self._residuals: np.ndarray = np.empty(0, dtype=float)
        self._last_value: float = float("nan")

    # ------------------------------------------------------------------ fit
    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> SeasonalNaiveForecaster:
        """Store history, infer step size, and build the residual distribution.

        ``exog`` is accepted for interface compatibility but ignored: a seasonal
        naive baseline uses only the target's own past.
        """
        if y is None or len(y) == 0:
            raise ValueError("SeasonalNaiveForecaster.fit requires a non-empty series")

        y = self._clean_series(y)
        if len(y) == 0:
            raise ValueError("SeasonalNaiveForecaster.fit: series has no finite values")

        self._y = y
        self._last_value = float(y.iloc[-1])

        self._step = self._infer_step(y.index)
        self._week_steps = max(1, int(round(_WEEK / self._step)))
        self._day_steps = max(1, int(round(_DAY / self._step)))

        self._residuals = self._compute_residuals(y)
        logger.debug(
            "%s fitted: n=%d step=%s week_steps=%d day_steps=%d residuals=%d",
            self.name,
            len(y),
            self._step,
            self._week_steps,
            self._day_steps,
            self._residuals.size,
        )
        return self

    # --------------------------------------------------------------- predict
    def predict(
        self,
        horizon_index: pd.DatetimeIndex,
        exog_future: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> ForecastResult:
        """Build a wide quantile DataFrame over ``horizon_index``."""
        if self._y is None:
            raise RuntimeError("SeasonalNaiveForecaster.predict called before fit()")

        horizon_index = self._coerce_index(horizon_index)
        quantiles = tuple(sorted(float(q) for q in quantiles))

        point = self._point_forecast(horizon_index)

        # Constant residual offsets per quantile (empirical, may be empty -> 0.0).
        offsets = self._residual_offsets(quantiles)

        data: dict[str, np.ndarray] = {}
        point_values = point.to_numpy(dtype=float)
        for q in quantiles:
            col = self._qcol(q)
            data[col] = point_values + offsets[q]

        frame = pd.DataFrame(data, index=horizon_index)
        frame = self._enforce_monotone(frame, quantiles)
        return ForecastResult(frame, self.name, self.version)

    # ------------------------------------------------------------- internals
    def _point_forecast(self, horizon_index: pd.DatetimeIndex) -> pd.Series:
        """Same-time-last-week value, falling back to last-day then last value."""
        assert self._y is not None  # noqa: S101 - guarded by predict()
        y = self._y
        values = np.empty(len(horizon_index), dtype=float)

        for i, ts in enumerate(horizon_index):
            val = self._lookup(y, ts, self._week_steps)
            if val is None:
                val = self._lookup(y, ts, self._day_steps)
            if val is None:
                val = self._last_value
            values[i] = val

        return pd.Series(values, index=horizon_index, name="point")

    def _lookup(self, y: pd.Series, ts: pd.Timestamp, steps: int) -> float | None:
        """Value at ``ts - steps*step``; tolerant of small index misalignment."""
        if self._step is None:
            return None
        target = ts - steps * self._step
        # Exact hit first (fast path for regular grids).
        if target in y.index:
            val = y.loc[target]
            # Guard against duplicate-label Series returning a sub-Series.
            if np.ndim(val) > 0:
                val = float(np.asarray(val, dtype=float)[-1])
            return float(val) if np.isfinite(val) else None
        # Nearest-within-half-a-step tolerance for slightly irregular indices.
        pos = y.index.get_indexer([target], method="nearest")
        if pos.size and pos[0] != -1:
            cand_ts = y.index[pos[0]]
            if abs(cand_ts - target) <= self._step / 2:
                val = float(y.iloc[pos[0]])
                return val if np.isfinite(val) else None
        return None

    def _compute_residuals(self, y: pd.Series) -> np.ndarray:
        """In-sample residuals y - seasonal_naive(y) using the same fallback rule."""
        fitted = np.empty(len(y), dtype=float)
        idx = y.index
        for i, ts in enumerate(idx):
            val = self._lookup(y, ts, self._week_steps)
            if val is None:
                val = self._lookup(y, ts, self._day_steps)
            if val is None:
                # No lagged anchor available for the earliest points; skip.
                fitted[i] = np.nan
            else:
                fitted[i] = val
        resid = y.to_numpy(dtype=float) - fitted
        resid = resid[np.isfinite(resid)]
        return resid

    def _residual_offsets(self, quantiles: tuple[float, ...]) -> dict[float, float]:
        """Empirical residual quantile per requested level (0.0 if no residuals)."""
        if self._residuals.size == 0:
            return {q: 0.0 for q in quantiles}
        levels = np.asarray(quantiles, dtype=float)
        qs = np.quantile(self._residuals, levels)
        return {float(q): float(v) for q, v in zip(quantiles, qs)}

    @staticmethod
    def _enforce_monotone(frame: pd.DataFrame, quantiles: tuple[float, ...]) -> pd.DataFrame:
        """Guard against quantile crossing by sorting values row-wise."""
        if frame.shape[1] <= 1:
            return frame
        cols = [SeasonalNaiveForecaster._qcol(q) for q in quantiles]
        sorted_vals = np.sort(frame[cols].to_numpy(dtype=float), axis=1)
        frame[cols] = sorted_vals
        return frame

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _qcol(q: float) -> str:
        """Column label for a quantile level, matching ForecastResult convention."""
        return f"q{q:g}"

    @staticmethod
    def _clean_series(y: pd.Series) -> pd.Series:
        """Coerce to a sorted, tz-aware (UTC) float Series without NaNs/dupes."""
        s = pd.Series(y).astype(float)
        s = s[~s.index.duplicated(keep="last")]
        s = s.sort_index()
        idx = SeasonalNaiveForecaster._coerce_index(s.index)
        s.index = idx
        return s.dropna()

    @staticmethod
    def _coerce_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Ensure a UTC tz-aware DatetimeIndex (all datetimes are UTC by contract)."""
        idx = pd.DatetimeIndex(index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        return idx

    @staticmethod
    def _infer_step(index: pd.DatetimeIndex) -> pd.Timedelta:
        """Median spacing of the index; defaults to 1h for degenerate histories."""
        if len(index) < 2:
            return pd.Timedelta(hours=1)
        deltas = np.diff(index.asi8)  # nanoseconds between consecutive timestamps
        deltas = deltas[deltas > 0]
        if deltas.size == 0:
            return pd.Timedelta(hours=1)
        step = pd.Timedelta(int(np.median(deltas)), unit="ns")
        if step <= pd.Timedelta(0):
            return pd.Timedelta(hours=1)
        return step
