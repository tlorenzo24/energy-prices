"""Conformalized Quantile Regression (CQR) wrapper for calibrated intervals.

A :class:`Forecaster` produces quantile forecasts, but its nominal bands rarely
hit their nominal coverage out of the box (our LightGBM/LEAR bands ran ~0.63 vs
0.80 on the MVP). CQR (Romano, Patterson & Candès, 2019) fixes this with a
distribution-free, model-agnostic post-hoc step:

1. Walk a rolling origin across the calibration tail in blocks of the operational
   ``horizon`` (e.g. one delivery day). At each origin, fit the base on the
   history strictly before the block and predict exactly that block — the same
   setup used in production, so lag features are real rather than NaN-imputed.
   (A single whole-tail ``predict`` would degrade lag-recursive models like
   LightGBM/LEAR and systematically inflate the offsets.)
2. Pool the per-block conformity scores across origins.
3. For each symmetric quantile pair (q_lo, q_hi) compute conformity scores
   ``E_i = max(q_lo_i - y_i, y_i - q_hi_i)`` and take the finite-sample-corrected
   empirical ``(q_hi - q_lo)`` quantile of the pooled E as an additive offset.
4. Refit the base on the full history; at predict time widen each band by its
   offset (offset may be negative -> tighten over-wide bands) and re-sort.

The result keeps the base model's shape but makes the bands honest. Wraps ANY
Forecaster (including the ensemble) and is itself a Forecaster.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from energy_prices.features.build import _periods_per_hour
from energy_prices.models.base import (
    DEFAULT_QUANTILES,
    Forecaster,
    ForecastResult,
)

logger = logging.getLogger(__name__)


def _symmetric_pairs(quantiles: tuple[float, ...]) -> list[tuple[float, float]]:
    """Symmetric (lo, hi) pairs around the median, e.g. (0.1,0.9), (0.25,0.75)."""
    qs = sorted(float(q) for q in quantiles)
    pairs: list[tuple[float, float]] = []
    i, j = 0, len(qs) - 1
    while i < j:
        lo, hi = qs[i], qs[j]
        if abs((lo + hi) - 1.0) < 1e-6:  # symmetric around 0.5
            pairs.append((lo, hi))
        i += 1
        j -= 1
    return pairs


class CalibratedForecaster(Forecaster):
    """Wrap a base Forecaster and conformalize its quantile bands (split CQR)."""

    def __init__(
        self,
        base: Forecaster,
        cal_fraction: float = 0.2,
        min_cal: int = 48,
        min_train: int = 168,
        horizon: int | None = None,
        max_cal_windows: int = 30,
    ) -> None:
        self.base = base
        self.cal_fraction = cal_fraction
        self.min_cal = min_cal
        self.min_train = min_train
        # Operational forecast length, in periods of the series resolution. Used
        # as the rolling-origin block size in calibration so conformity scores
        # reflect the real horizon. None -> inferred as one delivery day.
        self.horizon = horizon
        # Cap on rolling-origin refits, to bound calibration cost on long history.
        self.max_cal_windows = max_cal_windows
        # Signed additive offset applied to each quantile column at predict time.
        self._offset_by_level: dict[float, float] = {}
        self._calibrated = False

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"{getattr(self.base, 'name', 'model')}+cqr"

    @property
    def version(self) -> str:  # type: ignore[override]
        return getattr(self.base, "version", "0.1.0")

    def fit(
        self, y: pd.Series, exog: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> CalibratedForecaster:
        y = pd.Series(y).astype(float).sort_index()
        y = y[~y.index.duplicated(keep="last")].dropna()
        n = len(y)
        cal_n = max(self.min_cal, int(n * self.cal_fraction))

        self._offset_by_level = {}
        # Only calibrate when there is enough data for a clean train/cal split.
        if n - cal_n >= self.min_train and cal_n >= self.min_cal:
            try:
                self._calibrate(y, exog, cal_n, quantiles)
                self._calibrated = True
            except Exception as exc:  # noqa: BLE001 - calibration must never break fit
                logger.warning("CQR calibration failed (%s); using uncalibrated bands.", exc)
                self._offset_by_level = {}

        # Final model is always fit on the full history.
        self.base.fit(y, exog)
        return self

    def _resolve_horizon(self, index: pd.DatetimeIndex) -> int:
        """Rolling-origin block size: the operational horizon, in periods."""
        if self.horizon is not None and self.horizon > 0:
            return int(self.horizon)
        # Default: one operational delivery day in periods of the series resolution
        # (24 hourly, 96 for 15-min, 1 daily).
        ppr = _periods_per_hour(pd.DatetimeIndex(index))
        return max(1, int(round(24 * ppr)))

    def _calibrate(
        self, y: pd.Series, exog: pd.DataFrame | None, cal_n: int,
        quantiles: tuple[float, ...],
    ) -> None:
        n = len(y)
        horizon = self._resolve_horizon(y.index)
        cal_start = n - cal_n
        origins = list(range(cal_start, n, horizon))
        # Bound cost on long history: keep only the most recent windows.
        if self.max_cal_windows and len(origins) > self.max_cal_windows:
            origins = origins[-self.max_cal_windows :]

        pairs = _symmetric_pairs(quantiles)
        # Pool per-block conformity scores across rolling origins, per (lo,hi) pair.
        scores_by_pair: dict[tuple[float, float], list[np.ndarray]] = {p: [] for p in pairs}
        used = 0

        for origin in origins:
            end = min(origin + horizon, n)
            y_hist = y.iloc[:origin]
            if len(y_hist) < self.min_train:
                continue
            block_idx = y.index[origin:end]
            exog_hist = exog.reindex(y_hist.index) if exog is not None else None
            exog_block = exog.reindex(block_idx) if exog is not None else None

            # Fit strictly before the block, predict exactly the block — the
            # operational rolling-origin setup, so lag features are real. Relies
            # on fit() replacing internal state (true for all our models).
            self.base.fit(y_hist, exog_hist)
            result = self.base.predict(block_idx, exog_future=exog_block, quantiles=quantiles)
            q = result.quantiles.reindex(block_idx)
            actual = y.iloc[origin:end].to_numpy(dtype=float)
            used += 1

            for lo, hi in pairs:
                lo_col, hi_col = f"q{lo:g}", f"q{hi:g}"
                if lo_col not in q.columns or hi_col not in q.columns:
                    continue
                qlo = q[lo_col].to_numpy(dtype=float)
                qhi = q[hi_col].to_numpy(dtype=float)
                mask = np.isfinite(qlo) & np.isfinite(qhi) & np.isfinite(actual)
                if mask.any():
                    scores_by_pair[(lo, hi)].append(
                        np.maximum(qlo[mask] - actual[mask], actual[mask] - qhi[mask])
                    )

        for (lo, hi), chunks in scores_by_pair.items():
            if not chunks:
                continue
            scores = np.concatenate(chunks)
            m = scores.size
            if m < self.min_cal:
                continue
            nominal = hi - lo
            # Finite-sample conformal level: ceil((m+1)*nominal)/m, clipped to [0,1].
            level = min(1.0, np.ceil((m + 1) * nominal) / m)
            offset = float(np.quantile(scores, level, method="higher"))
            # Apply symmetrically: widen (or tighten) each side by `offset`.
            self._offset_by_level[lo] = self._offset_by_level.get(lo, 0.0) - offset
            self._offset_by_level[hi] = self._offset_by_level.get(hi, 0.0) + offset
        logger.info(
            "CQR calibrated on %d rolling-origin window(s), horizon=%d: %s",
            used, horizon,
            {f"q{k:g}": round(v, 2) for k, v in self._offset_by_level.items()},
        )

    def predict(
        self,
        horizon_index: pd.DatetimeIndex,
        exog_future: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> ForecastResult:
        result = self.base.predict(horizon_index, exog_future, quantiles)
        if not self._offset_by_level:
            # Re-label so the stored model_name reflects the (attempted) wrapper.
            return ForecastResult(result.quantiles, self.name, self.version)

        adjusted = result.quantiles.copy()
        for col in adjusted.columns:
            try:
                level = float(str(col).lstrip("q"))
            except ValueError:
                continue
            if level in self._offset_by_level:
                adjusted[col] = adjusted[col] + self._offset_by_level[level]

        # Re-sort across quantile columns so widening never crosses bands.
        levels = []
        for c in adjusted.columns:
            try:
                levels.append((float(str(c).lstrip("q")), c))
            except ValueError:
                pass
        ordered_cols = [c for _, c in sorted(levels)]
        if ordered_cols:
            vals = np.sort(adjusted[ordered_cols].to_numpy(dtype=float), axis=1)
            adjusted[ordered_cols] = vals
        return ForecastResult(adjusted, self.name, self.version)
