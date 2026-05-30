"""Conformalized Quantile Regression (CQR) wrapper for calibrated intervals.

A :class:`Forecaster` produces quantile forecasts, but its nominal bands rarely
hit their nominal coverage out of the box (our LightGBM/LEAR bands ran ~0.63 vs
0.80 on the MVP). CQR (Romano, Patterson & Candès, 2019) fixes this with a
distribution-free, model-agnostic post-hoc step:

1. Split the history into a training part (earlier) and a calibration part (the
   recent tail).
2. Fit the base model on the training part and predict the calibration part.
3. For each symmetric quantile pair (q_lo, q_hi) compute conformity scores
   ``E_i = max(q_lo_i - y_i, y_i - q_hi_i)`` and take the finite-sample-corrected
   empirical ``(q_hi - q_lo)`` quantile of E as an additive offset.
4. Refit the base on the full history; at predict time widen each band by its
   offset (offset may be negative -> tighten over-wide bands) and re-sort.

The result keeps the base model's shape but makes the bands honest. Wraps ANY
Forecaster (including the ensemble) and is itself a Forecaster.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

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
    ) -> None:
        self.base = base
        self.cal_fraction = cal_fraction
        self.min_cal = min_cal
        self.min_train = min_train
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

    def _calibrate(
        self, y: pd.Series, exog: pd.DataFrame | None, cal_n: int,
        quantiles: tuple[float, ...],
    ) -> None:
        y_train = y.iloc[:-cal_n]
        y_cal = y.iloc[-cal_n:]
        exog_train = exog.reindex(y_train.index) if exog is not None else None
        exog_cal = exog.reindex(y_cal.index) if exog is not None else None

        # The base must be re-instantiated-style fresh; we rely on fit() being
        # idempotent (re-fitting replaces internal state), which holds for all
        # our models.
        self.base.fit(y_train, exog_train)
        result = self.base.predict(y_cal.index, exog_future=exog_cal, quantiles=quantiles)
        q = result.quantiles.reindex(y_cal.index)
        actual = y_cal.to_numpy(dtype=float)

        for lo, hi in _symmetric_pairs(quantiles):
            lo_col, hi_col = f"q{lo:g}", f"q{hi:g}"
            if lo_col not in q.columns or hi_col not in q.columns:
                continue
            qlo = q[lo_col].to_numpy(dtype=float)
            qhi = q[hi_col].to_numpy(dtype=float)
            mask = np.isfinite(qlo) & np.isfinite(qhi) & np.isfinite(actual)
            if mask.sum() < self.min_cal:
                continue
            scores = np.maximum(qlo[mask] - actual[mask], actual[mask] - qhi[mask])
            nominal = hi - lo
            m = scores.size
            # Finite-sample conformal level: ceil((m+1)*nominal)/m, clipped to [0,1].
            level = min(1.0, np.ceil((m + 1) * nominal) / m)
            offset = float(np.quantile(scores, level, method="higher"))
            # Apply symmetrically: widen (or tighten) each side by `offset`.
            self._offset_by_level[lo] = self._offset_by_level.get(lo, 0.0) - offset
            self._offset_by_level[hi] = self._offset_by_level.get(hi, 0.0) + offset
        logger.info(
            "CQR calibrated offsets on %d points: %s",
            cal_n,
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
