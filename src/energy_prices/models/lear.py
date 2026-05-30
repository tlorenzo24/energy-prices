"""LEAR — LASSO-Estimated AutoRegressive forecaster (EPF Tier-1).

LEAR is one of the strongest open benchmarks for electricity price forecasting:
a high-dimensional linear model regularised with LASSO, fed autoregressive price
lags and calendar dummies. This module implements ``LearForecaster`` with two
back ends, selected automatically at fit time:

1. ``epftoolbox`` (OPTIONAL): if the package is importable we delegate the LEAR
   recalibration to its reference implementation. This is the canonical LEAR.
2. scikit-learn FALLBACK (always available): a robust per-horizon ``LassoCV``
   pipeline built on the project's leak-safe :func:`build_feature_frame`. The
   target is preprocessed with the EPF-standard ``arcsinh`` variance-stabilising
   transform followed by standardisation; the model predicts the conditional
   median and quantiles are formed by adding the empirical residual distribution
   to that point forecast (the same residual scheme used by the baseline).

The fallback NEVER raises :class:`ModelUnavailable`; only an explicit request for
the unavailable ``epftoolbox`` back end can. Which path executed is recorded in
``self.backend`` and logged. The forecaster is resolution-aware (it infers the
series MTU and lags by wall-clock hours) and returns a wide ``ForecastResult``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from energy_prices.features.build import (
    DEFAULT_LAG_HOURS,
    DEFAULT_ROLL_WINDOWS,
    build_feature_frame,
)
from energy_prices.models.base import (
    DEFAULT_QUANTILES,
    Forecaster,
    ForecastResult,
    ModelUnavailable,
)

logger = logging.getLogger(__name__)

# Optional reference back end. Guarded so missing deps never break import.
try:  # pragma: no cover - exercised only where epftoolbox is installed
    from epftoolbox.models import LEAR as _EpfLEAR  # type: ignore

    _HAS_EPFTOOLBOX = True
except Exception:  # noqa: BLE001 - any import failure means "not available"
    _EpfLEAR = None  # type: ignore[assignment]
    _HAS_EPFTOOLBOX = False


def _arcsinh_standardize(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """EPF-standard target transform: arcsinh then standardise.

    Returns the transformed array plus the (mean, std) used, so the inverse can
    be applied to predictions. ``arcsinh`` tames the heavy tails / spikes of
    power prices while remaining defined for the negative prices that occur in
    the Italian and wider European markets (unlike ``log``).
    """
    t = np.arcsinh(y.astype(float))
    mean = float(np.mean(t))
    std = float(np.std(t))
    if not np.isfinite(std) or std <= 1e-12:
        std = 1.0
    return (t - mean) / std, mean, std


def _inverse_arcsinh_standardize(t: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Invert :func:`_arcsinh_standardize`."""
    return np.sinh(t * std + mean)


class LearForecaster(Forecaster):
    """LASSO-Estimated AutoRegressive forecaster.

    Parameters
    ----------
    tz:
        IANA timezone used for calendar feature construction (default Rome).
    lag_hours, roll_windows:
        Day-ahead-safe autoregressive lag / rolling window definitions handed to
        :func:`build_feature_frame`.
    calibration_window:
        Number of trailing observations used to recalibrate the model. ``None``
        uses the full history. Bounds training cost on long histories and keeps
        the model adaptive, mirroring LEAR's rolling recalibration.
    n_alphas, cv:
        ``LassoCV`` hyper-parameters for the fallback path.
    prefer_epftoolbox:
        If True (default) and ``epftoolbox`` is importable, use it. If
        ``require_epftoolbox`` is True and the package is missing, raise
        :class:`ModelUnavailable`; otherwise transparently use the fallback.
    """

    name: str = "lear"
    version: str = "0.1.0"

    def __init__(
        self,
        tz: str = "Europe/Rome",
        lag_hours: tuple[int, ...] = DEFAULT_LAG_HOURS,
        roll_windows: tuple[int, ...] = DEFAULT_ROLL_WINDOWS,
        calibration_window: int | None = 365 * 24,
        n_alphas: int = 60,
        cv: int = 4,
        random_state: int = 0,
        prefer_epftoolbox: bool = True,
        require_epftoolbox: bool = False,
    ) -> None:
        if require_epftoolbox and not _HAS_EPFTOOLBOX:
            raise ModelUnavailable(
                "epftoolbox is not installed but require_epftoolbox=True. "
                "Install epftoolbox or set require_epftoolbox=False to use the "
                "scikit-learn fallback."
            )
        self.tz = tz
        self.lag_hours = tuple(lag_hours)
        self.roll_windows = tuple(roll_windows)
        self.calibration_window = calibration_window
        self.n_alphas = n_alphas
        self.cv = cv
        self.random_state = random_state

        self.backend: str = (
            "epftoolbox" if (prefer_epftoolbox and _HAS_EPFTOOLBOX) else "sklearn"
        )

        # State populated by fit().
        self._y: pd.Series | None = None
        self._exog: pd.DataFrame | None = None
        self._resolution_minutes: int = 60
        self._feature_cols: list[str] | None = None
        self._scaler: StandardScaler | None = None
        self._model: Any | None = None
        self._t_mean: float = 0.0
        self._t_std: float = 1.0
        self._residuals: np.ndarray | None = None
        self._epf_model: Any | None = None
        self._fitted: bool = False

    # ------------------------------------------------------------------ utils
    def _infer_resolution_minutes(self, index: pd.DatetimeIndex) -> int:
        """Median spacing of the index, in minutes (resolution-aware)."""
        if len(index) < 2:
            return 60
        delta = pd.Series(index).diff().median()
        if pd.isna(delta) or delta == pd.Timedelta(0):
            return 60
        return max(1, int(round(delta / pd.Timedelta(minutes=1))))

    @staticmethod
    def _ensure_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if index.tz is None:
            return index.tz_localize("UTC")
        return index.tz_convert("UTC")

    def _trim_history(self, y: pd.Series) -> pd.Series:
        if self.calibration_window is not None and len(y) > self.calibration_window:
            return y.iloc[-self.calibration_window :]
        return y

    # -------------------------------------------------------------------- fit
    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> LearForecaster:
        """Train on a UTC-indexed price Series and optional exogenous frame."""
        if y is None or len(y) == 0:
            raise ValueError("LearForecaster.fit requires a non-empty price series.")

        y = y.sort_index()
        y.index = self._ensure_utc(pd.DatetimeIndex(y.index))
        y = y[~y.index.duplicated(keep="last")].astype(float).dropna()
        if len(y) < max(self.lag_hours) // max(
            1, self._infer_resolution_minutes(y.index) // 60 or 1
        ) + 2:
            logger.warning(
                "lear: very short history (%d points); forecasts may be weak.", len(y)
            )

        self._resolution_minutes = self._infer_resolution_minutes(y.index)
        y = self._trim_history(y)
        self._y = y

        if exog is not None and not exog.empty:
            exog = exog.sort_index()
            exog.index = self._ensure_utc(pd.DatetimeIndex(exog.index))
            self._exog = exog
        else:
            self._exog = None

        if self.backend == "epftoolbox":
            try:
                self._fit_epftoolbox(y)
                self._fitted = True
                logger.info("lear: fitted via epftoolbox backend.")
                return self
            except ModelUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                logger.warning(
                    "lear: epftoolbox backend failed (%s); falling back to sklearn.",
                    exc,
                )
                self.backend = "sklearn"

        self._fit_sklearn(y, self._exog)
        self._fitted = True
        logger.info(
            "lear: fitted via sklearn fallback (%d features, res=%dmin).",
            0 if self._feature_cols is None else len(self._feature_cols),
            self._resolution_minutes,
        )
        return self

    def _fit_epftoolbox(self, y: pd.Series) -> None:
        """Prepare the epftoolbox LEAR. Recalibration happens at predict time.

        epftoolbox's LEAR is recalibrated per prediction day, so here we only
        capture the calibration window and instantiate the estimator. The actual
        ``recalibrate_and_forecast_next_day`` call is issued in
        :meth:`_predict_epftoolbox`.
        """
        if _EpfLEAR is None:  # defensive; backend should not be set otherwise
            raise ModelUnavailable("epftoolbox is not available.")
        self._epf_model = _EpfLEAR(calibration_window=self.calibration_window)

    def _fit_sklearn(self, y: pd.Series, exog: pd.DataFrame | None) -> None:
        """Single global LassoCV on the leak-safe arcsinh-standardised target."""
        frame = build_feature_frame(
            y,
            exog=exog,
            tz=self.tz,
            lag_hours=self.lag_hours,
            roll_windows=self.roll_windows,
            dropna=True,
        )
        if frame.empty:
            raise ValueError(
                "lear: feature frame is empty after dropna — history too short "
                "for the configured lags."
            )

        target = y.reindex(frame.index)
        mask = target.notna()
        frame = frame.loc[mask]
        target = target.loc[mask]
        if frame.empty:
            raise ValueError("lear: no aligned target/feature rows to train on.")

        self._feature_cols = list(frame.columns)
        x = frame.to_numpy(dtype=float)

        y_t, self._t_mean, self._t_std = _arcsinh_standardize(target.to_numpy())

        self._scaler = StandardScaler()
        x_scaled = self._scaler.fit_transform(x)

        n_samples = x_scaled.shape[0]
        cv = max(2, min(self.cv, n_samples)) if n_samples >= 4 else None
        self._model = LassoCV(
            n_alphas=self.n_alphas,
            cv=cv,
            random_state=self.random_state,
            max_iter=10_000,
            n_jobs=None,
        )
        self._model.fit(x_scaled, y_t)

        # In-sample residuals on the ORIGINAL price scale -> empirical quantiles.
        pred_t = self._model.predict(x_scaled)
        pred_price = _inverse_arcsinh_standardize(pred_t, self._t_mean, self._t_std)
        self._residuals = target.to_numpy() - pred_price

    # ---------------------------------------------------------------- predict
    def predict(
        self,
        horizon_index: pd.DatetimeIndex,
        exog_future: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> ForecastResult:
        """Forecast quantiles for each timestamp in ``horizon_index``."""
        if not self._fitted or self._y is None:
            raise RuntimeError("LearForecaster.predict called before fit().")
        if horizon_index is None or len(horizon_index) == 0:
            raise ValueError("horizon_index must be a non-empty DatetimeIndex.")

        horizon_index = pd.DatetimeIndex(horizon_index)
        horizon_index = self._ensure_utc(horizon_index).sort_values()
        quantiles = tuple(quantiles)

        if exog_future is not None and not exog_future.empty:
            exog_future = exog_future.sort_index()
            exog_future.index = self._ensure_utc(pd.DatetimeIndex(exog_future.index))

        if self.backend == "epftoolbox" and self._epf_model is not None:
            try:
                point = self._predict_epftoolbox(horizon_index, exog_future)
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                logger.warning(
                    "lear: epftoolbox predict failed (%s); using sklearn fallback.",
                    exc,
                )
                if self._model is None:
                    self._fit_sklearn(self._y, self._exog)
                self.backend = "sklearn"
                point = self._predict_sklearn(horizon_index, exog_future)
        else:
            point = self._predict_sklearn(horizon_index, exog_future)

        quant_df = self._quantiles_from_residuals(point, quantiles)
        return ForecastResult(quant_df, self.name, self.version)

    def _predict_sklearn(
        self, horizon_index: pd.DatetimeIndex, exog_future: pd.DataFrame | None
    ) -> pd.Series:
        """Recursive multi-step point forecast on the original price scale.

        We extend the history one step at a time so each step's autoregressive
        lags can use prior predictions when they fall inside the horizon, exactly
        as a day-ahead model would once D-1 prices are known.
        """
        assert self._model is not None and self._scaler is not None
        assert self._feature_cols is not None

        history = self._y.copy()
        exog_all = self._combine_exog(exog_future)

        preds: dict[pd.Timestamp, float] = {}
        for ts in horizon_index:
            # Append a NaN placeholder so lag/rolling features can be computed
            # for `ts` from the (real + previously predicted) history.
            if ts not in history.index:
                history.loc[ts] = np.nan
                history = history.sort_index()

            frame = build_feature_frame(
                history,
                exog=exog_all,
                tz=self.tz,
                lag_hours=self.lag_hours,
                roll_windows=self.roll_windows,
                dropna=False,
            )
            if ts not in frame.index:
                preds[ts] = float(history.dropna().iloc[-1]) if history.notna().any() else 0.0
                history.loc[ts] = preds[ts]
                continue

            row = frame.loc[[ts], self._feature_cols]
            row = row.fillna(self._feature_means())
            x = self._scaler.transform(row.to_numpy(dtype=float))
            pred_t = float(self._model.predict(x)[0])
            pred_price = float(
                _inverse_arcsinh_standardize(
                    np.array([pred_t]), self._t_mean, self._t_std
                )[0]
            )
            preds[ts] = pred_price
            history.loc[ts] = pred_price  # feed forward for downstream lags

        return pd.Series(preds, name="point").reindex(horizon_index)

    def _predict_epftoolbox(
        self, horizon_index: pd.DatetimeIndex, exog_future: pd.DataFrame | None
    ) -> pd.Series:  # pragma: no cover - requires epftoolbox installed
        """Delegate the point forecast to epftoolbox's LEAR recalibration.

        epftoolbox expects a daily, 24-column layout. We build a continuous
        hourly price frame from the calibration history, ask LEAR to recalibrate
        and forecast each target day spanned by ``horizon_index``, then map the
        per-day vectors back onto the requested timestamps.
        """
        assert self._epf_model is not None and self._y is not None

        # Hourly view of the history; epftoolbox's reference LEAR is hourly.
        hist = self._y.copy()
        hist.index = self._ensure_utc(pd.DatetimeIndex(hist.index))
        hourly = hist.resample("1h").mean()

        df = pd.DataFrame({"Price": hourly})
        exog_all = self._combine_exog(exog_future)
        if exog_all is not None and not exog_all.empty:
            exog_hourly = exog_all.resample("1h").mean()
            for i, col in enumerate(exog_hourly.columns, start=1):
                df[f"Exogenous {i}"] = exog_hourly[col]
        df = df.dropna(subset=["Price"])

        target_days = pd.DatetimeIndex(
            sorted({ts.normalize() for ts in horizon_index})
        )
        day_forecasts: dict[pd.Timestamp, np.ndarray] = {}
        for day in target_days:
            yp = self._epf_model.recalibrate_and_forecast_next_day(
                df=df, calibration_window=self.calibration_window, next_day_date=day
            )
            day_forecasts[day] = np.asarray(yp, dtype=float).ravel()

        values: list[float] = []
        for ts in horizon_index:
            day = ts.normalize()
            vec = day_forecasts.get(day)
            if vec is None or len(vec) == 0:
                values.append(float("nan"))
                continue
            hour = int(ts.tz_convert("UTC").hour) if ts.tzinfo else int(ts.hour)
            values.append(float(vec[min(hour, len(vec) - 1)]))

        point = pd.Series(values, index=horizon_index, name="point")
        if point.isna().any():
            point = point.interpolate().ffill().bfill()
        return point

    # ------------------------------------------------------------- internals
    def _combine_exog(self, exog_future: pd.DataFrame | None) -> pd.DataFrame | None:
        """Concatenate training exog with future exog for feature building."""
        parts = [p for p in (self._exog, exog_future) if p is not None and not p.empty]
        if not parts:
            return None
        combined = pd.concat(parts, axis=0)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        return combined

    def _feature_means(self) -> pd.Series:
        """Per-feature training means (scaler centres) for NaN imputation."""
        assert self._scaler is not None and self._feature_cols is not None
        means = getattr(self._scaler, "mean_", None)
        if means is None:
            return pd.Series(0.0, index=self._feature_cols)
        return pd.Series(means, index=self._feature_cols)

    def _quantiles_from_residuals(
        self, point: pd.Series, quantiles: tuple[float, ...]
    ) -> pd.DataFrame:
        """Build a wide quantile frame: point + empirical residual offsets.

        The residual distribution is centred (median offset removed) so that the
        q0.5 column equals the point forecast, then non-crossing is enforced by a
        row-wise cumulative max across ascending quantiles.
        """
        cols = [f"q{q}" for q in quantiles]
        if self._residuals is None or len(self._residuals) == 0:
            data = {c: point.to_numpy() for c in cols}
            return pd.DataFrame(data, index=point.index)

        resid = np.asarray(self._residuals, dtype=float)
        resid = resid[np.isfinite(resid)]
        median_resid = float(np.median(resid)) if resid.size else 0.0
        offsets = {
            q: float(np.quantile(resid, q)) - median_resid for q in quantiles
        }

        base = point.to_numpy(dtype=float)
        out = pd.DataFrame(index=point.index)
        for q in quantiles:
            out[f"q{q}"] = base + offsets[q]

        # Enforce monotone non-crossing quantiles across columns.
        ordered = sorted(quantiles)
        ordered_cols = [f"q{q}" for q in ordered]
        out[ordered_cols] = np.maximum.accumulate(out[ordered_cols].to_numpy(), axis=1)
        return out[[f"q{q}" for q in quantiles]]
