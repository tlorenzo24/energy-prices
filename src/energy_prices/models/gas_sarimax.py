"""SARIMAX forecaster for daily gas prices (PSV / TTF benchmarks).

A thin, robust wrapper around ``statsmodels`` SARIMAX tailored to a daily gas
price series. The model is fit on the (optionally log-transformed) daily level
with a sensible default order ``(1, 1, 1)`` and optional exogenous regressors
(e.g. heating-degree-days ``hdd`` and ``gas_storage_pct``) aligned to the target
index.

Probabilistic output is derived from ``get_forecast()``: the predicted mean and
standard error define a Normal predictive distribution, from which the requested
quantiles are taken via ``scipy.stats.norm.ppf``. If the optional ``arch``
package is installed, a GARCH(1,1) is fit on the standardised residuals as a
best-effort variance widener for the forecast horizon; failure is non-fatal and
falls back to the SARIMAX standard errors.

Datetimes are UTC and timezone-aware throughout; prices are EUR/MWh.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from energy_prices.models.base import (
    DEFAULT_QUANTILES,
    Forecaster,
    ForecastResult,
)

logger = logging.getLogger(__name__)

# Order tried first, then progressively simpler fallbacks on convergence trouble.
_DEFAULT_ORDER: tuple[int, int, int] = (1, 1, 1)
_FALLBACK_ORDERS: tuple[tuple[int, int, int], ...] = (
    (1, 1, 1),
    (1, 1, 0),
    (0, 1, 1),
    (1, 0, 0),
    (0, 1, 0),
)

# Floor on the predictive std (EUR/MWh) so degenerate fits still yield a spread.
_MIN_SIGMA = 1e-6


class SarimaxForecaster(Forecaster):
    """Daily-gas SARIMAX forecaster with Normal-approx quantiles.

    Parameters
    ----------
    order:
        SARIMAX ``(p, d, q)`` order. Defaults to ``(1, 1, 1)``.
    seasonal_order:
        Optional ``(P, D, Q, s)`` seasonal order. Defaults to no seasonality;
        daily gas rarely benefits from a fixed weekly SARIMA term and it slows
        the fit considerably.
    log_transform:
        Fit on ``log(price)`` and exponentiate the forecast. Only applied when
        the whole training series is strictly positive. Defaults to ``True``.
    use_garch:
        Best-effort GARCH(1,1) widening of the predictive variance when the
        optional ``arch`` package is available. Defaults to ``True``.
    exog_columns:
        Optional explicit subset/ordering of exogenous columns to use. When
        ``None`` all columns present in the ``exog`` frame are used.
    """

    name: str = "sarimax"
    version: str = "0.1.0"

    def __init__(
        self,
        order: tuple[int, int, int] = _DEFAULT_ORDER,
        seasonal_order: tuple[int, int, int, int] | None = None,
        log_transform: bool = True,
        use_garch: bool = True,
        exog_columns: list[str] | None = None,
    ) -> None:
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order) if seasonal_order else (0, 0, 0, 0)
        self.log_transform = log_transform
        self.use_garch = use_garch
        self.exog_columns = list(exog_columns) if exog_columns is not None else None

        # Populated by fit().
        self._result = None  # statsmodels SARIMAXResultsWrapper
        self._fitted_order: tuple[int, ...] = self.order
        self._log_applied: bool = False
        self._freq: str = "D"
        self._exog_cols: list[str] = []
        # Last in-sample exog row, broadcast forward when the horizon lacks exog.
        self._last_exog_row: pd.Series | None = None
        # Extra residual-vol multiplier from GARCH (1.0 == no widening).
        self._garch_scale: float = 1.0

    # ------------------------------------------------------------------ fit
    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> SarimaxForecaster:
        """Fit SARIMAX on the daily series ``y`` with optional ``exog``."""
        y = self._prepare_target(y)
        if y.empty:
            raise ValueError("SarimaxForecaster.fit: empty target series after cleaning.")

        self._log_applied = bool(self.log_transform and (y > 0).all())
        y_model = np.log(y) if self._log_applied else y.astype(float)

        exog_model = self._prepare_exog(exog, y_model.index, fitting=True)
        # Remember the last observed exog row so predict() can carry it forward
        # (a far more neutral default than zero for level regressors like hdd).
        if exog_model is not None and not exog_model.empty:
            self._last_exog_row = exog_model.iloc[-1].copy()

        result = self._fit_with_fallback(y_model, exog_model)
        self._result = result

        # Best-effort GARCH widening on the standardised in-sample residuals.
        self._garch_scale = 1.0
        if self.use_garch:
            try:
                self._garch_scale = self._estimate_garch_scale(result)
            except Exception as exc:  # pragma: no cover - defensive, best-effort
                logger.debug("GARCH variance widening skipped: %s", exc)
                self._garch_scale = 1.0

        return self

    # -------------------------------------------------------------- predict
    def predict(
        self,
        horizon_index: pd.DatetimeIndex,
        exog_future: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> ForecastResult:
        """Forecast quantiles for each timestamp in ``horizon_index``."""
        if self._result is None:
            raise RuntimeError("SarimaxForecaster.predict called before fit().")

        horizon_index = self._coerce_index(horizon_index)
        if len(horizon_index) == 0:
            empty = pd.DataFrame(
                columns=[f"q{q}" for q in quantiles],
                index=pd.DatetimeIndex([], tz="UTC", name="target_start"),
            )
            return ForecastResult(empty, self.name, self.version)

        steps = len(horizon_index)
        exog_future_model = self._prepare_exog(exog_future, horizon_index, fitting=False)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self._result.get_forecast(steps=steps, exog=exog_future_model)
            mean = np.asarray(forecast.predicted_mean, dtype=float)
            se = np.asarray(forecast.se_mean, dtype=float)

        # Widen the predictive std with the (best-effort) GARCH scale.
        sigma = np.clip(se * float(self._garch_scale), _MIN_SIGMA, None)

        frame = self._quantile_frame(mean, sigma, horizon_index, quantiles)
        return ForecastResult(frame, self.name, self.version)

    # ----------------------------------------------------------- internals
    def _fit_with_fallback(self, y_model: pd.Series, exog_model: pd.DataFrame | None):
        """Try the configured order, then progressively simpler ones."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        orders: list[tuple[int, ...]] = [self.order]
        for fb in _FALLBACK_ORDERS:
            if fb not in orders:
                orders.append(fb)

        last_error: Exception | None = None
        for order in orders:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SARIMAX(
                        y_model,
                        exog=exog_model,
                        order=order,
                        seasonal_order=self.seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                        trend=None,
                    )
                    result = model.fit(disp=False, maxiter=200, method="lbfgs")
                self._fitted_order = order
                if order != self.order:
                    logger.info(
                        "SARIMAX fell back from order %s to %s after convergence trouble.",
                        self.order,
                        order,
                    )
                return result
            except Exception as exc:  # noqa: BLE001 - try the next, simpler order
                last_error = exc
                logger.debug("SARIMAX order %s failed: %s", order, exc)
                continue

        raise RuntimeError(
            f"SARIMAX failed to converge for all attempted orders {orders}: {last_error}"
        )

    def _estimate_garch_scale(self, result) -> float:
        """Fit GARCH(1,1) on residuals; return a (>=1) vol-widening multiplier.

        Returns the ratio of the GARCH one-step conditional volatility forecast
        to the unconditional residual std. Values are clipped to ``[1.0, 3.0]``
        so the widening can never *shrink* the SARIMAX interval and stays sane.
        """
        try:
            from arch import arch_model  # optional dep, guarded
        except Exception as exc:  # pragma: no cover - optional dependency missing
            logger.debug("arch not installed; skipping GARCH widening: %s", exc)
            return 1.0

        resid = pd.Series(np.asarray(result.resid, dtype=float)).dropna()
        if len(resid) < 30:
            return 1.0

        base_std = float(resid.std(ddof=1))
        if not np.isfinite(base_std) or base_std <= 0:
            return 1.0

        # arch is happiest on a series with O(1)-O(100) scale; rescale to avoid
        # DataScaleWarning and poor optimisation on tiny (log-return) residuals.
        scale = 1.0
        if base_std < 1.0:
            scale = 1.0 / base_std
        scaled = resid * scale

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(scaled, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
            res = am.fit(disp="off", show_warning=False)
            fc = res.forecast(horizon=1, reindex=False)
            cond_var = float(np.asarray(fc.variance.values).ravel()[-1])

        if not np.isfinite(cond_var) or cond_var <= 0:
            return 1.0

        # Undo the rescale and compare to the unconditional residual std.
        cond_std = float(np.sqrt(cond_var)) / scale
        ratio = cond_std / base_std
        if not np.isfinite(ratio):
            return 1.0
        return float(np.clip(ratio, 1.0, 3.0))

    def _quantile_frame(
        self,
        mean: np.ndarray,
        sigma: np.ndarray,
        horizon_index: pd.DatetimeIndex,
        quantiles: tuple[float, ...],
    ) -> pd.DataFrame:
        """Build the wide quantile frame from a Normal predictive distribution."""
        data: dict[str, np.ndarray] = {}
        for q in quantiles:
            z = float(norm.ppf(q))
            vals = mean + z * sigma
            if self._log_applied:
                # Exponentiate back to price space (monotone, preserves ordering).
                vals = np.exp(vals)
            data[f"q{q}"] = vals

        frame = pd.DataFrame(data, index=horizon_index)
        frame.index.name = "target_start"
        # Guarantee monotone non-crossing quantiles row-wise (numerical safety).
        ordered_cols = [f"q{q}" for q in sorted(quantiles)]
        sorted_vals = np.sort(frame[ordered_cols].to_numpy(), axis=1)
        frame[ordered_cols] = sorted_vals
        return frame[[f"q{q}" for q in quantiles]]

    # ------------------------------------------------------- preprocessing
    def _prepare_target(self, y: pd.Series) -> pd.Series:
        """Coerce ``y`` to a clean, UTC daily-frequency, float Series."""
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series indexed by UTC datetimes.")
        y = y.copy()
        y.index = self._coerce_index(y.index)
        y = y[~y.index.duplicated(keep="last")].sort_index()
        y = pd.to_numeric(y, errors="coerce").dropna()
        if y.empty:
            return y
        # Resample to a regular daily frequency so SARIMAX's state space has a
        # well-defined step; interpolate small internal gaps only.
        self._freq = "D"
        y = y.asfreq("D")
        if y.isna().any():
            y = y.interpolate(method="time", limit_direction="both")
        return y.dropna()

    def _prepare_exog(
        self,
        exog: pd.DataFrame | None,
        target_index: pd.DatetimeIndex,
        fitting: bool,
    ) -> pd.DataFrame | None:
        """Align exogenous regressors to ``target_index`` (UTC daily)."""
        if fitting:
            if exog is None or exog.empty:
                self._exog_cols = []
                return None
            cols = self.exog_columns if self.exog_columns is not None else list(exog.columns)
            cols = [c for c in cols if c in exog.columns]
            if not cols:
                self._exog_cols = []
                return None
            self._exog_cols = cols
        else:
            if not self._exog_cols:
                return None  # model fit without exog -> forecast without exog

        if exog is None or exog.empty:
            # Model needs exog but none supplied for the horizon: broadcast the
            # last in-sample exog row forward (a neutral default near the data's
            # own level), falling back to zero only if we never saw one.
            if self._last_exog_row is not None:
                row = self._last_exog_row.reindex(self._exog_cols).fillna(0.0)
                aligned = pd.DataFrame(
                    [row.to_numpy(dtype=float)] * len(target_index),
                    index=target_index,
                    columns=self._exog_cols,
                )
            else:
                aligned = pd.DataFrame(
                    0.0, index=target_index, columns=self._exog_cols, dtype=float
                )
            return aligned

        ex = exog.copy()
        ex.index = self._coerce_index(ex.index)
        ex = ex[~ex.index.duplicated(keep="last")].sort_index()
        ex = ex.reindex(columns=self._exog_cols)
        ex = ex.apply(pd.to_numeric, errors="coerce")

        aligned = ex.reindex(target_index)
        # Fill horizon/training gaps conservatively without leaking the future.
        aligned = aligned.ffill().bfill().fillna(0.0).astype(float)
        aligned.index = target_index
        return aligned

    @staticmethod
    def _coerce_index(index) -> pd.DatetimeIndex:
        """Return a tz-aware (UTC) DatetimeIndex."""
        idx = pd.DatetimeIndex(index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        return idx

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"SarimaxForecaster(order={self.order}, "
            f"seasonal_order={self.seasonal_order}, "
            f"log_transform={self.log_transform}, use_garch={self.use_garch})"
        )
