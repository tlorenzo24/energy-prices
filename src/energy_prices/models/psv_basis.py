"""PSV = TTF + basis cointegration forecaster for Italian day-ahead gas.

The Italian gas hub (PSV) and the European benchmark (Dutch TTF) are *cointegrated*:
on the ingested history their levels are highly correlated (corr ~0.94, OLS R^2
~0.89) and individually carry a unit root, while their spread — the **basis**
``b_t = PSV_t - TTF_t`` — is stationary (ADF p ~ 0.001, mean ~2.4 EUR/MWh,
std ~3.1, AR(1) ~0.34). That is the textbook error-correction setup, so instead
of extrapolating the short, noisy PSV series on its own we:

1. forecast **TTF** with a robust log-SARIMAX (its own deep history),
2. forecast the **stationary, mean-reverting basis** with a closed-form AR(1),
3. reconstruct ``PSV_hat = TTF_hat + basis_hat`` and combine the two predictive
   variances (assumed independent) into Normal quantiles.

Leak-safety: TTF is **forecast internally** from the TTF history supplied at fit
time (the ``ttf`` column of ``exog``); the realised future TTF that the
walk-forward harness places in ``exog_future`` is deliberately ignored, because
tomorrow's TTF close is not known at the day-ahead gate. If no usable TTF history
is supplied (or the overlap with PSV is too short), the model degrades gracefully
to a plain PSV SARIMAX so the runner never loses a gas forecast.

All datetimes are UTC and tz-aware; prices are EUR/MWh.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

from energy_prices.models.base import (
    DEFAULT_QUANTILES,
    Forecaster,
    ForecastResult,
)
from energy_prices.models.gas_sarimax import SarimaxForecaster

logger = logging.getLogger(__name__)

# z-score of the 0.9 quantile — used to back out an effective Normal sigma from
# the TTF model's (q0.1, q0.9) predictive spread.
_Z90 = float(norm.ppf(0.9))
# Floor on any predictive std (EUR/MWh) so degenerate fits still yield a spread.
_MIN_SIGMA = 1e-6
# AR(1) persistence is clipped to this band: a price spread should not be an
# explosive (>=1) or strongly oscillatory process.
_PHI_LO, _PHI_HI = -0.5, 0.98


class PsvBasisForecaster(Forecaster):
    """Cointegration forecaster: ``PSV = TTF_forecast + mean-reverting basis``.

    Parameters
    ----------
    ttf_column:
        Name of the exogenous column carrying the TTF benchmark history.
        Defaults to ``"ttf"``.
    min_overlap:
        Minimum number of overlapping PSV/TTF observations required to estimate
        the basis; below this the model falls back to a plain PSV SARIMAX.
    log_ttf:
        Fit the TTF leg on ``log(TTF)`` (recommended for a positive, occasionally
        spiky benchmark). Passed through to the inner :class:`SarimaxForecaster`.
    """

    name: str = "psv_basis"
    version: str = "0.1.0"

    def __init__(
        self,
        ttf_column: str = "ttf",
        min_overlap: int = 30,
        log_ttf: bool = True,
    ) -> None:
        self.ttf_column = ttf_column
        self.min_overlap = int(min_overlap)
        self.log_ttf = bool(log_ttf)

        # Populated by fit().
        self._ttf_model: SarimaxForecaster | None = None
        self._basis_mu: float = 0.0
        self._basis_phi: float = 0.0
        self._basis_sigma_eps: float = 0.0
        self._basis_uncond_var: float = 0.0
        self._basis_last: float = 0.0
        # Graceful-degradation path: plain PSV SARIMAX when TTF is unusable.
        self._fallback: SarimaxForecaster | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> PsvBasisForecaster:
        """Fit the TTF leg and the PSV-TTF basis from ``y`` and ``exog['ttf']``."""
        psv = self._clean_series(y)
        if psv.empty:
            raise ValueError("PsvBasisForecaster.fit: empty PSV series after cleaning.")

        ttf = self._extract_ttf(exog)
        if ttf is None:
            return self._fit_fallback(psv, "no usable TTF history in exog")

        # Enforce leak-safety at the model boundary (not just by caller discipline):
        # the TTF leg is trained only on TTF up to the last PSV training timestamp,
        # so a caller that hands over an un-reindexed, future-containing TTF column
        # can never leak a post-origin TTF level into the fit.
        ttf = ttf[ttf.index <= psv.index.max()]
        if ttf.empty:
            return self._fit_fallback(psv, "no TTF history at/under the PSV training horizon")

        # Align PSV and TTF on a common daily grid (carry TTF over non-trading
        # days) and form the basis where both are present.
        joined = pd.concat(
            {"psv": psv, "ttf": ttf.reindex(psv.index).ffill().bfill()}, axis=1
        ).dropna()
        if len(joined) < self.min_overlap:
            return self._fit_fallback(
                psv, f"PSV/TTF overlap {len(joined)} < min_overlap {self.min_overlap}"
            )

        # TTF leg: a robust log-SARIMAX on the TTF history itself (GARCH widening
        # off — the basis variance dominates the short-horizon spread and it keeps
        # the many-window backtest fast and deterministic).
        try:
            self._ttf_model = SarimaxForecaster(
                log_transform=self.log_ttf, use_garch=False
            ).fit(ttf)
        except Exception as exc:  # noqa: BLE001 - degrade rather than fail the run
            return self._fit_fallback(psv, f"TTF SARIMAX fit failed: {exc}")

        # Basis leg: closed-form AR(1) with a constant mean (Yule-Walker style).
        self._estimate_basis_ar1(joined["psv"] - joined["ttf"])
        self._fallback = None
        logger.info(
            "PsvBasis fitted: overlap=%d basis mu=%.3f phi=%.3f sigma_eps=%.3f",
            len(joined),
            self._basis_mu,
            self._basis_phi,
            self._basis_sigma_eps,
        )
        return self

    # -------------------------------------------------------------- predict
    def predict(
        self,
        horizon_index: pd.DatetimeIndex,
        exog_future: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> ForecastResult:
        """Forecast PSV quantiles by recombining the TTF and basis forecasts.

        ``exog_future`` is intentionally ignored: the realised future TTF is not
        known at the day-ahead gate, so TTF is forecast internally from the
        history seen at fit time (leak-safe).
        """
        horizon_index = self._coerce_index(horizon_index)
        if self._fallback is not None:
            # Attribute the degraded forecast to THIS model, not the inner SARIMAX,
            # so a gas fallback never collides with the standalone TTF "sarimax".
            res = self._fallback.predict(horizon_index, None, quantiles)
            return ForecastResult(res.quantiles, self.name, self.version)
        if self._ttf_model is None:
            raise RuntimeError("PsvBasisForecaster.predict called before fit().")
        if len(horizon_index) == 0:
            empty = pd.DataFrame(
                columns=[f"q{q}" for q in quantiles],
                index=pd.DatetimeIndex([], tz="UTC", name="target_start"),
            )
            return ForecastResult(empty, self.name, self.version)

        steps = len(horizon_index)
        ttf_mean, ttf_sigma = self._forecast_ttf(horizon_index)
        basis_mean, basis_var = self._forecast_basis(steps)

        psv_mean = ttf_mean + basis_mean
        psv_sigma = np.clip(np.sqrt(ttf_sigma**2 + basis_var), _MIN_SIGMA, None)

        frame = self._quantile_frame(psv_mean, psv_sigma, horizon_index, quantiles)
        return ForecastResult(frame, self.name, self.version)

    # --------------------------------------------------------- TTF / basis
    def _forecast_ttf(self, horizon_index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        """Return per-step TTF (mean, sigma) from the inner SARIMAX quantiles.

        sigma is backed out of the (q0.1, q0.9) spread as a symmetric Normal
        proxy in price space — adequate for the short day-ahead/multi-day gas
        horizon and validated by the backtest's interval coverage.
        """
        assert self._ttf_model is not None
        res = self._ttf_model.predict(horizon_index, quantiles=(0.1, 0.5, 0.9))
        q = res.quantiles.reindex(horizon_index)
        mean = q["q0.5"].to_numpy(dtype=float)
        spread = (q["q0.9"] - q["q0.1"]).to_numpy(dtype=float)
        sigma = np.clip(spread / (2.0 * _Z90), _MIN_SIGMA, None)
        # Defensive: a degenerate SARIMAX step can yield NaN — carry the last
        # finite mean and a floor sigma so the recombination never propagates NaN.
        mean = pd.Series(mean).ffill().bfill().fillna(0.0).to_numpy()
        sigma = np.where(np.isfinite(sigma), sigma, _MIN_SIGMA)
        return mean, sigma

    def _forecast_basis(self, steps: int) -> tuple[np.ndarray, np.ndarray]:
        """AR(1) h-step mean reversion and variance for the basis.

        mean_h = mu + phi^h (b_last - mu);  var_h = sigma_eps^2 (1 - phi^{2h})/(1 - phi^2),
        which grows from the one-step innovation variance up to the unconditional
        basis variance as the horizon lengthens.
        """
        h = np.arange(1, steps + 1, dtype=float)
        phi = self._basis_phi
        mean = self._basis_mu + (phi**h) * (self._basis_last - self._basis_mu)
        if abs(phi) < 1.0 - 1e-9 and self._basis_sigma_eps > 0:
            var = self._basis_sigma_eps**2 * (1.0 - phi ** (2.0 * h)) / (1.0 - phi**2)
        else:
            var = np.full(steps, self._basis_uncond_var, dtype=float)
        # The AR(1) variance grows toward — and is capped at — the unconditional
        # basis variance; never let float drift push a step above it.
        upper = self._basis_uncond_var if self._basis_uncond_var > 0 else np.inf
        var = np.clip(var, 0.0, upper)
        return mean, var

    def _estimate_basis_ar1(self, basis: pd.Series) -> None:
        """Estimate AR(1) (mu, phi, sigma_eps) of the stationary basis in closed form."""
        b = basis.dropna().astype(float)
        mu = float(b.mean())
        self._basis_mu = mu
        self._basis_last = float(b.iloc[-1])
        self._basis_uncond_var = float(b.var(ddof=1)) if len(b) > 1 else 0.0

        bd = (b - mu).to_numpy()
        if len(bd) >= 3 and float(np.dot(bd[:-1], bd[:-1])) > 0:
            phi = float(np.dot(bd[1:], bd[:-1]) / np.dot(bd[:-1], bd[:-1]))
            phi = float(np.clip(phi, _PHI_LO, _PHI_HI))
            resid = bd[1:] - phi * bd[:-1]
            sigma_eps = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
        else:
            phi, sigma_eps = 0.0, float(np.sqrt(self._basis_uncond_var))
        self._basis_phi = phi
        self._basis_sigma_eps = sigma_eps

    # ----------------------------------------------------------- fallback
    def _fit_fallback(self, psv: pd.Series, reason: str) -> PsvBasisForecaster:
        """Degrade to a plain PSV SARIMAX when the TTF/basis path is unusable."""
        logger.info("PsvBasis falling back to plain PSV SARIMAX: %s", reason)
        self._ttf_model = None
        self._fallback = SarimaxForecaster(log_transform=True, use_garch=True).fit(psv)
        return self

    # ----------------------------------------------------------- helpers
    def _extract_ttf(self, exog: pd.DataFrame | None) -> pd.Series | None:
        """Pull a clean TTF history Series from the exog frame, or None."""
        if exog is None or self.ttf_column not in getattr(exog, "columns", []):
            return None
        ttf = pd.to_numeric(exog[self.ttf_column], errors="coerce").dropna()
        if ttf.empty:
            return None
        ttf.index = self._coerce_index(ttf.index)
        ttf = ttf[~ttf.index.duplicated(keep="last")].sort_index()
        return ttf.astype(float).rename("ttf")

    def _clean_series(self, y: pd.Series) -> pd.Series:
        s = pd.Series(y).astype(float)
        s.index = self._coerce_index(s.index)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        return s.dropna()

    def _quantile_frame(
        self,
        mean: np.ndarray,
        sigma: np.ndarray,
        horizon_index: pd.DatetimeIndex,
        quantiles: tuple[float, ...],
    ) -> pd.DataFrame:
        """Build the wide quantile frame from a Normal predictive distribution."""
        data = {f"q{q}": mean + float(norm.ppf(q)) * sigma for q in quantiles}
        frame = pd.DataFrame(data, index=horizon_index)
        frame.index.name = "target_start"
        ordered = [f"q{q}" for q in sorted(quantiles)]
        frame[ordered] = np.sort(frame[ordered].to_numpy(), axis=1)
        return frame[[f"q{q}" for q in quantiles]]

    @staticmethod
    def _coerce_index(index) -> pd.DatetimeIndex:
        idx = pd.DatetimeIndex(index)
        return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"PsvBasisForecaster(ttf_column={self.ttf_column!r}, "
            f"min_overlap={self.min_overlap}, log_ttf={self.log_ttf})"
        )
