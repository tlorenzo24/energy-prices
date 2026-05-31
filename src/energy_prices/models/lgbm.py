"""LightGBM quantile-regression forecaster — the pragmatic probabilistic workhorse.

Trains ONE :class:`lightgbm.LGBMRegressor` per requested quantile using
``objective="quantile"`` with the matching ``alpha``. Features come from the
leak-safe :func:`energy_prices.features.build.build_feature_frame` (calendar +
price lags >= 24h + rolling stats + optional exogenous forecasts).

Prediction over a future ``horizon_index`` reuses the training history: because
day-ahead horizons are short (24-48h) and every price lag is >= 24h, the lags
needed for the horizon rows are already present in the known history at predict
time. We therefore concatenate ``history + horizon`` on the target series,
rebuild the feature frame, and keep only the horizon rows. Calendar features are
deterministic; exogenous *forecasts* for the horizon are supplied via
``exog_future``.

Quantiles are sorted per row so the emitted quantiles never cross.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from energy_prices.features.build import (
    DEFAULT_LAG_HOURS,
    DEFAULT_ROLL_WINDOWS,
    build_feature_frame,
)
from energy_prices.models.base import DEFAULT_QUANTILES, Forecaster, ForecastResult

logger = logging.getLogger(__name__)


class LightGBMForecaster(Forecaster):
    """Gradient-boosted quantile regression forecaster (one model per quantile)."""

    name: str = "lightgbm"
    version: str = "0.1.0"

    def __init__(
        self,
        *,
        tz: str = "Europe/Rome",
        lag_hours: tuple[int, ...] = DEFAULT_LAG_HOURS,
        roll_windows: tuple[int, ...] = DEFAULT_ROLL_WINDOWS,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        random_state: int = 42,
        n_jobs: int = -1,
        **lgbm_kwargs: object,
    ) -> None:
        self.tz = tz
        self.lag_hours = tuple(lag_hours)
        self.roll_windows = tuple(roll_windows)
        self._params: dict[str, Any] = {
            "objective": "quantile",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbose": -1,
            **lgbm_kwargs,
        }

        # Populated by fit().
        self._models: dict[float, LGBMRegressor] = {}
        self._feature_cols: list[str] = []
        self._y_train: pd.Series | None = None
        self._exog_train: pd.DataFrame | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> LightGBMForecaster:
        """Train one quantile model per :data:`DEFAULT_QUANTILES` on ``y``."""
        y = self._coerce_series(y)
        if y.empty:
            raise ValueError("Cannot fit LightGBMForecaster on an empty series.")

        exog = self._coerce_exog(exog)
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
                "Feature frame is empty after dropna — history is too short for the "
                f"configured lags {self.lag_hours} / rolling windows {self.roll_windows}."
            )

        target = y.reindex(frame.index)
        self._feature_cols = list(frame.columns)
        X = frame.to_numpy(dtype=float)
        y_arr = target.to_numpy(dtype=float)

        self._models = {}
        for q in DEFAULT_QUANTILES:
            model = LGBMRegressor(alpha=float(q), **self._params)
            model.fit(X, y_arr)
            self._models[float(q)] = model

        # Keep history so predict() can build leak-safe future lag features.
        self._y_train = y
        self._exog_train = exog
        logger.info(
            "Fitted %s v%s: %d quantile models, %d features, %d training rows.",
            self.name,
            self.version,
            len(self._models),
            len(self._feature_cols),
            len(frame),
        )
        return self

    # -------------------------------------------------------------- predict
    def predict(
        self,
        horizon_index: pd.DatetimeIndex,
        exog_future: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> ForecastResult:
        """Forecast quantiles for each timestamp in ``horizon_index``."""
        if self._y_train is None or not self._models:
            raise RuntimeError("LightGBMForecaster.predict() called before fit().")

        horizon_index = self._coerce_index(horizon_index)
        if len(horizon_index) == 0:
            empty = pd.DataFrame(
                columns=[self._qcol(q) for q in quantiles],
                index=pd.DatetimeIndex([], tz="UTC", name=horizon_index.name),
            )
            return ForecastResult(empty, self.name, self.version)

        features = self._build_horizon_features(horizon_index, exog_future)

        # Predict each requested quantile; fall back to fitted models.
        cols = [self._qcol(q) for q in quantiles]
        preds = np.empty((len(features), len(quantiles)), dtype=float)
        X = features.to_numpy(dtype=float)
        for j, q in enumerate(quantiles):
            model = self._models.get(float(q))
            if model is None:
                model = LGBMRegressor(alpha=float(q), **self._params)
                model.fit(
                    self._fit_matrix(),
                    self._fit_target(),
                )
                self._models[float(q)] = model
                logger.debug("Trained on-demand quantile model for q=%.3f.", q)
            preds[:, j] = model.predict(X)

        # Enforce monotone, non-crossing quantiles by sorting each row.
        order = np.argsort([float(q) for q in quantiles])
        sorted_cols = preds[:, order]
        sorted_cols = np.sort(sorted_cols, axis=1)
        inverse = np.argsort(order)
        preds = sorted_cols[:, inverse]

        out = pd.DataFrame(preds, index=features.index, columns=cols)
        out = out.reindex(horizon_index)  # align to requested horizon order
        return ForecastResult(out, self.name, self.version)

    # ------------------------------------------------------------- helpers
    def _build_horizon_features(
        self,
        horizon_index: pd.DatetimeIndex,
        exog_future: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Assemble leak-safe feature rows for the horizon using known history."""
        assert self._y_train is not None  # for type-checkers; guarded in predict()

        # Extend the target series with NaNs over the (new) horizon timestamps so
        # build_feature_frame can compute calendar + lag/rolling features there.
        # Lags >= 24h reach back into real history; the NaN horizon values do not
        # leak because no lag/rolling window is shorter than 24h.
        new_steps = horizon_index.difference(self._y_train.index)
        y_ext = self._y_train
        if len(new_steps) > 0:
            filler = pd.Series(np.nan, index=new_steps, dtype=float)
            y_ext = pd.concat([self._y_train, filler])
        y_ext = y_ext[~y_ext.index.duplicated(keep="first")].sort_index()

        exog_ext = self._combine_exog(self._exog_train, exog_future)

        frame = build_feature_frame(
            y_ext,
            exog=exog_ext,
            tz=self.tz,
            lag_hours=self.lag_hours,
            roll_windows=self.roll_windows,
            dropna=False,  # keep horizon rows even though target is NaN there
        )
        frame = frame.reindex(self._feature_cols, axis=1)
        horizon_frame = frame.reindex(horizon_index)

        # Calendar features are always defined; lag/rolling/exog gaps -> fill so
        # LightGBM gets finite inputs (it can also handle NaN, but we keep it
        # deterministic and explicit for the contracted feature columns).
        missing = horizon_frame.isna().to_numpy().sum()
        if missing:
            logger.debug("Filling %d missing horizon feature cells.", int(missing))
            horizon_frame = self._fill_missing(horizon_frame)
        return horizon_frame

    def _fill_missing(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Forward/back-fill then fall back to training-feature medians, else 0."""
        filled = frame.ffill().bfill()
        if filled.isna().to_numpy().any():
            train_medians = self._feature_medians()
            filled = filled.fillna(train_medians)
        return filled.fillna(0.0)

    def _feature_medians(self) -> pd.Series:
        """Median of each training feature column (for filling horizon gaps)."""
        if self._y_train is None:
            return pd.Series(0.0, index=self._feature_cols)
        frame = build_feature_frame(
            self._y_train,
            exog=self._exog_train,
            tz=self.tz,
            lag_hours=self.lag_hours,
            roll_windows=self.roll_windows,
            dropna=True,
        )
        frame = frame.reindex(self._feature_cols, axis=1)
        return frame.median(numeric_only=True).reindex(self._feature_cols).fillna(0.0)

    def _fit_matrix(self) -> np.ndarray:
        frame = build_feature_frame(
            self._y_train,
            exog=self._exog_train,
            tz=self.tz,
            lag_hours=self.lag_hours,
            roll_windows=self.roll_windows,
            dropna=True,
        ).reindex(self._feature_cols, axis=1)
        return frame.to_numpy(dtype=float)

    def _fit_target(self) -> np.ndarray:
        assert self._y_train is not None
        frame = build_feature_frame(
            self._y_train,
            exog=self._exog_train,
            tz=self.tz,
            lag_hours=self.lag_hours,
            roll_windows=self.roll_windows,
            dropna=True,
        )
        return self._y_train.reindex(frame.index).to_numpy(dtype=float)

    @staticmethod
    def _combine_exog(
        exog_train: pd.DataFrame | None,
        exog_future: pd.DataFrame | None,
    ) -> pd.DataFrame | None:
        """Concatenate historical and future exogenous frames (future wins ties)."""
        parts = [p for p in (exog_train, exog_future) if p is not None and not p.empty]
        if not parts:
            return None
        if len(parts) == 1:
            combined = parts[0].copy()
        else:
            combined = pd.concat(parts)
            combined = combined[~combined.index.duplicated(keep="last")]
        return combined.sort_index()

    @staticmethod
    def _qcol(q: float) -> str:
        return f"q{float(q):g}"

    def _coerce_series(self, y: pd.Series) -> pd.Series:
        s = pd.Series(y).astype(float)
        s.index = self._coerce_index(s.index)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        return s.dropna()

    def _coerce_exog(self, exog: pd.DataFrame | None) -> pd.DataFrame | None:
        if exog is None or exog.empty:
            return None
        e = exog.copy()
        e.index = self._coerce_index(e.index)
        e = e[~e.index.duplicated(keep="last")].sort_index()
        return e

    @staticmethod
    def _coerce_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Ensure a timezone-aware (UTC) DatetimeIndex."""
        idx = pd.DatetimeIndex(index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        return idx
