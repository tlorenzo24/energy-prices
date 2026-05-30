"""The Forecaster contract every model implements.

A Forecaster is fit on a history Series (price indexed by UTC delivery_start) plus
an optional exogenous DataFrame, and produces a probabilistic forecast: a wide
DataFrame indexed by future target_start with one column per requested quantile
(named 'q0.1', 'q0.5', ...). The point forecast is the q0.5 (median) column.

Keep implementations self-contained and <500 lines. If an optional heavy
dependency (e.g. epftoolbox) is missing, raise ModelUnavailable in __init__ so
the runner can skip the model gracefully.
"""

from __future__ import annotations

import abc
import datetime as dt

import pandas as pd

# Decile quantiles — good default for trader-facing probabilistic forecasts.
DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)


class ModelUnavailable(RuntimeError):
    """Raised when a model's optional dependencies are not installed."""


class ForecastResult:
    """Thin wrapper around the wide quantile DataFrame, with metadata."""

    def __init__(self, quantiles: pd.DataFrame, model_name: str, model_version: str) -> None:
        self.quantiles = quantiles  # index: target_start (UTC); cols: 'q0.1'...
        self.model_name = model_name
        self.model_version = model_version

    @property
    def point(self) -> pd.Series:
        """Median (q0.5) point forecast."""
        col = "q0.5" if "q0.5" in self.quantiles.columns else self.quantiles.columns[0]
        return self.quantiles[col]

    def to_rows(
        self,
        market: str,
        zone: str | None,
        resolution_minutes: int,
        run_at: dt.datetime,
    ) -> list[dict]:
        """Flatten to ForecastRepository.save() row dicts (one per quantile)."""
        rows: list[dict] = []
        for ts, record in self.quantiles.iterrows():
            for col, value in record.items():
                if pd.isna(value):
                    continue
                q = float(str(col).lstrip("q"))
                rows.append(
                    {
                        "run_at": run_at,
                        "market": market,
                        "zone": zone,
                        "target_start": ts.to_pydatetime(),
                        "resolution_minutes": resolution_minutes,
                        "model_name": self.model_name,
                        "model_version": self.model_version,
                        "quantile": q,
                        "value": float(value),
                    }
                )
        return rows


class Forecaster(abc.ABC):
    """Abstract base for all price forecasters."""

    name: str = "base"
    version: str = "0.1.0"

    @abc.abstractmethod
    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> Forecaster:
        """Train on a UTC-indexed price Series `y` and optional exogenous `exog`."""

    @abc.abstractmethod
    def predict(
        self,
        horizon_index: pd.DatetimeIndex,
        exog_future: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> ForecastResult:
        """Forecast for each timestamp in `horizon_index`, returning quantiles."""

    def fit_predict(
        self,
        y: pd.Series,
        horizon_index: pd.DatetimeIndex,
        exog: pd.DataFrame | None = None,
        exog_future: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> ForecastResult:
        self.fit(y, exog)
        return self.predict(horizon_index, exog_future, quantiles)
