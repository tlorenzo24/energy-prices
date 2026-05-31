"""Forecasting models. Every model implements the `Forecaster` interface in base.py."""

from energy_prices.models.base import DEFAULT_QUANTILES, Forecaster, ForecastResult

__all__ = ["Forecaster", "ForecastResult", "DEFAULT_QUANTILES"]
