"""Quantile-averaging ensemble forecaster.

`EnsembleForecaster` combines several `Forecaster` members by fitting each one
and averaging their predicted quantile columns on a common horizon. Members that
cannot be constructed or fit because their optional heavy dependencies are
missing (raising :class:`ModelUnavailable`) are skipped gracefully, so the
ensemble degrades to whatever models are actually available in the environment.

The default ensemble targets day-ahead electricity prices (LEAR + LightGBM); the
:meth:`EnsembleForecaster.for_gas` classmethod builds the gas-oriented variant
(SARIMAX + LightGBM). All datetimes are UTC and prices are EUR/MWh.
"""

from __future__ import annotations

import importlib
import logging

import numpy as np
import pandas as pd

from energy_prices.models.base import (
    DEFAULT_QUANTILES,
    Forecaster,
    ForecastResult,
    ModelUnavailable,
)

logger = logging.getLogger(__name__)

# Candidate import locations for member models. Each entry maps a class name to
# the module paths (most-likely first) where it might live. We try them in order
# so the ensemble stays robust to the exact module file names of sibling models.
_MEMBER_MODULES: dict[str, tuple[str, ...]] = {
    "LearForecaster": (
        "energy_prices.models.lear",
    ),
    "LightGBMForecaster": (
        "energy_prices.models.lightgbm",
        "energy_prices.models.lgbm",
        "energy_prices.models.gbm",
    ),
    "SarimaxForecaster": (
        "energy_prices.models.gas_sarimax",
        "energy_prices.models.sarimax",
    ),
}


def _load_member(class_name: str) -> Forecaster:
    """Lazily import and instantiate a member model by class name.

    Raises :class:`ModelUnavailable` if the model's module/class cannot be
    imported or its optional dependencies are missing, so the caller can skip it.
    """
    last_error: Exception | None = None
    for module_path in _MEMBER_MODULES.get(class_name, ()):  # try known locations
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:  # module not present (yet) — try next candidate
            last_error = exc
            continue
        try:
            cls = getattr(module, class_name)
        except AttributeError as exc:
            last_error = exc
            continue
        return cls()  # may raise ModelUnavailable for missing heavy deps
    raise ModelUnavailable(
        f"Could not load member model {class_name!r}: {last_error}"
    )


class EnsembleForecaster(Forecaster):
    """Average the quantile forecasts of several member `Forecaster`s.

    Parameters
    ----------
    members:
        Explicit list of member forecasters. When ``None`` the default
        electricity ensemble ``[LearForecaster(), LightGBMForecaster()]`` is
        constructed lazily; members whose optional dependencies are missing are
        skipped.
    """

    name: str = "ensemble"
    version: str = "0.1.0"

    def __init__(self, members: list[Forecaster] | None = None) -> None:
        if members is None:
            members = self._build_default_members(
                ["LearForecaster", "LightGBMForecaster"]
            )
        self.members: list[Forecaster] = list(members)
        # Members that successfully fit; populated in fit().
        self._fitted: list[Forecaster] = []

    @staticmethod
    def _build_default_members(class_names: list[str]) -> list[Forecaster]:
        """Instantiate default members, skipping any that are unavailable."""
        built: list[Forecaster] = []
        for class_name in class_names:
            try:
                built.append(_load_member(class_name))
            except ModelUnavailable as exc:
                logger.warning("Skipping unavailable ensemble member %s: %s", class_name, exc)
        return built

    @classmethod
    def for_gas(cls) -> EnsembleForecaster:
        """Default gas ensemble: SARIMAX + LightGBM (unavailable members skipped)."""
        return cls(cls._build_default_members(["SarimaxForecaster", "LightGBMForecaster"]))

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> EnsembleForecaster:
        """Fit every member; members raising `ModelUnavailable` are skipped."""
        self._fitted = []
        for member in self.members:
            member_name = getattr(member, "name", member.__class__.__name__)
            try:
                member.fit(y, exog)
            except ModelUnavailable as exc:
                logger.warning("Ensemble member %s unavailable during fit, skipping: %s", member_name, exc)
                continue
            self._fitted.append(member)
        if not self._fitted:
            raise ModelUnavailable(
                "No ensemble members could be fitted (all members unavailable)."
            )
        return self

    def predict(
        self,
        horizon_index: pd.DatetimeIndex,
        exog_future: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> ForecastResult:
        """Average member quantile forecasts over a common horizon.

        Each fitted member is asked to predict the same ``horizon_index`` and
        ``quantiles``. The resulting wide quantile frames are aligned on the
        horizon index and quantile columns, then averaged cell-by-cell ignoring
        NaNs. The averaged quantiles are re-sorted across columns so they remain
        non-crossing.
        """
        if not self._fitted:
            raise ModelUnavailable(
                "EnsembleForecaster has no fitted members; call fit() first."
            )

        member_results: list[ForecastResult] = []
        for member in self._fitted:
            member_name = getattr(member, "name", member.__class__.__name__)
            try:
                result = member.predict(horizon_index, exog_future, quantiles)
            except ModelUnavailable as exc:
                logger.warning("Ensemble member %s unavailable during predict, skipping: %s", member_name, exc)
                continue
            member_results.append(result)

        if not member_results:
            raise ModelUnavailable("No ensemble members produced a forecast.")

        # Single available member: re-wrap in the ensemble's identity so the
        # persisted model_name is always 'ensemble' (matches the >=2-member path
        # and the CalibratedForecaster convention). Quantiles are already
        # non-crossing from the member, so no re-sorting needed.
        if len(member_results) == 1:
            return ForecastResult(
                member_results[0].quantiles,
                model_name=self.name,
                model_version=self.version,
            )

        averaged = self._average_quantiles(member_results, horizon_index, quantiles)
        averaged = self._sort_non_crossing(averaged)
        return ForecastResult(averaged, model_name=self.name, model_version=self.version)

    @staticmethod
    def _quantile_columns(quantiles: tuple[float, ...]) -> list[str]:
        """Canonical column names ('q0.1', ...) for the requested quantiles."""
        return [f"q{float(q):g}" for q in quantiles]

    def _average_quantiles(
        self,
        member_results: list[ForecastResult],
        horizon_index: pd.DatetimeIndex,
        quantiles: tuple[float, ...],
    ) -> pd.DataFrame:
        """Cell-wise mean of member quantile frames, aligned to a common grid."""
        columns = self._quantile_columns(quantiles)
        # Stack aligned frames; align on the requested horizon and quantile cols.
        aligned = []
        for result in member_results:
            frame = result.quantiles.reindex(index=horizon_index, columns=columns)
            aligned.append(frame.astype(float))
        # np.nanmean over the member axis ignores members missing a given cell.
        stacked = np.stack([frame.to_numpy(dtype=float) for frame in aligned], axis=0)
        with np.errstate(invalid="ignore"):
            mean_values = np.nanmean(stacked, axis=0)
        return pd.DataFrame(mean_values, index=horizon_index, columns=columns)

    @staticmethod
    def _sort_non_crossing(quantiles_frame: pd.DataFrame) -> pd.DataFrame:
        """Ensure quantile columns are monotonically non-decreasing per row.

        Averaging independent member quantiles can in principle produce crossing
        quantiles; sorting each row's values across the quantile columns enforces
        the non-crossing constraint while preserving NaNs.
        """
        if quantiles_frame.empty:
            return quantiles_frame
        # Order columns by ascending quantile level so the smallest sorted value
        # maps to the lowest level regardless of the caller's column order, then
        # write the sorted values back into those same positions (leaving the
        # frame's overall column order unchanged). Mirrors CalibratedForecaster.
        levels: list[tuple[float, str]] = []
        for c in quantiles_frame.columns:
            try:
                levels.append((float(str(c).lstrip("q")), c))
            except ValueError:
                continue
        if not levels:
            return quantiles_frame
        ordered_cols = [c for _, c in sorted(levels)]
        out = quantiles_frame.copy()
        # np.sort pushes NaNs to the end within the ascending subset.
        out[ordered_cols] = np.sort(out[ordered_cols].to_numpy(dtype=float), axis=1)
        return out
