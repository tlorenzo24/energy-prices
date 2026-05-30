"""Lag/rolling feature construction and the assembled feature frame.

LEAKAGE RULE: every feature for target time t must use only information available
strictly before gate closure for t. For day-ahead forecasting that means price
lags of >= 24h (we use D-1, D-2, D-3, D-7) and exogenous *forecasts* (not actuals)
aligned to t. The helpers here only ever look backwards via .shift(positive).
"""

from __future__ import annotations

import pandas as pd

from energy_prices.features.calendar import add_calendar_features

# Day-ahead-safe price lags (in periods of the series' own resolution is wrong;
# we lag by hours and resample-align upstream). Expressed in hours.
DEFAULT_LAG_HOURS: tuple[int, ...] = (24, 48, 72, 168)
DEFAULT_ROLL_WINDOWS: tuple[int, ...] = (24, 168)


def _periods_per_hour(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0
    delta = index.to_series().diff().median()
    if pd.isna(delta) or delta == pd.Timedelta(0):
        return 1.0
    return pd.Timedelta(hours=1) / delta


def add_lag_features(
    y: pd.Series, lag_hours: tuple[int, ...] = DEFAULT_LAG_HOURS
) -> pd.DataFrame:
    """Lagged copies of the target. Uses time-based shifting (resolution-aware)."""
    ppr = _periods_per_hour(y.index)
    out = pd.DataFrame(index=y.index)
    for h in lag_hours:
        periods = max(1, round(h * ppr))
        out[f"lag_{h}h"] = y.shift(periods)
    return out


def add_rolling_features(
    y: pd.Series, windows_hours: tuple[int, ...] = DEFAULT_ROLL_WINDOWS, min_lag_hours: int = 24
) -> pd.DataFrame:
    """Rolling mean/std/min/max, shifted by min_lag_hours to stay leak-safe."""
    ppr = _periods_per_hour(y.index)
    base_shift = max(1, round(min_lag_hours * ppr))
    shifted = y.shift(base_shift)
    out = pd.DataFrame(index=y.index)
    for w in windows_hours:
        win = max(1, round(w * ppr))
        out[f"roll_mean_{w}h"] = shifted.rolling(win, min_periods=max(1, win // 2)).mean()
        out[f"roll_std_{w}h"] = shifted.rolling(win, min_periods=max(1, win // 2)).std()
        out[f"roll_min_{w}h"] = shifted.rolling(win, min_periods=max(1, win // 2)).min()
        out[f"roll_max_{w}h"] = shifted.rolling(win, min_periods=max(1, win // 2)).max()
    return out


def build_feature_frame(
    y: pd.Series,
    exog: pd.DataFrame | None = None,
    tz: str = "Europe/Rome",
    lag_hours: tuple[int, ...] = DEFAULT_LAG_HOURS,
    roll_windows: tuple[int, ...] = DEFAULT_ROLL_WINDOWS,
    dropna: bool = True,
) -> pd.DataFrame:
    """Assemble calendar + lag + rolling + exogenous features for the target `y`.

    Returns a frame aligned to y's index. Exogenous columns are joined as-is
    (the caller is responsible for them being forecasts available at gate close).
    """
    base = pd.DataFrame(index=y.index)
    base = add_calendar_features(base, tz=tz)
    lags = add_lag_features(y, lag_hours)
    rolls = add_rolling_features(y, roll_windows)
    frame = pd.concat([base, lags, rolls], axis=1)
    if exog is not None and not exog.empty:
        frame = frame.join(exog, how="left")
    # Drop degenerate all-NaN columns first (e.g. a rolling-std window that
    # collapses to a single period on daily data) so they don't wipe every row
    # in the row-wise dropna below.
    frame = frame.dropna(axis=1, how="all")
    if dropna:
        frame = frame.dropna()
    return frame
