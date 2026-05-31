"""Calendar features for Italian energy markets.

Cyclic encodings (hour, day-of-week, month), holiday flags (Italian national
holidays), and a heating-season flag useful for gas demand.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache

import holidays
import numpy as np
import pandas as pd


@lru_cache(maxsize=8)
def italian_holidays(start_year: int, end_year: int) -> frozenset[dt.date]:
    """Set of Italian national holiday dates in [start_year, end_year]."""
    it = holidays.Italy(years=range(start_year, end_year + 1))
    return frozenset(it.keys())


def _cyclic(values: pd.Series, period: int, prefix: str) -> pd.DataFrame:
    radians = 2.0 * np.pi * values / period
    return pd.DataFrame(
        {f"{prefix}_sin": np.sin(radians), f"{prefix}_cos": np.cos(radians)},
        index=values.index,
    )


def add_calendar_features(df: pd.DataFrame, tz: str = "Europe/Rome") -> pd.DataFrame:
    """Add calendar features. `df` must have a UTC DatetimeIndex.

    Features are computed in local (Italian) time so hour-of-day aligns with
    demand patterns and DST is handled correctly.
    """
    # NB: check the row count, not df.empty — a frame with rows but zero columns
    # (as build_feature_frame passes) is "empty" by pandas' definition, which
    # would silently skip every calendar feature.
    if len(df.index) == 0:
        return df.copy()

    out = df.copy()
    idx_utc = out.index
    if idx_utc.tz is None:
        idx_utc = idx_utc.tz_localize("UTC")
    local = idx_utc.tz_convert(tz)

    hour = pd.Series(local.hour, index=out.index)
    dow = pd.Series(local.dayofweek, index=out.index)
    month = pd.Series(local.month, index=out.index)
    doy = pd.Series(local.dayofyear, index=out.index)

    out = pd.concat(
        [
            out,
            _cyclic(hour, 24, "hour"),
            _cyclic(dow, 7, "dow"),
            _cyclic(month, 12, "month"),
            _cyclic(doy, 365, "doy"),
        ],
        axis=1,
    )
    out["hour"] = hour
    out["dow"] = dow
    out["is_weekend"] = (dow >= 5).astype(int)

    years = local.year
    hol = italian_holidays(int(years.min()), int(years.max()))
    dates = pd.Series([d.date() for d in local], index=out.index)
    out["is_holiday"] = dates.isin(hol).astype(int)

    # Heating season (Oct–Mar) — gas demand driver.
    out["is_heating_season"] = month.isin([10, 11, 12, 1, 2, 3]).astype(int)
    return out
