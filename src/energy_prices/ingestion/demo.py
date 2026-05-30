"""Synthetic DEMO data generator so the dashboard runs with ZERO credentials.

`seed_demo()` fabricates a realistic, fully deterministic history of Italian
electricity & gas prices plus the exogenous drivers the models expect, and
upserts everything through the repositories with ``source="demo"`` so it can
never collide with real ingested data.

The electricity model reproduces the salient features of Italian zonal MGP
prices: a daily double-peak (morning + evening) shape, weekday > weekend
levels, a smooth seasonal level, zonal offsets (islands SICI/SARD priced
higher, NORD as the cheap baseline), spatially correlated noise, occasional
positive price spikes and rare negative prices during sunny midday hours.
The PUN pseudo-zone is the ``PUN_ZONE_WEIGHTS``-weighted average of the seven
zonal series. Gas day-ahead (PSV) is a seasonal random walk with fat-tailed
jumps; TTF sits a small basis below PSV. All series use a single numpy
``Generator`` seeded from the integer ``seed`` argument for reproducibility.

Everything is hourly except gas/TTF which are daily. Datetimes are UTC and
tz-aware; prices are EUR/MWh.
"""

from __future__ import annotations

import datetime as dt
import logging
import math

import numpy as np
import pandas as pd

from energy_prices.config.enums import (
    MARKET_ZONES,
    PUN_ZONE_WEIGHTS,
    Market,
    Resolution,
    Zone,
)
from energy_prices.storage.repositories import (
    ExogenousRepository,
    PriceRepository,
)

logger = logging.getLogger(__name__)

SOURCE = "demo"
_BATCH = 5000

# Base electricity level (EUR/MWh) and per-zone additive offsets. NORD is the
# cheap, well-interconnected baseline; the islands carry a structural premium.
_BASE_ELEC = 110.0
_ZONE_OFFSET: dict[Zone, float] = {
    Zone.NORD: 0.0,
    Zone.CNOR: 4.0,
    Zone.CSUD: 6.0,
    Zone.SUD: 3.0,
    Zone.CALA: 5.0,
    Zone.SICI: 18.0,
    Zone.SARD: 12.0,
}
# How strongly each zone shares the common (system-wide) shock vs. its own.
_ZONE_BETA: dict[Zone, float] = {
    Zone.NORD: 0.95,
    Zone.CNOR: 0.90,
    Zone.CSUD: 0.88,
    Zone.SUD: 0.85,
    Zone.CALA: 0.80,
    Zone.SICI: 0.70,
    Zone.SARD: 0.72,
}

# Indicative zonal peak loads (MW) for the synthetic load_forecast series.
_ZONE_PEAK_LOAD: dict[Zone, float] = {
    Zone.NORD: 17000.0,
    Zone.CNOR: 4500.0,
    Zone.CSUD: 6500.0,
    Zone.SUD: 4000.0,
    Zone.CALA: 1800.0,
    Zone.SICI: 2600.0,
    Zone.SARD: 1700.0,
}
# Indicative installed wind+solar capacity (MW) per zone.
_ZONE_RENEWABLE_CAP: dict[Zone, float] = {
    Zone.NORD: 9000.0,
    Zone.CNOR: 2500.0,
    Zone.CSUD: 5500.0,
    Zone.SUD: 6000.0,
    Zone.CALA: 2200.0,
    Zone.SICI: 3000.0,
    Zone.SARD: 2400.0,
}


def _hour_of_day(idx: pd.DatetimeIndex, tz: str) -> np.ndarray:
    """Local clock hour (0-23) as float, for shaping the daily profile."""
    local = idx.tz_convert(tz)
    return local.hour.to_numpy(dtype=float) + local.minute.to_numpy(dtype=float) / 60.0


def _seasonal_level(idx: pd.DatetimeIndex, amplitude: float, peak_doy: int) -> np.ndarray:
    """Annual cosine: max near `peak_doy`, min half a year away."""
    doy = idx.dayofyear.to_numpy(dtype=float)
    return amplitude * np.cos(2.0 * math.pi * (doy - peak_doy) / 365.25)


def _daily_double_peak(hours: np.ndarray) -> np.ndarray:
    """Normalised intraday shape: trough overnight, morning + (taller) evening peak."""
    morning = np.exp(-0.5 * ((hours - 8.5) / 2.2) ** 2)
    evening = np.exp(-0.5 * ((hours - 19.5) / 2.4) ** 2)
    midday_dip = -0.35 * np.exp(-0.5 * ((hours - 14.0) / 2.5) ** 2)
    return 0.85 * morning + 1.0 * evening + midday_dip


def _solar_shape(hours: np.ndarray) -> np.ndarray:
    """Daylight bell centred on solar noon, zero at night (in [0, 1])."""
    bell = np.exp(-0.5 * ((hours - 13.0) / 3.0) ** 2)
    bell[(hours < 6.0) | (hours > 20.0)] = 0.0
    return bell


def _hourly_index(start: dt.datetime, end: dt.datetime) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="h", tz="UTC", inclusive="left")


def _daily_index(start_day: dt.date, end_day: dt.date) -> pd.DatetimeIndex:
    return pd.date_range(
        start=pd.Timestamp(start_day, tz="UTC"),
        end=pd.Timestamp(end_day, tz="UTC"),
        freq="D",
        tz="UTC",
        inclusive="left",
    )


def _flush(repo: PriceRepository | ExogenousRepository, buf: list[dict]) -> int:
    """Upsert any accumulated rows in capped batches; return rows written."""
    written = 0
    while buf:
        chunk, del_count = buf[:_BATCH], min(_BATCH, len(buf))
        written += repo.upsert(chunk)
        del buf[:del_count]
    return written


def _build_electricity(
    idx: pd.DatetimeIndex, rng: np.random.Generator, tz: str
) -> dict[Zone, np.ndarray]:
    """Generate correlated zonal hourly prices (EUR/MWh) for all physical zones."""
    n = len(idx)
    hours = _hour_of_day(idx, tz)
    dow = idx.tz_convert(tz).dayofweek.to_numpy()  # 0=Mon..6=Sun
    is_weekend = dow >= 5

    profile = _daily_double_peak(hours)  # ~[-0.35, 1.0]
    season = _seasonal_level(idx, amplitude=22.0, peak_doy=15)  # winter-heavy
    summer_bump = 14.0 * np.exp(-0.5 * ((idx.dayofyear.to_numpy(dtype=float) - 205) / 28.0) ** 2)
    weekend_adj = np.where(is_weekend, -16.0, 0.0)

    # System-wide AR(1) shock shared (scaled by beta) across zones.
    common = np.zeros(n)
    eps = rng.normal(0.0, 9.0, size=n)
    rho = 0.82
    for t in range(1, n):
        common[t] = rho * common[t - 1] + eps[t]

    # Sparse positive spikes (scarcity) and rare negative midday prices (solar glut).
    spike = np.zeros(n)
    spike_mask = rng.random(n) < 0.004
    spike[spike_mask] = rng.gamma(shape=2.0, scale=45.0, size=spike_mask.sum())

    solar = _solar_shape(hours)
    # Rare negative prices during high-solar midday hours (solar glut), more
    # likely at the weekend when demand is low.
    neg_prob = np.where(is_weekend, 0.16, 0.07)
    neg_mask = (rng.random(n) < neg_prob) & (solar > 0.6)
    neg_pull = np.where(neg_mask, -solar * rng.uniform(60.0, 130.0, size=n), 0.0)

    out: dict[Zone, np.ndarray] = {}
    for zone in MARKET_ZONES:
        beta = _ZONE_BETA[zone]
        idio = rng.normal(0.0, 4.0, size=n)
        level = (
            _BASE_ELEC
            + _ZONE_OFFSET[zone]
            + season
            + summer_bump
            + weekend_adj
        )
        intraday = 30.0 * profile  # EUR/MWh swing across the day
        # Islands react more to solar (lower interconnection): deeper midday troughs.
        island_solar = -solar * (10.0 if zone in (Zone.SICI, Zone.SARD) else 4.0)
        price = level + intraday + beta * common + idio + island_solar + spike + neg_pull
        # Clip to a sane band; allow modest negatives but no extreme tails.
        out[zone] = np.clip(price, -150.0, 900.0)
    return out


def _build_pun(zonal: dict[Zone, np.ndarray]) -> np.ndarray:
    """PUN_ZONE_WEIGHTS-weighted average of the zonal price arrays."""
    total_w = sum(PUN_ZONE_WEIGHTS.get(z, 0.0) for z in zonal)
    acc = np.zeros_like(next(iter(zonal.values())))
    for zone, arr in zonal.items():
        acc += PUN_ZONE_WEIGHTS.get(zone, 0.0) * arr
    return acc / total_w if total_w else acc


def _build_gas_daily(
    didx: pd.DatetimeIndex, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return (psv_gas, ttf) daily arrays in EUR/MWh.

    PSV (Italian day-ahead, market=GAS_DAYAHEAD) is a persistent random walk
    around a winter-heavy seasonal level with fat-tailed jumps; TTF sits a
    small positive basis below PSV (PSV = TTF + spread).
    """
    n = len(didx)
    season = _seasonal_level(didx, amplitude=11.0, peak_doy=10)  # cold-season premium
    base = 35.0 + season

    # Mean-reverting random walk for the stochastic component.
    walk = np.zeros(n)
    kappa = 0.05  # reversion speed toward 0
    for t in range(1, n):
        shock = rng.normal(0.0, 1.4)
        if rng.random() < 0.02:  # fat-tailed jump
            shock += rng.normal(0.0, 9.0)
        walk[t] = walk[t - 1] * (1.0 - kappa) + shock

    psv = np.clip(base + walk, 5.0, 250.0)
    # TTF below PSV by a slowly varying transport/basis spread (EUR/MWh).
    spread = 2.2 + 0.8 * np.sin(2.0 * math.pi * didx.dayofyear.to_numpy(dtype=float) / 365.25)
    ttf = np.clip(psv - spread, 2.0, 245.0)
    return psv, ttf


def _build_exogenous(
    idx: pd.DatetimeIndex,
    didx: pd.DatetimeIndex,
    rng: np.random.Generator,
    tz: str,
) -> list[dict]:
    """Assemble exogenous observation rows (load, wind+solar, storage, HDD/CDD, temp)."""
    rows: list[dict] = []
    hours = _hour_of_day(idx, tz)
    dow = idx.tz_convert(tz).dayofweek.to_numpy()
    is_weekend = dow >= 5
    doy = idx.dayofyear.to_numpy(dtype=float)

    # Daily load shape (double hump) and seasonal heating/cooling demand.
    load_shape = 0.62 + 0.38 * _daily_double_peak(hours) / 1.0
    load_season = 1.0 + 0.12 * np.cos(2.0 * math.pi * (doy - 15) / 365.25)
    load_summer = 1.0 + 0.10 * np.exp(-0.5 * ((doy - 205) / 30.0) ** 2)
    wk = np.where(is_weekend, 0.86, 1.0)

    solar = _solar_shape(hours)
    # Wind: smooth AR(1) [0,1] factor, slightly stronger in winter.
    wind = np.zeros(len(idx))
    we = rng.normal(0.0, 0.18, size=len(idx))
    for t in range(1, len(idx)):
        wind[t] = np.clip(0.85 * wind[t - 1] + we[t], -0.6, 1.2)
    wind_range = float(np.ptp(wind)) or 1.0
    wind = (wind - wind.min()) / wind_range
    wind_season = 1.0 + 0.25 * np.cos(2.0 * math.pi * (doy - 15) / 365.25)

    for zone in MARKET_ZONES:
        zname = zone.value
        peak = _ZONE_PEAK_LOAD[zone]
        load = peak * load_shape * load_season * load_summer * wk
        load *= 1.0 + rng.normal(0.0, 0.02, size=len(idx))
        cap = _ZONE_RENEWABLE_CAP[zone]
        # Islands are sunnier; NORD is wind-poor.
        solar_w = 1.25 if zone in (Zone.SICI, Zone.SARD, Zone.SUD) else 1.0
        wind_w = 0.5 if zone is Zone.NORD else 1.0
        renew = cap * (0.55 * solar * solar_w + 0.45 * wind * wind_season * wind_w)
        renew = np.clip(renew, 0.0, None)
        for ts, lv, rv in zip(idx, load, renew):
            pyts = ts.to_pydatetime()
            rows.append(
                {
                    "series": "load_forecast",
                    "zone": zname,
                    "valid_start": pyts,
                    "resolution_minutes": int(Resolution.HOUR),
                    "value": float(lv),
                    "source": SOURCE,
                    "unit": "MW",
                }
            )
            rows.append(
                {
                    "series": "wind_solar_forecast",
                    "zone": zname,
                    "valid_start": pyts,
                    "resolution_minutes": int(Resolution.HOUR),
                    "value": float(rv),
                    "source": SOURCE,
                    "unit": "MW",
                }
            )

    # --- Daily national fundamentals ---
    ddoy = didx.dayofyear.to_numpy(dtype=float)
    # Population-weighted Italian temperature (deg C): warm summer, cool winter.
    temp = 15.5 - 9.5 * np.cos(2.0 * math.pi * (ddoy - 200) / 365.25)
    temp += rng.normal(0.0, 1.6, size=len(didx))
    hdd = np.clip(18.0 - temp, 0.0, None)
    cdd = np.clip(temp - 22.0, 0.0, None)
    # Gas storage: fills over summer, draws down in winter -> seasonal saw-tooth.
    frac = ((ddoy - 90) % 365.25) / 365.25  # 0 at ~April 1 (start of injection)
    storage = 30.0 + 60.0 * np.sin(math.pi * np.clip(frac, 0.0, 1.0))
    storage = np.clip(storage + rng.normal(0.0, 1.5, size=len(didx)), 0.0, 100.0)

    daily_series = {
        "gas_storage_pct": (storage, "%"),
        "hdd": (hdd, "degC-day"),
        "cdd": (cdd, "degC-day"),
        "temp_pop_it": (temp, "degC"),
    }
    for series, (arr, unit) in daily_series.items():
        for ts, val in zip(didx, arr):
            rows.append(
                {
                    "series": series,
                    "zone": None,
                    "valid_start": ts.to_pydatetime(),
                    "resolution_minutes": int(Resolution.DAILY),
                    "value": float(val),
                    "source": SOURCE,
                    "unit": unit,
                }
            )
    return rows


def seed_demo(
    session,
    days: int = 540,
    end: dt.date | None = None,
    seed: int = 42,
) -> int:
    """Generate and upsert a deterministic synthetic dataset; return total rows written.

    Args:
        session: an open SQLAlchemy Session (repositories are constructed on it).
        days: length of the synthetic history in days, ending at `end`.
        end: last delivery day (exclusive upper bound is `end`); defaults to today (UTC).
        seed: integer seed for the numpy Generator (fixed -> reproducible output).

    Only observations are written (electricity zonal + PUN, gas, TTF, exogenous);
    forecasts are produced separately by the real runner. All rows carry
    ``source="demo"``.
    """
    rng = np.random.default_rng(seed)

    end_day = end or dt.datetime.now(dt.UTC).date()
    start_day = end_day - dt.timedelta(days=days)
    start = dt.datetime(start_day.year, start_day.month, start_day.day, tzinfo=dt.UTC)
    end_dt = dt.datetime(end_day.year, end_day.month, end_day.day, tzinfo=dt.UTC)

    hidx = _hourly_index(start, end_dt)
    didx = _daily_index(start_day, end_day)
    if len(hidx) == 0 or len(didx) == 0:
        logger.warning("seed_demo: empty window (days=%s, end=%s) -> nothing to write", days, end_day)
        return 0

    tz = "Europe/Rome"
    logger.info(
        "seed_demo: generating %s days (%s -> %s), %s hours, seed=%s",
        days, start_day, end_day, len(hidx), seed,
    )

    price_repo = PriceRepository(session)
    exo_repo = ExogenousRepository(session)
    total = 0
    res_hour = int(Resolution.HOUR)
    res_daily = int(Resolution.DAILY)

    # --- Electricity: zonal + PUN (hourly) ---
    zonal = _build_electricity(hidx, rng, tz)
    pun = _build_pun(zonal)

    price_buf: list[dict] = []
    for zone in MARKET_ZONES:
        arr = zonal[zone]
        zname = zone.value
        for ts, val in zip(hidx, arr):
            price_buf.append(
                {
                    "market": Market.ELEC_DAYAHEAD.value,
                    "zone": zname,
                    "delivery_start": ts.to_pydatetime(),
                    "resolution_minutes": res_hour,
                    "price": float(val),
                    "source": SOURCE,
                }
            )
        if len(price_buf) >= _BATCH:
            total += _flush(price_repo, price_buf)

    for ts, val in zip(hidx, pun):
        price_buf.append(
            {
                "market": Market.ELEC_DAYAHEAD.value,
                "zone": Zone.PUN.value,
                "delivery_start": ts.to_pydatetime(),
                "resolution_minutes": res_hour,
                "price": float(val),
                "source": SOURCE,
            }
        )
        if len(price_buf) >= _BATCH:
            total += _flush(price_repo, price_buf)
    total += _flush(price_repo, price_buf)

    # --- Gas day-ahead (PSV) + TTF (daily, national) ---
    psv, ttf = _build_gas_daily(didx, rng)
    gas_buf: list[dict] = []
    for ts, gval, tval in zip(didx, psv, ttf):
        pyts = ts.to_pydatetime()
        gas_buf.append(
            {
                "market": Market.GAS_DAYAHEAD.value,
                "zone": None,
                "delivery_start": pyts,
                "resolution_minutes": res_daily,
                "price": float(gval),
                "source": SOURCE,
            }
        )
        gas_buf.append(
            {
                "market": Market.TTF.value,
                "zone": None,
                "delivery_start": pyts,
                "resolution_minutes": res_daily,
                "price": float(tval),
                "source": SOURCE,
            }
        )
        if len(gas_buf) >= _BATCH:
            total += _flush(price_repo, gas_buf)
    total += _flush(price_repo, gas_buf)

    # --- Exogenous drivers ---
    exo_rows = _build_exogenous(hidx, didx, rng, tz)
    total += _flush(exo_repo, exo_rows)

    logger.info("seed_demo: wrote %s rows (source=%s)", total, SOURCE)
    return total
