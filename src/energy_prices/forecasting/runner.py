"""Batch forecast runner: load history, fit model(s), and persist quantile forecasts.

Orchestration glue between the storage repositories, the feature builder and the
family of :class:`~energy_prices.models.base.Forecaster` implementations. For a
given ``(market, zone)`` it loads the trailing price history (inferring the 60 or
15-minute electricity grid, or daily for gas/TTF), selects a model (electricity ->
ensemble, gas -> gas ensemble, TTF -> SARIMAX) with a robust fall back to the
seasonal-naive baseline, builds the future ``horizon_index`` (next delivery day for
day-ahead electricity, a multi-day horizon for gas/TTF), pulls exogenous driver
history plus any *future* forecasts already ingested for the horizon (ENTSO-E load /
wind+solar, gas storage, weather degree-days), then fits, predicts and saves the
flattened :class:`ForecastResult` via :class:`ForecastRepository`.

All datetimes are UTC and timezone-aware; prices are EUR/MWh. Heavy model classes
are imported lazily to avoid import cycles and to let optional dependencies fail
gracefully (raising ``ModelUnavailable``).
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import TYPE_CHECKING

import pandas as pd

from energy_prices.config.enums import MARKET_ZONES, Market, Zone
from energy_prices.storage.db import session_scope
from energy_prices.storage.repositories import (
    ExogenousRepository,
    ForecastRepository,
    PriceRepository,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids importing at runtime
    from sqlalchemy.orm import Session

    from energy_prices.models.base import Forecaster

logger = logging.getLogger(__name__)

# --- Horizon defaults -------------------------------------------------------
# Day-ahead electricity: a single delivery day (24h). Gas/TTF: two weeks of
# daily steps. These apply when the caller does not pass an explicit horizon.
_DEFAULT_ELEC_HORIZON_HOURS = 24
_DEFAULT_GAS_HORIZON_HOURS = 14 * 24

# How much trailing history to pull for fitting (keeps queries bounded). Roughly
# three years of hourly data is plenty for the seasonal/ML models we use.
_ELEC_HISTORY_DAYS = 3 * 365
_GAS_HISTORY_DAYS = 5 * 365

# Local timezone for "next delivery day" boundaries (Italian market clock).
_LOCAL_TZ = "Europe/Rome"

# Exogenous driver series, keyed by market family. Electricity drivers are
# zone-specific; gas/TTF drivers are national (zone=None).
_ELEC_EXOG_SERIES = ("load_forecast", "wind_solar_forecast")
_GAS_EXOG_SERIES = ("gas_storage_pct", "hdd", "cdd")

_DAILY_MINUTES = 1440
_HOUR_MINUTES = 60
_QUARTER_MINUTES = 15


# ---------------------------------------------------------------------------
# Resolution & horizon helpers
# ---------------------------------------------------------------------------
def _resolve_resolution_minutes(
    market: str, recorded: int | None, index: pd.DatetimeIndex
) -> int:
    """Determine the market time unit (minutes) for building the horizon.

    Prefers the ``resolution_minutes`` recorded on the latest observation. For
    electricity it is snapped to the two supported grids (15 or 60); for gas/TTF
    a recorded value is trusted as-is. Falls back to the median index spacing and
    finally to sensible market defaults (daily for gas/TTF, hourly for elec).
    """
    if _is_electricity(market):
        if recorded is not None:
            return _QUARTER_MINUTES if recorded <= 30 else _HOUR_MINUTES
        spacing = _index_spacing_minutes(index)
        if spacing is not None:
            return _QUARTER_MINUTES if spacing <= 30 else _HOUR_MINUTES
        return _HOUR_MINUTES

    if recorded is not None and recorded > 0:
        return recorded
    spacing = _index_spacing_minutes(index)
    if spacing is not None and spacing > 0:
        return spacing
    return _DAILY_MINUTES


def _index_spacing_minutes(index: pd.DatetimeIndex) -> int | None:
    """Median spacing of a DatetimeIndex in whole minutes (``None`` if unknown)."""
    if len(index) < 2:
        return None
    delta = index.to_series().diff().median()
    if pd.isna(delta) or delta <= pd.Timedelta(0):
        return None
    return int(round(delta / pd.Timedelta(minutes=1)))


def _is_electricity(market: str) -> bool:
    return market == Market.ELEC_DAYAHEAD.value


def _build_horizon_index(
    market: str,
    resolution_minutes: int,
    last_delivery: dt.datetime | None,
    horizon_hours: int | None,
    run_at: dt.datetime,
) -> pd.DatetimeIndex:
    """Construct the future ``target_start`` index (UTC, tz-aware).

    For day-ahead electricity the horizon starts at the *next* local delivery
    day's 00:00 (Europe/Rome) and spans ``horizon_hours`` at the data
    resolution. For gas/TTF it is a run of daily steps beginning the day after
    the last observed delivery (or the run day if history is empty).
    """
    freq = pd.Timedelta(minutes=resolution_minutes)

    if _is_electricity(market):
        hours = horizon_hours if horizon_hours is not None else _DEFAULT_ELEC_HORIZON_HOURS
        anchor_local = pd.Timestamp(run_at).tz_convert(_LOCAL_TZ)
        # Next delivery day 00:00 local; if we already have data past run day,
        # start the day after the last delivered day so we never re-forecast past.
        start_day = (anchor_local + pd.Timedelta(days=1)).normalize()
        if last_delivery is not None:
            last_local_day = pd.Timestamp(last_delivery).tz_convert(_LOCAL_TZ).normalize()
            candidate = last_local_day + pd.Timedelta(days=1)
            if candidate > start_day:
                start_day = candidate
        start_utc = start_day.tz_convert("UTC")
        periods = max(1, int(round(hours * 60 / resolution_minutes)))
        return pd.date_range(start=start_utc, periods=periods, freq=freq, tz="UTC")

    # Gas / TTF: daily steps from the day after the last observed delivery.
    hours = horizon_hours if horizon_hours is not None else _DEFAULT_GAS_HORIZON_HOURS
    if last_delivery is not None:
        last_day = pd.Timestamp(last_delivery).tz_convert("UTC").normalize()
        start_utc = last_day + pd.Timedelta(days=1)
    else:
        start_utc = pd.Timestamp(run_at).tz_convert("UTC").normalize() + pd.Timedelta(days=1)
    periods = max(1, int(round(hours * 60 / resolution_minutes)))
    return pd.date_range(start=start_utc, periods=periods, freq=freq, tz="UTC")


# ---------------------------------------------------------------------------
# History & exogenous loading
# ---------------------------------------------------------------------------
def _load_price_series(
    prices: PriceRepository, market: str, zone: str | None, run_at: dt.datetime
) -> tuple[pd.Series, int | None]:
    """Return the trailing price history and the latest recorded resolution.

    The series is a clean UTC-indexed float ``Series``; the second element is the
    ``resolution_minutes`` stored on the most recent observation (or ``None`` when
    the history is empty / lacks the column).
    """
    history_days = _ELEC_HISTORY_DAYS if _is_electricity(market) else _GAS_HISTORY_DAYS
    start = run_at - dt.timedelta(days=history_days)
    df = prices.get_prices(market, zone=zone, start=start, end=run_at)
    if df.empty:
        return pd.Series(dtype=float, name="price"), None

    recorded_res: int | None = None
    if "resolution_minutes" in df.columns:
        latest = df["resolution_minutes"].dropna()
        if not latest.empty:
            try:
                recorded_res = int(latest.iloc[-1])
            except (TypeError, ValueError):  # pragma: no cover - defensive
                recorded_res = None

    series = df["price"].astype(float)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series.name = "price"
    return series.dropna(), recorded_res


def _exog_series_names(market: str) -> tuple[str, ...]:
    return _ELEC_EXOG_SERIES if _is_electricity(market) else _GAS_EXOG_SERIES


def _load_exog_frame(
    exog_repo: ExogenousRepository,
    market: str,
    zone: str | None,
    start: dt.datetime | None,
    end: dt.datetime | None,
) -> pd.DataFrame:
    """Assemble the relevant exogenous driver series into one wide DataFrame.

    Electricity drivers are looked up per-zone (with a national fallback); gas
    drivers are national (``zone=None``). Empty/missing series are simply
    omitted, so the frame degrades to whatever data has been ingested.
    """
    columns: dict[str, pd.Series] = {}
    elec = _is_electricity(market)
    for name in _exog_series_names(market):
        lookup_zone = zone if elec else None
        s = exog_repo.get_series(name, zone=lookup_zone, start=start, end=end)
        if (s is None or s.empty) and elec:
            # National fallback for zonal drivers (e.g. PUN pseudo-zone).
            s = exog_repo.get_series(name, zone=None, start=start, end=end)
        if s is not None and not s.empty:
            columns[name] = s.astype(float)
    if not columns:
        return pd.DataFrame()
    frame = pd.concat(columns, axis=1)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def _split_exog(
    exog_repo: ExogenousRepository,
    market: str,
    zone: str | None,
    history_index: pd.DatetimeIndex | None,
    horizon_index: pd.DatetimeIndex,
    run_at: dt.datetime,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Return ``(exog_history, exog_future)`` aligned to history & horizon.

    ``exog_future`` is built only from driver values whose ``valid_start`` falls
    on the forecast horizon — these are genuine forecasts (load / RES /
    weather) ingested ahead of delivery, so using them does not leak actuals.
    """
    hist_start = None
    if history_index is not None and len(history_index) > 0:
        hist_start = history_index.min().to_pydatetime()
    horizon_end = horizon_index.max().to_pydatetime() if len(horizon_index) > 0 else run_at

    full = _load_exog_frame(exog_repo, market, zone, hist_start, horizon_end)
    if full.empty:
        return None, None

    exog_history: pd.DataFrame | None = None
    if history_index is not None and len(history_index) > 0:
        hist = full.reindex(history_index).ffill()
        if not hist.dropna(how="all").empty:
            exog_history = hist

    future = full.reindex(horizon_index).ffill()
    exog_future: pd.DataFrame | None = None
    if not future.dropna(how="all").empty:
        exog_future = future

    return exog_history, exog_future


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
def _select_model(market: str) -> Forecaster:
    """Lazily build the primary forecaster for a market.

    Electricity -> :class:`EnsembleForecaster` (LEAR + LightGBM); gas ->
    ``EnsembleForecaster.for_gas()`` (SARIMAX + LightGBM); TTF ->
    :class:`SarimaxForecaster`. Raises ``ModelUnavailable`` / ``Exception`` to
    the caller, which falls back to the seasonal-naive baseline.
    """
    if _is_electricity(market):
        from energy_prices.models.ensemble import EnsembleForecaster

        return EnsembleForecaster()
    if market == Market.GAS_DAYAHEAD.value:
        from energy_prices.models.ensemble import EnsembleForecaster

        return EnsembleForecaster.for_gas()
    # TTF and any other daily benchmark.
    from energy_prices.models.gas_sarimax import SarimaxForecaster

    return SarimaxForecaster()


def _fallback_model() -> Forecaster:
    """The always-available seasonal-naive baseline."""
    from energy_prices.models.baseline import SeasonalNaiveForecaster

    return SeasonalNaiveForecaster()


def _fit_predict_with_fallback(
    market: str,
    y: pd.Series,
    horizon_index: pd.DatetimeIndex,
    exog: pd.DataFrame | None,
    exog_future: pd.DataFrame | None,
    calibrate: bool = False,
):
    """Fit the primary model and predict; fall back to seasonal-naive on failure.

    Returns a :class:`ForecastResult`. The seasonal-naive baseline ignores exog
    and depends only on core deps, so it is a dependable last resort. When
    ``calibrate`` is set, the primary model is wrapped in a CQR
    :class:`CalibratedForecaster` for honest interval coverage.
    """
    try:
        model = _select_model(market)
        if calibrate:
            from energy_prices.models.calibration import CalibratedForecaster

            model = CalibratedForecaster(model)
        result = model.fit_predict(
            y, horizon_index, exog=exog, exog_future=exog_future
        )
        if result.quantiles is None or result.quantiles.dropna(how="all").empty:
            raise ValueError("primary model produced an empty forecast")
        return result
    except Exception as exc:  # noqa: BLE001 - intentional broad fallback
        logger.warning(
            "Primary model for market=%s failed (%s); falling back to seasonal-naive.",
            market,
            exc,
        )
        baseline = _fallback_model()
        return baseline.fit_predict(y, horizon_index)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_forecasts(
    market: str,
    zone: str | None = None,
    horizon_hours: int | None = None,
    run_at: dt.datetime | None = None,
    session: Session | None = None,
    calibrate: bool = False,
) -> int:
    """Generate and persist a probabilistic forecast for one ``(market, zone)``.

    Parameters
    ----------
    market:
        A :class:`~energy_prices.config.enums.Market` value, e.g.
        ``"elec_dayahead"``, ``"gas_dayahead"`` or ``"ttf"``.
    zone:
        Bidding zone (e.g. ``"NORD"``) or the ``"PUN"`` pseudo-zone for
        electricity; ``None`` for gas/TTF.
    horizon_hours:
        Forecast horizon in hours. Defaults to 24h for electricity, 14 days for
        gas/TTF when ``None``.
    run_at:
        Run timestamp (UTC). Defaults to ``datetime.now(timezone.utc)``.
    session:
        Optional SQLAlchemy session. When ``None`` a transactional
        :func:`session_scope` is opened and committed automatically.

    Returns
    -------
    int
        The number of forecast rows persisted (one per (target_start, quantile)).
    """
    if run_at is None:
        run_at = dt.datetime.now(dt.UTC)
    elif run_at.tzinfo is None:
        run_at = run_at.replace(tzinfo=dt.UTC)
    else:
        run_at = run_at.astimezone(dt.UTC)

    if session is not None:
        return _run_forecasts_with_session(
            session, market, zone, horizon_hours, run_at, calibrate
        )

    with session_scope() as scoped:
        return _run_forecasts_with_session(
            scoped, market, zone, horizon_hours, run_at, calibrate
        )


def _run_forecasts_with_session(
    session: Session,
    market: str,
    zone: str | None,
    horizon_hours: int | None,
    run_at: dt.datetime,
    calibrate: bool = False,
) -> int:
    """Core run logic against an already-open session (no commit handling)."""
    prices = PriceRepository(session)
    exog_repo = ExogenousRepository(session)
    forecasts = ForecastRepository(session)

    y, recorded_res = _load_price_series(prices, market, zone, run_at)
    if y.empty:
        logger.warning(
            "No price history for market=%s zone=%s; skipping forecast.", market, zone
        )
        return 0

    resolution_minutes = _resolve_resolution_minutes(market, recorded_res, y.index)

    last_delivery = prices.latest_delivery(market, zone=zone)

    horizon_index = _build_horizon_index(
        market, resolution_minutes, last_delivery, horizon_hours, run_at
    )
    if len(horizon_index) == 0:
        logger.warning("Empty horizon for market=%s zone=%s; nothing to forecast.", market, zone)
        return 0

    exog_history, exog_future = _split_exog(
        exog_repo, market, zone, y.index, horizon_index, run_at
    )

    result = _fit_predict_with_fallback(
        market, y, horizon_index, exog_history, exog_future, calibrate
    )

    rows = result.to_rows(
        market=market,
        zone=zone,
        resolution_minutes=resolution_minutes,
        run_at=run_at,
    )
    saved = forecasts.save(rows)
    logger.info(
        "Saved %d forecast rows: market=%s zone=%s model=%s horizon=%d steps res=%dmin run_at=%s",
        saved,
        market,
        zone,
        result.model_name,
        len(horizon_index),
        resolution_minutes,
        run_at.isoformat(),
    )
    return saved


def run_all_electricity_zones(session: Session | None = None, calibrate: bool = False) -> int:
    """Run electricity forecasts for every physical zone plus the PUN index.

    Returns the total number of forecast rows saved across all zones.
    """
    run_at = dt.datetime.now(dt.UTC)
    zones = [z.value for z in MARKET_ZONES] + [Zone.PUN.value]

    if session is not None:
        return _run_zones(session, zones, run_at, calibrate)

    with session_scope() as scoped:
        return _run_zones(scoped, zones, run_at, calibrate)


def _run_zones(
    session: Session, zones: list[str], run_at: dt.datetime, calibrate: bool = False
) -> int:
    total = 0
    for zone in zones:
        try:
            total += _run_forecasts_with_session(
                session, Market.ELEC_DAYAHEAD.value, zone, None, run_at, calibrate
            )
        except Exception as exc:  # noqa: BLE001 - isolate per-zone failures
            logger.exception("Forecast run failed for elec zone=%s: %s", zone, exc)
    return total


def run_all(session: Session | None = None, calibrate: bool = False) -> int:
    """Run the full batch: all electricity zones + PUN, gas day-ahead and TTF.

    Returns the total number of forecast rows saved.
    """
    run_at = dt.datetime.now(dt.UTC)

    if session is not None:
        return _run_all_with_session(session, run_at, calibrate)

    with session_scope() as scoped:
        return _run_all_with_session(scoped, run_at, calibrate)


def _run_all_with_session(
    session: Session, run_at: dt.datetime, calibrate: bool = False
) -> int:
    total = 0

    # Electricity: physical zones + PUN.
    zones = [z.value for z in MARKET_ZONES] + [Zone.PUN.value]
    total += _run_zones(session, zones, run_at, calibrate)

    # Gas day-ahead (national, zone=None).
    for market in (Market.GAS_DAYAHEAD.value, Market.TTF.value):
        try:
            total += _run_forecasts_with_session(
                session, market, None, None, run_at, calibrate
            )
        except Exception as exc:  # noqa: BLE001 - isolate per-market failures
            logger.exception("Forecast run failed for market=%s: %s", market, exc)

    logger.info("run_all complete: %d total forecast rows saved.", total)
    return total
