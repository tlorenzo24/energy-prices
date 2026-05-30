"""Streamlit dashboard — entry point for the GME energy-prices MVP.

Run with ``streamlit run src/energy_prices/dashboard/app.py`` (or ``energy
dashboard`` via the project CLI). Reads go exclusively through the repositories
(observed prices via :class:`PriceRepository`, forecasts via
:class:`ForecastRepository`); every cached read opens and closes its own
session so Streamlit's worker threads never share a SQLAlchemy ``Session``.

Layout: title + GME attribution footer, a synthetic-data badge when
``settings.demo_mode`` is on, and a data-freshness banner; a sidebar with
market / zone / date-range / model selectors; and a main area with a recent
observed-price line chart, a probabilistic forecast chart (median q0.5 plus
shaded q0.1-q0.9 / q0.25-q0.75 bands and overlapping actuals) and KPI metrics.
Gas additionally overlays the TTF benchmark. All datetimes are UTC and
tz-aware; prices are EUR/MWh.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import select

from energy_prices.config import MARKET_ZONES, Market, Zone, get_settings
from energy_prices.storage.db import session_scope
from energy_prices.storage.models import Forecast
from energy_prices.storage.repositories import (
    ForecastRepository,
    PriceRepository,
)

logger = logging.getLogger(__name__)

# --- Constants -------------------------------------------------------------

_CACHE_TTL = 300  # seconds: DB reads are cached for 5 minutes.
_ATTRIBUTION = "Fonte: Gestore dei Mercati Energetici S.p.A. (GME)."
_TTF_SOURCE = "Benchmark TTF (proxy front-month)."

# Display label -> Market enum.
_MARKET_LABELS: dict[str, Market] = {
    "Elettricità (MGP)": Market.ELEC_DAYAHEAD,
    "Gas (PSV day-ahead)": Market.GAS_DAYAHEAD,
}

# Quantile column names produced by ForecastRepository.get_forecasts.
_Q_LOW, _Q_LOWMID, _Q_MID, _Q_HIMID, _Q_HIGH = (
    "q0.1",
    "q0.25",
    "q0.5",
    "q0.75",
    "q0.9",
)

# Plotly colours.
_C_OBSERVED = "#1f77b4"
_C_FORECAST = "#d62728"
_C_BAND_OUTER = "rgba(214,39,40,0.12)"
_C_BAND_INNER = "rgba(214,39,40,0.25)"
_C_TTF = "#2ca02c"


# --- Cached DB reads -------------------------------------------------------
# Each function opens its own session (safe from Streamlit rerun threads) and
# returns plain pandas / primitives that st.cache_data can cache cleanly.


@st.cache_data(ttl=_CACHE_TTL, show_spinner=False)
def load_prices(
    market: str,
    zone: str | None,
    start: dt.datetime | None,
    end: dt.datetime | None,
) -> pd.DataFrame:
    """Observed prices for one market/zone window, UTC-indexed."""
    with session_scope() as session:
        return PriceRepository(session).get_prices(
            market=market, zone=zone, start=start, end=end
        )


@st.cache_data(ttl=_CACHE_TTL, show_spinner=False)
def load_latest_delivery(market: str, zone: str | None) -> dt.datetime | None:
    """Timestamp of the most recent observed delivery for the freshness banner."""
    with session_scope() as session:
        return PriceRepository(session).latest_delivery(market=market, zone=zone)


@st.cache_data(ttl=_CACHE_TTL, show_spinner=False)
def load_forecast(
    market: str, zone: str | None, model_name: str | None
) -> pd.DataFrame:
    """Latest-run wide forecast (cols q0.1..q0.9) for the selection."""
    with session_scope() as session:
        return ForecastRepository(session).get_forecasts(
            market=market, zone=zone, model_name=model_name, latest=True
        )


@st.cache_data(ttl=_CACHE_TTL, show_spinner=False)
def load_forecast_run_at(
    market: str, zone: str | None, model_name: str | None
) -> dt.datetime | None:
    """Run timestamp of the latest forecast for the selection (for captions)."""
    with session_scope() as session:
        return ForecastRepository(session).latest_run_at(
            market=market, zone=zone, model_name=model_name
        )


@st.cache_data(ttl=_CACHE_TTL, show_spinner=False)
def list_model_names(market: str, zone: str | None) -> list[str]:
    """Distinct model names with stored forecasts for the selection.

    Read-only discovery query (the repositories expose no such helper); kept
    inside the dashboard's own session scope.
    """
    with session_scope() as session:
        stmt = select(Forecast.model_name).where(Forecast.market == market).distinct()
        if zone is not None:
            stmt = stmt.where(Forecast.zone == zone)
        names = [n for (n,) in session.execute(stmt).all() if n]
    return sorted(names)


# --- Small helpers ---------------------------------------------------------


def _fmt_eur(value: float | None) -> str:
    """Format a EUR/MWh value, tolerating None/NaN."""
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.2f} €/MWh"


def _to_utc(d: dt.date, *, end_of_day: bool = False) -> dt.datetime:
    """Convert a date picked in the sidebar to a tz-aware UTC datetime."""
    t = dt.time(23, 59, 59) if end_of_day else dt.time(0, 0, 0)
    return dt.datetime.combine(d, t, tzinfo=dt.UTC)


def _freshness_banner(latest: dt.datetime | None, tz: str) -> None:
    """Render the 'last updated' banner from the latest observed delivery."""
    if latest is None:
        st.info(
            "Nessun dato disponibile. Esegui `energy seed-demo` per dati "
            "sintetici di esempio oppure `energy ingest` per dati reali."
        )
        return
    # latest is a tz-aware UTC datetime; show it in the configured local tz.
    shown = pd.Timestamp(latest).tz_convert(tz)
    now = dt.datetime.now(dt.UTC)
    age_h = (now - latest).total_seconds() / 3600.0
    label = shown.strftime("%Y-%m-%d %H:%M %Z")
    if age_h <= 36:
        st.success(f"Ultimo dato di consegna: {label} (≈{age_h:.0f} h fa).")
    else:
        st.warning(
            f"Ultimo dato di consegna: {label} (≈{age_h / 24:.1f} giorni fa). "
            "I dati potrebbero non essere aggiornati — esegui `energy ingest`."
        )


def _empty_message(what: str) -> None:
    st.warning(
        f"Nessun {what} per la selezione corrente. "
        "Esegui `energy seed-demo` (dati demo) o `energy ingest` (dati reali), "
        "e per le previsioni `energy forecast`."
    )


def _resample_for_plot(s: pd.Series, max_points: int = 2000) -> pd.Series:
    """Downsample a long series to keep the chart responsive (mean per bin)."""
    if len(s) <= max_points:
        return s
    factor = max(1, len(s) // max_points)
    return s.iloc[::factor]


# --- Chart builders --------------------------------------------------------


def _observed_chart(prices: pd.DataFrame, title: str, tz: str) -> go.Figure:
    """Line chart of recent observed prices (local-time x-axis)."""
    fig = go.Figure()
    if not prices.empty:
        series = _resample_for_plot(prices["price"].sort_index())
        x = series.index.tz_convert(tz)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=series.to_numpy(),
                mode="lines",
                name="Osservato",
                line=dict(color=_C_OBSERVED, width=1.6),
                hovertemplate="%{x|%d %b %H:%M}<br>%{y:.2f} €/MWh<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=f"Tempo ({tz})",
        yaxis_title="€/MWh",
        margin=dict(l=10, r=10, t=40, b=10),
        height=360,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def _add_band(
    fig: go.Figure, x: Any, fc: pd.DataFrame, lo: str, hi: str, color: str, label: str
) -> None:
    """Add a shaded quantile band (filled area between the lo/hi columns)."""
    if lo not in fc.columns or hi not in fc.columns:
        return
    fig.add_trace(
        go.Scatter(
            x=x, y=fc[hi].to_numpy(), mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip", name=hi,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=fc[lo].to_numpy(), mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=color, name=label, hoverinfo="skip",
        )
    )


def _forecast_chart(
    forecast: pd.DataFrame,
    actuals: pd.Series | None,
    title: str,
    tz: str,
) -> go.Figure:
    """Forecast chart: q0.1-q0.9 + q0.25-q0.75 bands, median, and actuals."""
    fig = go.Figure()
    fc = forecast.sort_index()
    x = fc.index.tz_convert(tz)

    _add_band(fig, x, fc, _Q_LOW, _Q_HIGH, _C_BAND_OUTER, "Banda 10–90%")
    _add_band(fig, x, fc, _Q_LOWMID, _Q_HIMID, _C_BAND_INNER, "Banda 25–75%")

    # Median (point forecast).
    if _Q_MID in fc.columns:
        fig.add_trace(
            go.Scatter(
                x=x, y=fc[_Q_MID].to_numpy(), mode="lines",
                name="Previsione (mediana)",
                line=dict(color=_C_FORECAST, width=2.2),
                hovertemplate="%{x|%d %b %H:%M}<br>%{y:.2f} €/MWh<extra></extra>",
            )
        )

    # Overlapping actuals (forecast-vs-actual).
    if actuals is not None and not actuals.empty:
        ax = actuals.sort_index()
        fig.add_trace(
            go.Scatter(
                x=ax.index.tz_convert(tz), y=ax.to_numpy(), mode="lines",
                name="Osservato (reale)",
                line=dict(color=_C_OBSERVED, width=1.6, dash="dot"),
                hovertemplate="%{x|%d %b %H:%M}<br>%{y:.2f} €/MWh<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=f"Tempo ({tz})",
        yaxis_title="€/MWh",
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


# --- KPI + sections --------------------------------------------------------


def _render_kpis(prices: pd.DataFrame, forecast: pd.DataFrame) -> None:
    """Latest price, next-period forecast, and day min/max metrics."""
    latest_price = next_fc = day_min = day_max = None

    if not prices.empty:
        p = prices["price"].sort_index()
        latest_price = float(p.iloc[-1])
        last_day = p[p.index >= (p.index[-1] - pd.Timedelta(hours=24))]
        if not last_day.empty:
            day_min, day_max = float(last_day.min()), float(last_day.max())

    if not forecast.empty and _Q_MID in forecast.columns:
        fmid = forecast[_Q_MID].dropna()
        if not fmid.empty:
            next_fc = float(fmid.iloc[0])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ultimo prezzo", _fmt_eur(latest_price))
    c2.metric("Prossima previsione", _fmt_eur(next_fc))
    c3.metric("Min (24h)", _fmt_eur(day_min))
    c4.metric("Max (24h)", _fmt_eur(day_max))


def _actuals_over_forecast(
    market: str, zone: str | None, forecast: pd.DataFrame
) -> pd.Series | None:
    """Fetch observed prices overlapping the forecast horizon, if any exist."""
    if forecast.empty:
        return None
    start = forecast.index.min().to_pydatetime()
    end = forecast.index.max().to_pydatetime()
    overlap = load_prices(market, zone, start, end)
    if overlap.empty:
        return None
    return overlap["price"].sort_index()


def _render_forecast_section(
    market: str, zone: str | None, model_name: str | None, tz: str, title: str
) -> None:
    """Forecast chart + run caption, defensive against missing forecasts."""
    forecast = load_forecast(market, zone, model_name)
    if forecast.empty:
        _empty_message("previsione disponibile")
        return
    run_at = load_forecast_run_at(market, zone, model_name)
    if run_at is not None:
        st.caption(
            "Ultima esecuzione del modello: "
            f"{pd.Timestamp(run_at).tz_convert(tz).strftime('%Y-%m-%d %H:%M %Z')}"
        )
    actuals = _actuals_over_forecast(market, zone, forecast)
    st.plotly_chart(
        _forecast_chart(forecast, actuals, title, tz),
        width="stretch",
    )


def _render_ttf_overlay(tz: str, default_window: tuple[dt.datetime, dt.datetime]) -> None:
    """Overlay the TTF benchmark series for the gas view."""
    start, end = default_window
    ttf = load_prices(Market.TTF.value, None, start, end)
    if ttf.empty:
        st.info("Serie TTF non disponibile per il periodo selezionato.")
        return
    s = ttf["price"].sort_index()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=s.index.tz_convert(tz), y=s.to_numpy(), mode="lines",
            name="TTF", line=dict(color=_C_TTF, width=1.8),
            hovertemplate="%{x|%d %b}<br>%{y:.2f} €/MWh<extra></extra>",
        )
    )
    fig.update_layout(
        title="Benchmark TTF (giornaliero)",
        xaxis_title=f"Tempo ({tz})",
        yaxis_title="€/MWh",
        margin=dict(l=10, r=10, t=40, b=10),
        height=320,
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(_TTF_SOURCE)


# --- Sidebar ---------------------------------------------------------------


def _zone_options() -> list[str]:
    """7 physical zones + PUN as selectable electricity-zone labels."""
    return [z.value for z in MARKET_ZONES] + [Zone.PUN.value]


def _sidebar(tz: str) -> dict[str, Any]:
    """Render sidebar controls; return the current selection state."""
    st.sidebar.header("Filtri")
    market_label = st.sidebar.radio(
        "Mercato", list(_MARKET_LABELS.keys()), index=0
    )
    market = _MARKET_LABELS[market_label]

    zone: str | None
    if market is Market.ELEC_DAYAHEAD:
        zone = st.sidebar.selectbox(
            "Zona", _zone_options(), index=len(_zone_options()) - 1,
            help="7 zone fisiche + indice PUN.",
        )
    else:
        zone = None  # gas/PSV is national (zone is NULL in storage).

    today = dt.datetime.now(dt.UTC).date()
    default_start = today - dt.timedelta(days=30)
    date_range = st.sidebar.date_input(
        "Intervallo date",
        value=(default_start, today),
        max_value=today + dt.timedelta(days=7),
    )
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_d, end_d = date_range
    else:  # single date selected so far
        start_d = end_d = (
            date_range[0] if isinstance(date_range, (tuple, list)) else date_range
        )

    # Model selector (only meaningful where forecasts exist).
    model_names = list_model_names(market.value, zone)
    model_choice = "auto"
    if model_names:
        options = ["auto (ultimo run)"] + model_names
        chosen = st.sidebar.selectbox("Modello previsione", options, index=0)
        model_choice = None if chosen.startswith("auto") else chosen
    else:
        st.sidebar.caption("Nessuna previsione disponibile per questa selezione.")
        model_choice = None

    return {
        "market": market,
        "zone": zone,
        "start": _to_utc(start_d),
        "end": _to_utc(end_d, end_of_day=True),
        "model_name": model_choice,
        "tz": tz,
    }


# --- Main ------------------------------------------------------------------


def main() -> None:
    settings = get_settings()
    tz = settings.timezone

    st.set_page_config(
        page_title="GME — Prezzi Energia & Previsioni",
        page_icon="⚡",
        layout="wide",
    )
    st.title("GME — Prezzi Energia & Previsioni")

    if settings.demo_mode:
        st.error(
            "⚠️ DATI DEMO SINTETICI — i valori mostrati sono generati "
            "artificialmente e NON rappresentano prezzi di mercato reali."
        )

    sel = _sidebar(tz)
    market: Market = sel["market"]
    zone: str | None = sel["zone"]

    # Freshness banner from the latest observed delivery for the selection.
    latest = load_latest_delivery(market.value, zone)
    _freshness_banner(latest, tz)

    zone_label = f" — Zona {zone}" if zone else ""
    market_name = "Elettricità" if market is Market.ELEC_DAYAHEAD else "Gas (PSV)"

    # --- Observed prices ---
    prices = load_prices(market.value, zone, sel["start"], sel["end"])
    st.subheader(f"Prezzi osservati — {market_name}{zone_label}")
    if prices.empty:
        _empty_message("prezzo osservato")
    else:
        _render_kpis(prices, load_forecast(market.value, zone, sel["model_name"]))
        st.plotly_chart(
            _observed_chart(prices, f"Prezzi {market_name}{zone_label}", tz),
            width="stretch",
        )

    # --- Forecast vs actual ---
    st.subheader("Previsione probabilistica")
    _render_forecast_section(
        market.value,
        zone,
        sel["model_name"],
        tz,
        f"Previsione {market_name}{zone_label}",
    )

    # --- Gas: TTF overlay ---
    if market is Market.GAS_DAYAHEAD:
        st.subheader("Confronto con il benchmark europeo")
        _render_ttf_overlay(tz, (sel["start"], sel["end"]))

    st.divider()
    st.caption(_ATTRIBUTION)


# Streamlit runs the module top-to-bottom on each rerun, so invoke main()
# unconditionally; this also works under `python -m`.
main()
