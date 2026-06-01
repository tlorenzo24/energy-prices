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
import hmac
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

# --- Aesthetic: refined energy trading-desk (dark) -------------------------
# Committed palette; see _inject_css for the matching CSS variables.
_BG = "#0A0E14"          # deep slate app background
_SURFACE = "#121823"     # card surface
_BORDER = "#1E2A38"      # hairline borders
_TEXT = "#E6EDF3"        # primary text
_MUTED = "#7D8B9A"       # secondary / muted text
_GRID = "#19222E"        # chart gridlines
_AMBER = "#FFB300"       # electricity accent (energy/gold)
_TEAL = "#1FD1A3"        # gas accent
_OBSERVED = "#5AA9FF"    # observed-price line (cool blue)

# Distinct per-zone hues (electricity); PUN = bright neutral (it is an index).
_ZONE_COLORS: dict[str, str] = {
    "NORD": "#FFB300", "CNOR": "#FF7A45", "CSUD": "#FF4D6D", "SUD": "#C77DFF",
    "CALA": "#4EA8FF", "SICI": "#1FD1A3", "SARD": "#9EE493", "PUN": "#E6EDF3",
}


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """'#RRGGBB' + alpha -> 'rgba(r,g,b,a)' for translucent fills."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _accent_for(market: str, zone: str | None) -> str:
    """The accent colour for a market/zone selection."""
    if market == Market.ELEC_DAYAHEAD.value and zone:
        return _ZONE_COLORS.get(zone.upper(), _AMBER)
    if market == Market.GAS_DAYAHEAD.value:
        return _TEAL
    return _AMBER


# --- Look & feel (CSS + chart theme) ---------------------------------------
# Hex values mirror the palette constants above (kept literal so the CSS block
# stays a plain, brace-safe string).
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

:root {
  --bg:#0A0E14; --surface:#121823; --surface-2:#0F1620; --border:#1E2A38;
  --text:#E6EDF3; --muted:#7D8B9A; --amber:#FFB300; --teal:#1FD1A3;
}

/* App canvas with a subtle top glow for atmosphere */
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1100px 480px at 18% -8%, rgba(255,179,0,.07), transparent 60%),
    radial-gradient(900px 420px at 92% -6%, rgba(31,209,163,.06), transparent 60%),
    var(--bg);
}
html, body, [class*="css"], [data-testid="stMarkdownContainer"]{
  font-family:'Sora', system-ui, sans-serif; color:var(--text);
}
[data-testid="stMainBlockContainer"]{ padding-top:1.1rem; max-width:1280px; }

/* Strip default Streamlit chrome for a cleaner product feel */
#MainMenu, footer, [data-testid="stToolbar"]{ visibility:hidden; }
header[data-testid="stHeader"]{ background:transparent; }

/* Sidebar */
[data-testid="stSidebar"]{
  background:var(--surface-2); border-right:1px solid var(--border);
}
[data-testid="stSidebar"] .stRadio label, [data-testid="stSidebar"] label{ color:var(--text); }

/* Header bar */
.ep-header{
  display:flex; align-items:center; justify-content:space-between; gap:1rem;
  padding:16px 22px; margin:0 0 14px 0; border-radius:16px;
  background:linear-gradient(135deg, rgba(255,179,0,.10), rgba(31,209,163,.06)), var(--surface);
  border:1px solid var(--border);
}
.ep-brand{ display:flex; align-items:center; gap:13px; }
.ep-logo{
  width:42px; height:42px; border-radius:12px; display:grid; place-items:center;
  font-size:22px; background:linear-gradient(135deg,#FFB300,#FF7A45);
  box-shadow:0 6px 20px rgba(255,179,0,.28);
}
.ep-title{ font-size:1.32rem; font-weight:700; letter-spacing:.2px; line-height:1.1; }
.ep-sub{ color:var(--muted); font-size:.78rem; font-weight:500; letter-spacing:.4px;
  text-transform:uppercase; }
.ep-pill{
  font-family:'JetBrains Mono', monospace; font-size:.74rem; font-weight:700;
  padding:7px 13px; border-radius:999px; border:1px solid var(--border);
  display:inline-flex; align-items:center; gap:7px; white-space:nowrap;
}
.ep-pill::before{ content:''; width:8px; height:8px; border-radius:50%; }
.ep-pill.fresh{ color:#9EE493; background:rgba(31,209,163,.10); }
.ep-pill.fresh::before{ background:#1FD1A3; box-shadow:0 0 9px #1FD1A3; }
.ep-pill.stale{ color:#FFCE66; background:rgba(255,179,0,.10); }
.ep-pill.stale::before{ background:#FFB300; box-shadow:0 0 9px #FFB300; }
.ep-pill.none{ color:var(--muted); background:rgba(125,139,154,.12); }
.ep-pill.none::before{ background:var(--muted); }

/* KPI cards */
.ep-kpis{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:4px 0 8px; }
.ep-kpi{
  background:var(--surface); border:1px solid var(--border); border-radius:14px;
  padding:15px 17px; position:relative; overflow:hidden;
  transition:transform .12s ease, border-color .12s ease;
}
.ep-kpi:hover{ transform:translateY(-2px); border-color:#2C3A4B; }
.ep-kpi::after{ content:''; position:absolute; left:0; top:0; bottom:0; width:3px;
  background:var(--accent); }
.ep-kpi .lbl{ color:var(--muted); font-size:.72rem; font-weight:600; letter-spacing:.6px;
  text-transform:uppercase; }
.ep-kpi .val{ font-family:'JetBrains Mono', monospace; font-size:1.55rem; font-weight:700;
  margin-top:6px; line-height:1; }
.ep-kpi .unit{ color:var(--muted); font-size:.82rem; font-weight:500; }
.ep-kpi .sub{ color:var(--muted); font-family:'JetBrains Mono', monospace; font-size:.74rem;
  margin-top:6px; }

/* Section headers */
.ep-section{ display:flex; align-items:center; gap:10px; margin:22px 0 8px; }
.ep-section .bar{ width:4px; height:20px; border-radius:3px; background:var(--accent); }
.ep-section .txt{ font-size:1.06rem; font-weight:600; letter-spacing:.2px; }

/* Plotly cards */
[data-testid="stPlotlyChart"]{
  background:var(--surface); border:1px solid var(--border); border-radius:14px;
  padding:6px 8px 2px;
}
.ep-foot{ color:var(--muted); font-size:.74rem; margin-top:6px; }
::-webkit-scrollbar{ width:10px; height:10px; }
::-webkit-scrollbar-thumb{ background:#1E2A38; border-radius:8px; }
</style>
"""


def _inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def _style_fig(fig: go.Figure, height: int, accent: str) -> go.Figure:
    """Apply the dark trading-desk theme to a Plotly figure."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=14, t=14, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", size=12, color=_MUTED),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=_SURFACE, bordercolor=accent,
            font=dict(family="JetBrains Mono, monospace", size=12, color=_TEXT),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(color=_MUTED, size=11),
        ),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=_BORDER, color=_MUTED),
        yaxis=dict(
            title="€/MWh", gridcolor=_GRID, zeroline=False, linecolor=_BORDER, color=_MUTED,
        ),
    )
    return fig


def _section(title: str, accent: str) -> None:
    st.markdown(
        f"<div class='ep-section'><span class='bar' style='background:{accent}'></span>"
        f"<span class='txt'>{title}</span></div>",
        unsafe_allow_html=True,
    )


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


def _to_utc(d: dt.date, *, end_of_day: bool = False) -> dt.datetime:
    """Convert a date picked in the sidebar to a tz-aware UTC datetime."""
    t = dt.time(23, 59, 59) if end_of_day else dt.time(0, 0, 0)
    return dt.datetime.combine(d, t, tzinfo=dt.UTC)


def _freshness_pill(latest: dt.datetime | None, tz: str) -> str:
    """Live data-freshness badge (HTML) from the latest observed delivery."""
    if latest is None:
        return "<span class='ep-pill none'>NESSUN DATO</span>"
    shown = pd.Timestamp(latest).tz_convert(tz)
    age_h = (dt.datetime.now(dt.UTC) - latest).total_seconds() / 3600.0
    label = shown.strftime("%d %b %H:%M")
    if age_h <= 36:
        return f"<span class='ep-pill fresh'>LIVE · {label} (≈{age_h:.0f}h fa)</span>"
    return f"<span class='ep-pill stale'>{label} (≈{age_h / 24:.1f}g fa)</span>"


def _header(market_name: str, zone: str | None, latest: dt.datetime | None, tz: str) -> None:
    """Top brand bar: logo, title/subtitle and the live freshness pill."""
    zone_label = f" · {zone}" if zone else ""
    st.markdown(
        "<div class='ep-header'>"
        "<div class='ep-brand'>"
        "<div class='ep-logo'>⚡</div>"
        "<div><div class='ep-title'>GME · Energy Desk</div>"
        f"<div class='ep-sub'>{market_name}{zone_label} — prezzi & previsioni probabilistiche</div>"
        "</div></div>"
        f"{_freshness_pill(latest, tz)}"
        "</div>",
        unsafe_allow_html=True,
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


def _observed_chart(prices: pd.DataFrame, tz: str, accent: str) -> go.Figure:
    """Area+line chart of recent observed prices (local-time x-axis)."""
    fig = go.Figure()
    if not prices.empty:
        series = _resample_for_plot(prices["price"].sort_index())
        fig.add_trace(
            go.Scatter(
                x=series.index.tz_convert(tz),
                y=series.to_numpy(),
                mode="lines",
                name="Osservato",
                line=dict(color=accent, width=1.8, shape="spline", smoothing=0.4),
                fill="tozeroy",
                fillcolor=_hex_to_rgba(accent, 0.06),
                hovertemplate="%{x|%d %b %H:%M}  <b>%{y:.2f}</b> €/MWh<extra></extra>",
            )
        )
    return _style_fig(fig, height=320, accent=accent)


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
    tz: str,
    accent: str,
) -> go.Figure:
    """Forecast chart: q0.1-q0.9 + q0.25-q0.75 bands, median, and actuals."""
    fig = go.Figure()
    fc = forecast.sort_index()
    x = fc.index.tz_convert(tz)

    _add_band(fig, x, fc, _Q_LOW, _Q_HIGH, _hex_to_rgba(accent, 0.12), "Banda 10–90%")
    _add_band(fig, x, fc, _Q_LOWMID, _Q_HIMID, _hex_to_rgba(accent, 0.24), "Banda 25–75%")

    # Median (point forecast).
    if _Q_MID in fc.columns:
        fig.add_trace(
            go.Scatter(
                x=x, y=fc[_Q_MID].to_numpy(), mode="lines",
                name="Previsione (mediana)",
                line=dict(color=accent, width=2.6, shape="spline", smoothing=0.4),
                hovertemplate="%{x|%d %b %H:%M}  <b>%{y:.2f}</b> €/MWh<extra></extra>",
            )
        )

    # Overlapping actuals (forecast-vs-actual).
    if actuals is not None and not actuals.empty:
        ax = actuals.sort_index()
        fig.add_trace(
            go.Scatter(
                x=ax.index.tz_convert(tz), y=ax.to_numpy(), mode="lines",
                name="Osservato (reale)",
                line=dict(color=_OBSERVED, width=1.8, dash="dot"),
                hovertemplate="%{x|%d %b %H:%M}  <b>%{y:.2f}</b> €/MWh<extra></extra>",
            )
        )
    return _style_fig(fig, height=400, accent=accent)


# --- KPI + sections --------------------------------------------------------


def _kpi_card(label: str, value: float | None, accent: str, sub: str = "") -> str:
    """One KPI card as HTML (mono value, accent bar, optional sub-line)."""
    val = f"{value:,.2f}<span class='unit'> €/MWh</span>" if value is not None else "—"
    sub_html = f"<div class='sub'>{sub}</div>" if sub else ""
    return (
        f"<div class='ep-kpi' style='--accent:{accent}'>"
        f"<div class='lbl'>{label}</div><div class='val'>{val}</div>{sub_html}</div>"
    )


def _render_kpis(prices: pd.DataFrame, forecast: pd.DataFrame, accent: str) -> None:
    """Latest price, next-period forecast, and day min/max as hero cards."""
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

    # Forecast-vs-last delta as the "Prossima previsione" sub-line.
    delta = ""
    if next_fc is not None and latest_price is not None and latest_price != 0:
        pct = (next_fc - latest_price) / abs(latest_price) * 100
        arrow = "▲" if pct >= 0 else "▼"
        delta = f"{arrow} {pct:+.1f}% vs ultimo"

    cards = (
        _kpi_card("Ultimo prezzo", latest_price, accent)
        + _kpi_card("Prossima previsione", next_fc, accent, delta)
        + _kpi_card("Min · 24h", day_min, accent)
        + _kpi_card("Max · 24h", day_max, accent)
    )
    st.markdown(f"<div class='ep-kpis'>{cards}</div>", unsafe_allow_html=True)


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
    market: str, zone: str | None, model_name: str | None, tz: str
) -> None:
    """Forecast chart + run caption, defensive against missing forecasts."""
    forecast = load_forecast(market, zone, model_name)
    if forecast.empty:
        _empty_message("previsione disponibile")
        return
    accent = _accent_for(market, zone)
    actuals = _actuals_over_forecast(market, zone, forecast)
    st.plotly_chart(
        _forecast_chart(forecast, actuals, tz, accent),
        width="stretch",
    )
    run_at = load_forecast_run_at(market, zone, model_name)
    if run_at is not None:
        st.markdown(
            "<div class='ep-foot'>Ultima esecuzione del modello: "
            f"{pd.Timestamp(run_at).tz_convert(tz).strftime('%Y-%m-%d %H:%M %Z')}</div>",
            unsafe_allow_html=True,
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
            name="TTF", line=dict(color=_TEAL, width=2.0, shape="spline", smoothing=0.4),
            fill="tozeroy", fillcolor=_hex_to_rgba(_TEAL, 0.06),
            hovertemplate="%{x|%d %b}  <b>%{y:.2f}</b> €/MWh<extra></extra>",
        )
    )
    st.plotly_chart(_style_fig(fig, height=300, accent=_TEAL), width="stretch")
    st.markdown(f"<div class='ep-foot'>{_TTF_SOURCE}</div>", unsafe_allow_html=True)


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
    # st.date_input returns a single date or a (start, end) tuple depending on
    # how much of the range the user has picked; normalise to (start_d, end_d).
    date_range: Any = st.sidebar.date_input(
        "Intervallo date",
        value=(default_start, today),
        max_value=today + dt.timedelta(days=7),
    )
    if isinstance(date_range, (tuple, list)):
        seq = list(date_range)
        start_d = seq[0]
        end_d = seq[1] if len(seq) > 1 else seq[0]
    else:  # single date selected so far
        start_d = end_d = date_range

    # Model selector (only meaningful where forecasts exist).
    model_names = list_model_names(market.value, zone)
    model_choice: str | None = "auto"
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


def _require_password(settings) -> bool:
    """Shared-secret gate for internal sharing. No-op if no password is set.

    Returns True when access is granted. Defence-in-depth only: pair it with a
    VPN / network isolation (see README deploy notes) — it is NOT a substitute
    for real per-user auth (use Streamlit native OIDC for that).
    """
    if not settings.dashboard_password:
        return True
    if st.session_state.get("_authed"):
        return True
    st.title("🔒 energy-prices")
    pwd = st.text_input("Password", type="password")
    if not pwd:
        st.stop()
    if hmac.compare_digest(pwd, settings.dashboard_password):
        st.session_state["_authed"] = True
        st.rerun()
    st.error("Password errata.")
    st.stop()
    return False


def main() -> None:
    settings = get_settings()
    tz = settings.timezone

    st.set_page_config(
        page_title="GME — Prezzi Energia & Previsioni",
        page_icon="⚡",
        layout="wide",
    )
    _inject_css()
    _require_password(settings)

    sel = _sidebar(tz)
    market: Market = sel["market"]
    zone: str | None = sel["zone"]
    accent = _accent_for(market.value, zone)
    market_name = "Elettricità" if market is Market.ELEC_DAYAHEAD else "Gas (PSV)"

    latest = load_latest_delivery(market.value, zone)
    _header(market_name, zone, latest, tz)

    if settings.demo_mode:
        st.error(
            "⚠️ DATI DEMO SINTETICI — i valori mostrati sono generati "
            "artificialmente e NON rappresentano prezzi di mercato reali."
        )
    if latest is None:
        st.info(
            "Nessun dato disponibile. Esegui `energy seed-demo` per dati sintetici "
            "oppure `energy ingest` per i dati reali."
        )

    # --- Observed prices ---
    prices = load_prices(market.value, zone, sel["start"], sel["end"])
    _section(f"Prezzi osservati · {market_name}", accent)
    if prices.empty:
        _empty_message("prezzo osservato")
    else:
        _render_kpis(prices, load_forecast(market.value, zone, sel["model_name"]), accent)
        st.plotly_chart(_observed_chart(prices, tz, accent), width="stretch")

    # --- Forecast vs actual ---
    _section("Previsione probabilistica", accent)
    _render_forecast_section(market.value, zone, sel["model_name"], tz)

    # --- Gas: TTF overlay ---
    if market is Market.GAS_DAYAHEAD:
        _section("Confronto con il benchmark europeo TTF", _TEAL)
        _render_ttf_overlay(tz, (sel["start"], sel["end"]))

    st.divider()
    st.markdown(f"<div class='ep-foot'>{_ATTRIBUTION}</div>", unsafe_allow_html=True)


# Streamlit runs the module top-to-bottom on each rerun, so invoke main()
# unconditionally; this also works under `python -m`.
main()
