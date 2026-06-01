"""Command-line interface for energy-prices.

Commands: init-db, seed-demo, ingest, backfill, gme-inspect, forecast, backtest,
alerts, dashboard, scheduler. Run `energy --help` or `energy <command> --help`.
"""

from __future__ import annotations

import datetime as dt
import logging
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from energy_prices.config import Market, Zone, get_settings

app = typer.Typer(add_completion=False, help="GME electricity & gas prices + forecasting.")
console = Console()


def _setup_logging() -> None:
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _parse_date(value: str | None) -> dt.date | None:
    if value is None:
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(f"invalid date {value!r} (expected YYYY-MM-DD)") from exc


def _validate_elec_zone(zone: str) -> str:
    """Validate an electricity zone against the Zone enum (case-insensitive)."""
    z = zone.upper()
    valid = {m.value for m in Zone}
    if z not in valid:
        raise typer.BadParameter(f"unknown zone {zone!r}; expected one of {sorted(valid)}")
    return z


def _redacted_url(url: str) -> str:
    """Render a DB URL with the password masked (so init-db never logs secrets)."""
    from sqlalchemy.engine import make_url

    return make_url(url).render_as_string(hide_password=True)


# Friendly market aliases -> Market enum value.
_MARKET_ALIASES = {
    "elec": Market.ELEC_DAYAHEAD.value,
    "electricity": Market.ELEC_DAYAHEAD.value,
    "elettricita": Market.ELEC_DAYAHEAD.value,
    "elec_dayahead": Market.ELEC_DAYAHEAD.value,
    "gas": Market.GAS_DAYAHEAD.value,
    "gas_dayahead": Market.GAS_DAYAHEAD.value,
    "ttf": Market.TTF.value,
}


@app.command("init-db")
def init_db_cmd() -> None:
    """Create database tables (and TimescaleDB hypertables on Postgres)."""
    _setup_logging()
    from energy_prices.storage.db import init_db

    init_db()
    console.print(f"[green]Database initialised[/] at {_redacted_url(get_settings().database_url)}")


@app.command("seed-demo")
def seed_demo_cmd(
    days: int = typer.Option(540, help="Length of synthetic history in days."),
    seed: int = typer.Option(42, help="RNG seed (deterministic output)."),
) -> None:
    """Generate realistic SYNTHETIC data so the dashboard works with no credentials."""
    _setup_logging()
    from energy_prices.ingestion.demo import seed_demo
    from energy_prices.storage.db import init_db, session_scope

    init_db()
    with console.status("Generating synthetic dataset…"):
        with session_scope() as session:
            n = seed_demo(session, days=days, seed=seed)
    console.print(f"[green]Seeded {n:,} demo rows[/] (source='demo'). Now run: energy forecast")


@app.command()
def ingest(
    source: str = typer.Option("all", help="all | entsoe | gme | ttf | gie | weather"),
    start: str | None = typer.Option(None, help="Start date YYYY-MM-DD (default: 30d ago)."),
    end: str | None = typer.Option(None, help="End date YYYY-MM-DD (default: +2d)."),
) -> None:
    """Ingest real market data from the configured sources into the database."""
    _setup_logging()
    from energy_prices.ingestion.scheduler import run_ingestion
    from energy_prices.storage.db import init_db

    init_db()
    results = run_ingestion(source=source, start=_parse_date(start), end=_parse_date(end))

    table = Table(title="Ingestion results")
    table.add_column("Source")
    table.add_column("Rows", justify="right")
    for name, rows in results.items():
        table.add_row(name, f"{rows:,}")
    console.print(table)
    if sum(results.values()) == 0:
        console.print(
            "[yellow]No rows ingested.[/] Check API credentials in .env "
            "(ENTSO-E token / GME username+password), or use `energy seed-demo`."
        )


@app.command()
def forecast(
    market: str = typer.Option("all", help="all | elec | gas | ttf"),
    zone: str | None = typer.Option(None, help="Electricity zone (NORD…SARD or PUN)."),
    horizon_hours: int | None = typer.Option(None, help="Forecast horizon in hours."),
    calibrate: bool = typer.Option(
        False,
        "--calibrate",
        help="Conformalize (CQR) the intervals for honest coverage. "
        "Electricity is always CQR-calibrated; this flag only adds CQR to gas/TTF.",
    ),
) -> None:
    """Compute and persist probabilistic forecasts."""
    _setup_logging()
    from energy_prices.forecasting import runner

    market = market.lower()
    if market == "all":
        saved = runner.run_all(calibrate=calibrate)
    elif market in _MARKET_ALIASES and _MARKET_ALIASES[market] == Market.ELEC_DAYAHEAD.value:
        if zone:
            saved = runner.run_forecasts(
                Market.ELEC_DAYAHEAD.value, _validate_elec_zone(zone), horizon_hours,
                calibrate=calibrate,
            )
        else:
            saved = runner.run_all_electricity_zones(calibrate=calibrate)
    elif market in _MARKET_ALIASES:
        if zone is not None:
            raise typer.BadParameter(
                f"--zone is only valid for electricity; got market={market!r}"
            )
        saved = runner.run_forecasts(_MARKET_ALIASES[market], None, horizon_hours, calibrate=calibrate)
    else:
        raise typer.BadParameter(f"unknown market {market!r}")
    console.print(f"[green]Saved {saved:,} forecast rows.[/]")


_MODEL_FACTORIES = {
    "baseline": ("energy_prices.models.baseline", "SeasonalNaiveForecaster"),
    "lightgbm": ("energy_prices.models.lgbm", "LightGBMForecaster"),
    "lear": ("energy_prices.models.lear", "LearForecaster"),
    "sarimax": ("energy_prices.models.gas_sarimax", "SarimaxForecaster"),
}


@app.command()
def backtest(
    market: str = typer.Option("elec", help="elec | gas | ttf"),
    zone: str = typer.Option("PUN", help="Electricity zone for the backtest."),
    model: str = typer.Option("lightgbm", help="baseline | lightgbm | lear | sarimax | ensemble"),
    horizon: int = typer.Option(
        None,
        help="Forecast horizon in periods of the series resolution. "
        "Default: one full delivery day (96 for 15-min, 24 hourly, 1 daily).",
    ),
    windows: int = typer.Option(30, help="Number of rolling-origin windows."),
    calibrate: bool = typer.Option(
        False, "--calibrate", help="Wrap the model in CQR conformal calibration."
    ),
) -> None:
    """Rolling-origin walk-forward backtest with rMAE / pinball / coverage metrics."""
    _setup_logging()
    import importlib

    from energy_prices.forecasting.evaluation import walk_forward
    from energy_prices.storage.db import session_scope
    from energy_prices.storage.repositories import PriceRepository

    market_value = _MARKET_ALIASES.get(market.lower(), market)
    is_elec = market_value == Market.ELEC_DAYAHEAD.value
    lookup_zone = _validate_elec_zone(zone) if is_elec else None

    with session_scope() as session:
        prices = PriceRepository(session)
        df = prices.get_prices(market_value, zone=lookup_zone)
    if df.empty:
        console.print("[red]No price history.[/] Run `energy seed-demo` or `energy ingest` first.")
        raise typer.Exit(code=1)

    y = df["price"].astype(float).sort_index()
    y = y[~y.index.duplicated(keep="last")]

    # Default horizon = one full delivery day, in periods of the series' own
    # resolution (so 15-min PUN → 96 steps = true day-ahead, not 24 steps = 6h).
    if horizon is None:
        import pandas as pd

        spacing = y.index.to_series().diff().median()
        if pd.isna(spacing) or spacing <= pd.Timedelta(0):
            horizon = 24
        else:
            minutes = spacing / pd.Timedelta(minutes=1)
            horizon = max(1, int(round(24 * 60 / minutes)))

    def make_factory():
        if model == "ensemble":
            from energy_prices.models.ensemble import EnsembleForecaster

            return (EnsembleForecaster.for_gas if not is_elec else EnsembleForecaster)
        mod_name, cls_name = _MODEL_FACTORIES[model]
        cls = getattr(importlib.import_module(mod_name), cls_name)
        return cls

    if model not in _MODEL_FACTORIES and model != "ensemble":
        raise typer.BadParameter(f"unknown model {model!r}")

    base_factory = make_factory()
    if calibrate:
        from energy_prices.models.calibration import CalibratedForecaster

        def factory():
            return CalibratedForecaster(base_factory())
    else:
        factory = base_factory
    console.print(
        f"Backtesting [cyan]{model}{'+cqr' if calibrate else ''}[/] on "
        f"{market_value}/{lookup_zone or '-'} (horizon={horizon}, windows={windows})…"
    )
    result = walk_forward(y, factory, horizon=horizon, step=horizon, n_windows=windows)
    agg = result["aggregate"]

    table = Table(title=f"Backtest — {model} — {market_value}/{lookup_zone or '-'}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("rMAE (vs naive, <1 better)", f"{agg['rmae']:.3f}")
    table.add_row("MAE (EUR/MWh)", f"{agg['mae']:.2f}")
    table.add_row("Avg pinball", f"{agg['avg_pinball']:.3f}")
    cov = agg["coverage"]
    nom = agg["nominal_coverage"]
    table.add_row("Coverage / nominal", f"{cov:.2f} / {nom:.2f}" if cov == cov else "n/a")
    table.add_row("Windows / points", f"{agg['n_windows']} / {agg['n']:,}")
    console.print(table)


@app.command()
def dashboard(
    port: int = typer.Option(8501, help="Port to serve the dashboard on."),
) -> None:
    """Launch the Streamlit dashboard."""
    app_path = Path(__file__).parent / "dashboard" / "app.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(port),
    ]
    console.print(f"[green]Starting dashboard[/] on http://localhost:{port} …")
    subprocess.run(cmd, check=False)


@app.command()
def scheduler(
    once: bool = typer.Option(
        False, "--once", help="Run a single ingest+forecast+alert cycle and exit (dry run)."
    ),
    run_now: bool = typer.Option(True, help="Run the job once at startup before scheduling."),
    notify: bool = typer.Option(
        True, "--notify/--no-notify", help="Deliver triggered alerts to configured channels."
    ),
) -> None:
    """Start the blocking daily ingest + forecast scheduler (13:30 Europe/Rome).

    Use ``--once`` to run a single full cycle and exit (a 'giro a vuoto' to
    validate the pipeline) instead of starting the blocking loop.
    """
    _setup_logging()
    from energy_prices.ingestion.scheduler import run_once, start_scheduler

    if once:
        summary = run_once(notify=notify)
        console.print("[green]Ciclo completato[/] (dry run).")
        table = Table(title="daily_job — riepilogo")
        table.add_column("Fase")
        table.add_column("Risultato", justify="right")
        ing = summary.get("ingested", {})
        table.add_row("Ingest (righe)", f"{sum(ing.values()):,}" if ing else "0")
        table.add_row("Forecast (righe)", f"{summary.get('forecast_rows', 0):,}")
        table.add_row("Alert attivi", str(summary.get("alerts_triggered", 0)))
        disp = summary.get("dispatch") or {}
        table.add_row("Alert consegnati", str(disp.get("delivered", 0)))
        console.print(table)
        return

    start_scheduler(run_now=run_now)


@app.command("gme-inspect")
def gme_inspect_cmd(
    segment: str = typer.Option("MGP", help="GME segment, e.g. MGP or MGP-GAS."),
    data_name: str = typer.Option("ME_ZonalPrices", help="GME DataName."),
    days: int = typer.Option(2, help="How many days back to sample."),
) -> None:
    """Fetch a small GME dataset and show its raw fields (validate the parser).

    Requires GME credentials in .env. Use this once when real access arrives to
    confirm the live field names/format before trusting ingestion.
    """
    _setup_logging()
    import json

    from energy_prices.ingestion.gme_client import inspect as gme_inspect

    end = dt.datetime.now(dt.UTC).date()
    info = gme_inspect(segment, data_name, end - dt.timedelta(days=days), end)
    if not info.get("ok"):
        console.print(f"[red]GME inspect failed:[/] {info.get('error')}")
        raise typer.Exit(code=1)
    console.print(f"[green]OK[/] {segment}/{data_name} {info['window']} — "
                  f"{info['n_records']} records")
    console.print(f"[bold]Field names:[/] {info['field_names']}")
    console.print("[bold]Sample records:[/]")
    console.print(json.dumps(info["samples"], indent=2, default=str, ensure_ascii=False))
    console.print("[bold]Mapped preview (what the parser would store):[/]")
    console.print(json.dumps(info["mapped_preview"], indent=2, default=str, ensure_ascii=False))


@app.command()
def backfill(
    start: str = typer.Option(..., "--from", help="Start date YYYY-MM-DD (e.g. 2015-01-01)."),
    end: str | None = typer.Option(None, "--to", help="End date YYYY-MM-DD (default: today)."),
    source: str = typer.Option("all", help="all | entsoe | gme | ttf | gie | weather"),
    chunk_days: int | None = typer.Option(
        None, help="Days per request chunk (default: source-aware; small for GME)."
    ),
    skip_gas: bool = typer.Option(
        False, "--skip-gas", help="Skip the GME gas sub-request (halves GME calls)."
    ),
) -> None:
    """Historical backfill in bounded chunks (respects API query/quotas limits)."""
    _setup_logging()
    from energy_prices.ingestion.scheduler import run_backfill
    from energy_prices.storage.db import init_db

    init_db()
    totals = run_backfill(
        source=source,
        start=_parse_date(start),
        end=_parse_date(end),
        chunk_days=chunk_days,
        skip_gas=skip_gas,
    )
    table = Table(title="Backfill results")
    table.add_column("Source")
    table.add_column("Rows", justify="right")
    for name, rows in totals.items():
        table.add_row(name, f"{rows:,}")
    console.print(table)


@app.command()
def alerts(
    notify: bool = typer.Option(
        False, "--notify", help="Deliver triggered alerts to configured channels (webhook/email)."
    ),
) -> None:
    """Evaluate the default price-alert rules against the latest forecasts.

    With ``--notify`` the triggered alerts are also dispatched to the configured
    webhook (n8n) and/or SMTP email. With no channel configured, the payload
    that WOULD be sent is logged (stub mode).
    """
    _setup_logging()
    from energy_prices.alerts import evaluate_alerts
    from energy_prices.storage.db import session_scope

    with session_scope() as session:
        triggered = evaluate_alerts(session)
    if not triggered:
        console.print("[green]Nessun alert.[/] Tutte le previsioni sono entro le soglie.")
        return
    table = Table(title=f"⚠️  {len(triggered)} alert attivi")
    table.add_column("Regola")
    table.add_column("Valore peggiore", justify="right")
    table.add_column("Quando")
    table.add_column("Step")
    for a in triggered:
        when = a["worst_target"].strftime("%Y-%m-%d %H:%M") if a.get("worst_target") else "—"
        table.add_row(a["rule"], f"{a['worst_value']:.1f} €/MWh", when, str(a["n_crossings"]))
    console.print(table)

    if notify:
        from energy_prices.notifications import dispatch_alerts

        result = dispatch_alerts(triggered)
        if result.get("skipped"):
            console.print(
                "[yellow]Nessun canale configurato[/] — payload stub loggato. "
                "Imposta ENERGY_ALERT_WEBHOOK_URL (n8n) o ENERGY_SMTP_* in .env."
            )
        else:
            chans = ", ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in result["channels"].items())
            console.print(f"[green]Consegnati {result['delivered']} alert[/] ({chans}).")


if __name__ == "__main__":  # pragma: no cover
    app()
