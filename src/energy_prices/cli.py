"""Command-line interface for energy-prices.

Commands: init-db, seed-demo, ingest, forecast, backtest, dashboard, scheduler.
Run `energy --help` or `energy <command> --help` for details.
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

from energy_prices.config import Market, get_settings

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
    return dt.date.fromisoformat(value)


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
    console.print(f"[green]Database initialised[/] at {get_settings().database_url}")


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
) -> None:
    """Compute and persist probabilistic forecasts."""
    _setup_logging()
    from energy_prices.forecasting import runner

    market = market.lower()
    if market == "all":
        saved = runner.run_all()
    elif market in _MARKET_ALIASES and _MARKET_ALIASES[market] == Market.ELEC_DAYAHEAD.value:
        if zone:
            saved = runner.run_forecasts(Market.ELEC_DAYAHEAD.value, zone.upper(), horizon_hours)
        else:
            saved = runner.run_all_electricity_zones()
    elif market in _MARKET_ALIASES:
        saved = runner.run_forecasts(_MARKET_ALIASES[market], None, horizon_hours)
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
    horizon: int = typer.Option(24, help="Forecast horizon in periods per window."),
    windows: int = typer.Option(30, help="Number of rolling-origin windows."),
) -> None:
    """Rolling-origin walk-forward backtest with rMAE / pinball / coverage metrics."""
    _setup_logging()
    import importlib

    from energy_prices.forecasting.evaluation import walk_forward
    from energy_prices.storage.db import session_scope
    from energy_prices.storage.repositories import PriceRepository

    market_value = _MARKET_ALIASES.get(market.lower(), market)
    is_elec = market_value == Market.ELEC_DAYAHEAD.value
    lookup_zone = zone.upper() if is_elec else None

    with session_scope() as session:
        prices = PriceRepository(session)
        df = prices.get_prices(market_value, zone=lookup_zone)
    if df.empty:
        console.print("[red]No price history.[/] Run `energy seed-demo` or `energy ingest` first.")
        raise typer.Exit(code=1)

    y = df["price"].astype(float).sort_index()
    y = y[~y.index.duplicated(keep="last")]

    def make_factory():
        if model == "ensemble":
            from energy_prices.models.ensemble import EnsembleForecaster

            return (EnsembleForecaster.for_gas if not is_elec else EnsembleForecaster)
        mod_name, cls_name = _MODEL_FACTORIES[model]
        cls = getattr(importlib.import_module(mod_name), cls_name)
        return cls

    if model not in _MODEL_FACTORIES and model != "ensemble":
        raise typer.BadParameter(f"unknown model {model!r}")

    factory = make_factory()
    console.print(
        f"Backtesting [cyan]{model}[/] on {market_value}/{lookup_zone or '-'} "
        f"(horizon={horizon}, windows={windows})…"
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
    run_now: bool = typer.Option(True, help="Run the ingest+forecast job once at startup."),
) -> None:
    """Start the blocking daily ingest + forecast scheduler (13:30 Europe/Rome)."""
    _setup_logging()
    from energy_prices.ingestion.scheduler import start_scheduler

    start_scheduler(run_now=run_now)


if __name__ == "__main__":  # pragma: no cover
    app()
