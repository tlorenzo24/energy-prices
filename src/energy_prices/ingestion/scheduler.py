"""Ingestion orchestration + scheduled daily loop.

`run_ingestion()` is the single source of truth for "pull data from sources into
the DB"; both the CLI (`energy ingest`) and the scheduler use it. The scheduler
runs a daily job (~13:30 Europe/Rome, after the MGP noon close + publication)
that ingests fresh data and recomputes all forecasts.

Entry point for the Docker `ingest` service: `python -m energy_prices.ingestion.scheduler`.
"""

from __future__ import annotations

import datetime as dt
import importlib
import logging
import time

from energy_prices.config import get_settings
from energy_prices.storage.db import init_db, session_scope

logger = logging.getLogger(__name__)

# Logical source name -> module exposing `ingest(session, start, end) -> int`.
SOURCE_MODULES: dict[str, str] = {
    "entsoe": "energy_prices.ingestion.entsoe_client",
    "gme": "energy_prices.ingestion.gme_client",
    "ttf": "energy_prices.ingestion.ttf_client",
    "gie": "energy_prices.ingestion.gie_client",
    "weather": "energy_prices.ingestion.weather_client",
}
ALL_SOURCES = tuple(SOURCE_MODULES)

# Default incremental window: a little history (to backfill gaps) through the
# next two days (to capture freshly published day-ahead prices).
_DEFAULT_LOOKBACK_DAYS = 30
_DEFAULT_LOOKAHEAD_DAYS = 2

# Backfill chunk size per source (days). GME returns 15-minute data across ~25
# zones, so a chunk is huge and the API rate-limits aggressively (429) — keep
# its windows small. ENTSO-E caps queries near a year. Others are light/daily.
_BACKFILL_CHUNK_DAYS: dict[str, int] = {
    "gme": 15,
    "entsoe": 300,
}
_DEFAULT_CHUNK_DAYS = 180

# Pause between successive backfill chunks per source (seconds), to spread load
# and avoid tripping rate limits on a long historical pull.
_BACKFILL_CHUNK_PAUSE_S: dict[str, float] = {
    "gme": 5.0,
}


def _resolve_sources(source: str) -> list[str]:
    if source == "all":
        # Weather (Open-Meteo) is opt-in: its free tier is non-commercial-use
        # only, so it is excluded from "all" unless explicitly enabled. An
        # explicit `source="weather"` still honours the request.
        sources = list(ALL_SOURCES)
        if not get_settings().enable_weather and "weather" in sources:
            sources.remove("weather")
        return sources
    if source not in SOURCE_MODULES:
        raise ValueError(
            f"unknown source {source!r}; choose from: all, {', '.join(ALL_SOURCES)}"
        )
    return [source]


def run_ingestion(
    source: str = "all",
    start: dt.date | None = None,
    end: dt.date | None = None,
    skip_gas: bool = False,
) -> dict[str, int]:
    """Ingest one or all sources for a date window. Returns {source: rows}.

    Each source isolates its own transaction/session so one failing source never
    rolls back another. Sources without configured credentials log a warning and
    contribute 0 rows. ``skip_gas`` is forwarded to the GME source only (it omits
    the gas sub-request to halve GME calls); other sources ignore it.
    """
    today = dt.datetime.now(dt.UTC).date()
    start = start or (today - dt.timedelta(days=_DEFAULT_LOOKBACK_DAYS))
    end = end or (today + dt.timedelta(days=_DEFAULT_LOOKAHEAD_DAYS))

    results: dict[str, int] = {}
    for name in _resolve_sources(source):
        module = importlib.import_module(SOURCE_MODULES[name])
        kwargs = {"skip_gas": skip_gas} if name == "gme" else {}
        try:
            with session_scope() as session:
                rows = module.ingest(session, start, end, **kwargs)
            results[name] = int(rows or 0)
            logger.info("ingest[%s]: %s rows (%s -> %s)", name, results[name], start, end)
        except Exception as exc:  # noqa: BLE001 - isolate per-source failures
            logger.exception("ingest[%s] failed: %s", name, exc)
            results[name] = 0
    return results


def run_backfill(
    source: str = "all",
    start: dt.date | None = None,
    end: dt.date | None = None,
    chunk_days: int | None = None,
    skip_gas: bool = False,
) -> dict[str, int]:
    """Historical backfill: ingest a wide range in bounded chunks, per source.

    Each source is backfilled independently with its own chunk size (``None`` =
    source-aware default: small for GME's rate-limited 15-min data, larger for
    ENTSO-E) and an inter-chunk pause, so a long pull makes steady, resumable
    progress (each chunk commits on its own) without tripping rate limits. An
    explicit ``chunk_days`` overrides the per-source default for every source.
    ``skip_gas`` forwards to GME to halve its calls. Returns cumulative
    {source: rows}.
    """
    today = dt.datetime.now(dt.UTC).date()
    start = start or dt.date(2015, 1, 1)
    end = end or today
    if start > end:
        raise ValueError(f"start {start} is after end {end}")

    totals: dict[str, int] = {}
    for name in _resolve_sources(source):
        step_days = chunk_days or _BACKFILL_CHUNK_DAYS.get(name, _DEFAULT_CHUNK_DAYS)
        step = dt.timedelta(days=max(1, step_days))
        pause = _BACKFILL_CHUNK_PAUSE_S.get(name, 0.0)

        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + step - dt.timedelta(days=1), end)
            logger.info("backfill chunk %s..%s (source=%s)", cursor, chunk_end, name)
            chunk = run_ingestion(
                source=name, start=cursor, end=chunk_end, skip_gas=skip_gas
            )
            totals[name] = totals.get(name, 0) + chunk.get(name, 0)
            cursor = chunk_end + dt.timedelta(days=1)
            if pause and cursor <= end:
                time.sleep(pause)
    logger.info("backfill complete %s..%s: %s", start, end, totals)
    return totals


def daily_job(notify: bool = True) -> dict:
    """Full daily cycle: ingest all sources, recompute forecasts, fire alerts.

    Each stage is isolated so a failure in one never aborts the others (the
    scheduler must keep running day after day). Returns a summary dict with the
    per-source ingest counts, forecast rows saved, and the alert dispatch result.
    When ``notify`` is False, alerts are still evaluated but not delivered.
    """
    summary: dict = {"started_at": dt.datetime.now(dt.UTC).isoformat()}

    logger.info("daily_job: starting ingestion")
    try:
        ingested = run_ingestion("all")
    except Exception as exc:  # noqa: BLE001 - isolate stage
        logger.exception("daily_job: ingestion failed: %s", exc)
        ingested = {}
    summary["ingested"] = ingested
    logger.info("daily_job: ingested %s", ingested)

    logger.info("daily_job: recomputing forecasts")
    try:
        from energy_prices.forecasting.runner import run_all

        saved = run_all()
    except Exception as exc:  # noqa: BLE001 - isolate stage
        logger.exception("daily_job: forecasting failed: %s", exc)
        saved = 0
    summary["forecast_rows"] = saved
    logger.info("daily_job: saved %s forecast rows", saved)

    logger.info("daily_job: evaluating alerts")
    try:
        from energy_prices.alerts import evaluate_alerts
        from energy_prices.notifications import dispatch_alerts

        with session_scope() as session:
            triggered = evaluate_alerts(session)
        summary["alerts_triggered"] = len(triggered)
        summary["dispatch"] = dispatch_alerts(triggered) if notify else {"skipped": True}
    except Exception as exc:  # noqa: BLE001 - isolate stage
        logger.exception("daily_job: alert evaluation/dispatch failed: %s", exc)
        summary["alerts_triggered"] = 0

    summary["finished_at"] = dt.datetime.now(dt.UTC).isoformat()
    logger.info("daily_job: complete %s", summary)
    return summary


def run_once(notify: bool = True) -> dict:
    """Run a single ingest+forecast+alert cycle and return, without scheduling.

    This is the scheduler "dry run" (`energy scheduler --once`): exercises the
    exact daily pipeline once so you can validate it end-to-end without starting
    the blocking loop.
    """
    init_db()
    return daily_job(notify=notify)


def _bootstrap_if_empty() -> None:
    """Seed demo data on first boot when the DB is empty and demo_mode is on.

    Keeps a fresh Postgres/Docker deploy's dashboard from being blank before the
    first real ingest completes. No-op when data already exists or demo_mode is
    off (a credentialed prod deploy fills the DB via the run-on-start daily job).
    """
    settings = get_settings()
    if not settings.demo_mode:
        return
    from energy_prices.config import Market, Zone
    from energy_prices.ingestion.demo import seed_demo
    from energy_prices.storage.repositories import PriceRepository

    with session_scope() as session:
        if PriceRepository(session).latest_delivery(
            Market.ELEC_DAYAHEAD.value, zone=Zone.PUN.value
        ) is not None:
            return  # already has data
        rows = seed_demo(session)
        logger.info("Bootstrap: DB empty + demo_mode on -> seeded %d demo rows.", rows)


def start_scheduler(run_now: bool | None = None) -> None:
    """Start a blocking scheduler with the configured daily Europe/Rome job.

    Used by the Docker `ingest` service. Schedule (hour/minute), the
    run-once-at-startup behaviour and the misfire grace window are all driven by
    settings (``ENERGY_SCHEDULER_*``). ``run_now`` overrides
    ``settings.scheduler_run_on_start`` when given.
    """
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    settings = get_settings()
    init_db()
    _bootstrap_if_empty()

    if run_now is None:
        run_now = settings.scheduler_run_on_start
    if run_now:
        try:
            daily_job()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Initial daily_job failed: %s", exc)

    scheduler = BlockingScheduler(timezone=settings.timezone)
    scheduler.add_job(
        daily_job,
        CronTrigger(
            hour=settings.scheduler_hour,
            minute=settings.scheduler_minute,
            timezone=settings.timezone,
        ),
        id="daily_ingest_forecast",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=settings.scheduler_misfire_grace,
    )
    logger.info(
        "Scheduler started: daily job at %02d:%02d %s (misfire grace %ds). Ctrl+C to stop.",
        settings.scheduler_hour, settings.scheduler_minute,
        settings.timezone, settings.scheduler_misfire_grace,
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):  # pragma: no cover
        logger.info("Scheduler stopped.")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    start_scheduler()
