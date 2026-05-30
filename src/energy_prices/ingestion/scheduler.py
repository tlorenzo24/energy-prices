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


def _resolve_sources(source: str) -> list[str]:
    if source == "all":
        return list(ALL_SOURCES)
    if source not in SOURCE_MODULES:
        raise ValueError(
            f"unknown source {source!r}; choose from: all, {', '.join(ALL_SOURCES)}"
        )
    return [source]


def run_ingestion(
    source: str = "all",
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> dict[str, int]:
    """Ingest one or all sources for a date window. Returns {source: rows}.

    Each source isolates its own transaction/session so one failing source never
    rolls back another. Sources without configured credentials log a warning and
    contribute 0 rows.
    """
    today = dt.datetime.now(dt.UTC).date()
    start = start or (today - dt.timedelta(days=_DEFAULT_LOOKBACK_DAYS))
    end = end or (today + dt.timedelta(days=_DEFAULT_LOOKAHEAD_DAYS))

    results: dict[str, int] = {}
    for name in _resolve_sources(source):
        module = importlib.import_module(SOURCE_MODULES[name])
        try:
            with session_scope() as session:
                rows = module.ingest(session, start, end)
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
    chunk_days: int = 180,
) -> dict[str, int]:
    """Historical backfill: ingest a wide range in bounded chunks.

    Chunking keeps each request within ENTSO-E's ~1-year query cap and GME's
    per-call quotas, and lets a long backfill make steady, resumable progress
    (each chunk commits independently). Returns cumulative {source: rows}.
    """
    today = dt.datetime.now(dt.UTC).date()
    start = start or dt.date(2015, 1, 1)
    end = end or today
    if start > end:
        raise ValueError(f"start {start} is after end {end}")

    totals: dict[str, int] = {}
    cursor = start
    step = dt.timedelta(days=max(1, chunk_days))
    while cursor <= end:
        chunk_end = min(cursor + step - dt.timedelta(days=1), end)
        logger.info("backfill chunk %s..%s (source=%s)", cursor, chunk_end, source)
        chunk = run_ingestion(source=source, start=cursor, end=chunk_end)
        for name, rows in chunk.items():
            totals[name] = totals.get(name, 0) + rows
        cursor = chunk_end + dt.timedelta(days=1)
    logger.info("backfill complete %s..%s: %s", start, end, totals)
    return totals


def daily_job() -> None:
    """Full daily cycle: ingest all sources, then recompute all forecasts."""
    logger.info("daily_job: starting ingestion")
    ingested = run_ingestion("all")
    logger.info("daily_job: ingested %s", ingested)

    from energy_prices.forecasting.runner import run_all

    logger.info("daily_job: recomputing forecasts")
    saved = run_all()
    logger.info("daily_job: saved %s forecast rows", saved)


def start_scheduler(run_now: bool = True) -> None:
    """Start a blocking scheduler with a daily 13:30 Europe/Rome job.

    Used by the Docker `ingest` service. Runs the job once at startup (so a fresh
    deployment is populated immediately) unless `run_now` is False.
    """
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    settings = get_settings()
    init_db()

    if run_now:
        try:
            daily_job()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Initial daily_job failed: %s", exc)

    scheduler = BlockingScheduler(timezone=settings.timezone)
    scheduler.add_job(
        daily_job,
        CronTrigger(hour=13, minute=30, timezone=settings.timezone),
        id="daily_ingest_forecast",
        max_instances=1,
        coalesce=True,
    )
    logger.info("Scheduler started: daily job at 13:30 %s. Ctrl+C to stop.", settings.timezone)
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
