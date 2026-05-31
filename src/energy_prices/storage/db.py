"""Engine/session management and schema initialization.

Backend-agnostic via SQLAlchemy. SQLite for local dev (zero setup); Postgres +
TimescaleDB in production. Timescale hypertables/compression are applied only
when running on Postgres and the extension is available — everything works on
plain SQLite too.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from energy_prices.config import get_settings
from energy_prices.storage.models import Base

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    url = settings.database_url
    connect_args: dict = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    engine = create_engine(url, future=True, pool_pre_ping=True, connect_args=connect_args)
    return engine


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


def get_session() -> Session:
    """Return a new Session. Caller is responsible for closing it."""
    return _session_factory()()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session scope: commits on success, rolls back on error."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# Tables converted to Timescale hypertables, mapped to their partition column.
_HYPERTABLES: dict[str, str] = {
    "price_observations": "delivery_start",
    "forecasts": "target_start",
}


def _ensure_partition_in_pk(conn, table: str, time_col: str) -> None:
    """Rewrite the table PK to include the partition column (Timescale requires it).

    The ORM keeps a single-column ``id`` PK so SQLite's INTEGER-PRIMARY-KEY
    autoincrement keeps working for local dev/tests. On Postgres, Timescale
    rejects any unique index (including the PK) that omits the partition column,
    so here we widen the PK to ``(id, <time_col>)``. ``id`` still draws from its
    own sequence, so it stays effectively unique and the ORM identity map (keyed
    on ``id``) is unaffected. Idempotent: skips if ``time_col`` is already in the PK.
    """
    in_pk = conn.execute(
        text(
            "SELECT COUNT(*) FROM information_schema.key_column_usage k "
            "JOIN information_schema.table_constraints c "
            "  ON k.constraint_name = c.constraint_name "
            "WHERE c.table_name = :t AND c.constraint_type = 'PRIMARY KEY' "
            "  AND k.column_name = :col"
        ),
        {"t": table, "col": time_col},
    ).scalar()
    if in_pk:
        return  # PK already includes the partition column
    conn.execute(text(f'ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {table}_pkey;'))
    conn.execute(text(f'ALTER TABLE {table} ADD PRIMARY KEY (id, {time_col});'))
    logger.info("Widened %s primary key to (id, %s) for Timescale partitioning.", table, time_col)


def timescale_hypertable_count(engine: Engine) -> int:
    """Number of our tables that are registered Timescale hypertables (0 on plain PG)."""
    names = tuple(_HYPERTABLES)
    with engine.connect() as conn:
        try:
            rows = conn.execute(
                text(
                    "SELECT hypertable_name FROM timescaledb_information.hypertables "
                    "WHERE hypertable_name = ANY(:names)"
                ),
                {"names": list(names)},
            ).all()
        except Exception:  # timescaledb_information view absent -> not a Timescale DB
            return 0
    return len(rows)


def _enable_timescale(engine: Engine) -> bool:
    """Best-effort TimescaleDB setup on Postgres. Returns True if hypertables exist.

    No longer swallows failures silently: a create_hypertable error is logged at
    ERROR with the offending table, and the caller verifies the hypertable count
    so a broken Timescale deploy is loud, not invisible.
    """
    with engine.begin() as conn:
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))
        except Exception as exc:  # extension not installed — fine, plain Postgres
            logger.warning("TimescaleDB extension unavailable (%s); using plain Postgres.", exc)
            return False
        for table, time_col in _HYPERTABLES.items():
            try:
                _ensure_partition_in_pk(conn, table, time_col)
                conn.execute(
                    text(
                        "SELECT create_hypertable(:t, :c, "
                        "if_not_exists => TRUE, migrate_data => TRUE);"
                    ),
                    {"t": table, "c": time_col},
                )
            except Exception as exc:
                logger.error(
                    "Could not create Timescale hypertable %s on %s: %s",
                    table, time_col, exc,
                )
    count = timescale_hypertable_count(engine)
    if count < len(_HYPERTABLES):
        logger.error(
            "Timescale setup incomplete: only %d/%d hypertables created. "
            "Time-partitioning/compression will NOT be active.",
            count, len(_HYPERTABLES),
        )
    else:
        logger.info("TimescaleDB ready: %d hypertables active.", count)
    return count > 0


def init_db() -> None:
    """Create all tables (and Timescale hypertables on Postgres)."""
    settings = get_settings()
    engine = get_engine()
    Base.metadata.create_all(engine)
    if settings.is_postgres:
        _enable_timescale(engine)
    logger.info("Database initialized at %s", settings.database_url)
