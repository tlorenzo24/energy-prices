"""SQLAlchemy ORM models.

Two normalized time-series tables (prices, forecasts) + an ingestion audit log.
Forecasts are stored one row per quantile so the schema supports both point and
probabilistic forecasts uniformly (quantile=0.5 is the point/median forecast).

All datetimes are stored in UTC. `delivery_start` / `target_start` mark the
start of the delivery interval; `resolution_minutes` gives its length.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    DateTime,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class PriceObservation(Base):
    """An observed (settled/published) market price for one delivery interval."""

    __tablename__ = "price_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market: Mapped[str] = mapped_column(String(32), nullable=False)
    # "" is the sentinel for national/zone-less series (gas, TTF). NOT NULL so the
    # uq_price_obs unique index actually fires on re-ingest: SQL treats NULL != NULL,
    # so a nullable zone would never match in ON CONFLICT and rows would duplicate.
    zone: Mapped[str] = mapped_column(String(8), nullable=False, default="", server_default="")
    delivery_start: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    resolution_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(8), default="EUR", nullable=False)
    unit: Mapped[str] = mapped_column(String(16), default="EUR/MWh", nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    ingested_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "market", "zone", "delivery_start", "source", name="uq_price_obs"
        ),
        Index("ix_price_lookup", "market", "zone", "delivery_start"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Price {self.market}/{self.zone} {self.delivery_start:%Y-%m-%d %H:%M} "
            f"{self.price:.2f} {self.unit} ({self.source})>"
        )


class Forecast(Base):
    """A forecasted price quantile for one future delivery interval, from one run."""

    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    market: Mapped[str] = mapped_column(String(32), nullable=False)
    # "" sentinel for national/zone-less series — see PriceObservation.zone.
    zone: Mapped[str] = mapped_column(String(8), nullable=False, default="", server_default="")
    target_start: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    resolution_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    model_name: Mapped[str] = mapped_column(String(48), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), default="0.1.0", nullable=False)
    quantile: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(16), default="EUR/MWh", nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "run_at", "market", "zone", "target_start", "model_name", "quantile",
            name="uq_forecast",
        ),
        Index("ix_forecast_lookup", "market", "zone", "model_name", "target_start"),
        Index("ix_forecast_run", "run_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Forecast {self.model_name} {self.market}/{self.zone} "
            f"{self.target_start:%Y-%m-%d %H:%M} q{self.quantile}={self.value:.2f}>"
        )


class ExogenousObservation(Base):
    """An exogenous driver series value (load/RES forecast, gas storage, weather, …).

    These are the features that make forecasts accurate. Stored generically by
    `series` name so new drivers need no schema change. LEAKAGE: store
    *forecasts* (available before gate close) for load/RES, and lag fundamentals
    to their real release time when joining as features.
    """

    __tablename__ = "exogenous_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    series: Mapped[str] = mapped_column(String(48), nullable=False)  # e.g. load_forecast
    # "" sentinel for national/zone-less series — see PriceObservation.zone.
    zone: Mapped[str] = mapped_column(String(8), nullable=False, default="", server_default="")
    valid_start: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    resolution_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str | None] = mapped_column(String(16), nullable=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    ingested_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("series", "zone", "valid_start", "source", name="uq_exog_obs"),
        Index("ix_exog_lookup", "series", "zone", "valid_start"),
    )


class IngestionRun(Base):
    """Audit log of each ingestion job execution."""

    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="running", nullable=False)
    rows_ingested: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    message: Mapped[str | None] = mapped_column(String(512), nullable=True)

    __table_args__ = (Index("ix_ingestion_source_time", "source", "started_at"),)
