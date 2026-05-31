"""Application settings, loaded from environment / .env (prefix ENERGY_)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = three levels up from this file (src/energy_prices/config/settings.py).
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    """Central configuration. All fields overridable via ENERGY_* env vars."""

    model_config = SettingsConfigDict(
        env_prefix="ENERGY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Database ---
    database_url: str = Field(
        default=f"sqlite:///{(DATA_DIR / 'energy_prices.db').as_posix()}",
        description="SQLAlchemy URL. SQLite for local dev, Postgres+Timescale for prod.",
    )

    # --- ENTSO-E (primary electricity source) ---
    entsoe_api_token: str | None = None

    # --- GME official REST API ---
    gme_api_base_url: str = "https://api.mercatoelettrico.org/request"
    gme_api_username: str | None = None
    gme_api_password: str | None = None

    # --- Gas fundamentals ---
    agsi_api_key: str | None = None
    eia_api_key: str | None = None

    # --- Scheduler (daily ingest + forecast loop) ---
    # GME publishes the MGP results around 13:00–13:30 CET; the daily job runs a
    # little after to be safe. All overridable via ENERGY_SCHEDULER_* env vars.
    scheduler_hour: int = 13
    scheduler_minute: int = 30
    scheduler_run_on_start: bool = True
    # Grace window (seconds) for a missed fire (e.g. machine asleep / GME late).
    scheduler_misfire_grace: int = 3600

    # --- Alert delivery (webhook + email) ---
    # n8n (cloud or self-hosted) webhook: triggered alerts are POSTed as JSON.
    alert_webhook_url: str | None = None
    alert_webhook_token: str | None = None  # optional bearer/secret header
    # SMTP email fan-out (any provider). Leave host empty to disable email.
    alert_email_to: str | None = None       # comma-separated recipients
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_from: str | None = None
    smtp_starttls: bool = True

    # --- Dashboard ---
    # Optional shared-secret gate for the Streamlit dashboard (internal use).
    # Leave empty for no gate (rely on VPN/network isolation instead).
    dashboard_password: str | None = None

    # --- Behaviour ---
    timezone: str = "Europe/Rome"
    demo_mode: bool = True
    log_level: str = "INFO"
    # Open-Meteo weather ingestion. OFF by default: the free tier is
    # non-commercial-use only, so a commercial deploy must either set this true
    # under a paid/self-hosted Open-Meteo plan or leave it off.
    enable_weather: bool = False

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgresql")

    @property
    def has_entsoe(self) -> bool:
        return bool(self.entsoe_api_token)

    @property
    def has_gme(self) -> bool:
        return bool(self.gme_api_username and self.gme_api_password)

    @property
    def has_webhook(self) -> bool:
        return bool(self.alert_webhook_url)

    @property
    def has_email(self) -> bool:
        return bool(self.smtp_host and self.alert_email_to)

    @property
    def email_recipients(self) -> list[str]:
        if not self.alert_email_to:
            return []
        return [addr.strip() for addr in self.alert_email_to.split(",") if addr.strip()]

    def ensure_dirs(self) -> None:
        """Create runtime data directories if missing."""
        for sub in ("raw", "processed", "parquet"):
            (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton settings accessor."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
