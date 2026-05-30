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

    # --- Behaviour ---
    timezone: str = "Europe/Rome"
    demo_mode: bool = True
    log_level: str = "INFO"

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgresql")

    @property
    def has_entsoe(self) -> bool:
        return bool(self.entsoe_api_token)

    @property
    def has_gme(self) -> bool:
        return bool(self.gme_api_username and self.gme_api_password)

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
