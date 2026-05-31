"""Shared pytest fixtures + hermetic-environment guard.

The settings object reads the real ``.env`` (pydantic-settings ``env_file``), so on
a developer/CI box that has live credentials configured, tests would otherwise pick
up a real webhook URL, SMTP host or API token — making results machine-dependent and,
worse, letting the alert-dispatch tests fire a real POST at the production n8n webhook.

The autouse ``_hermetic_env`` fixture blanks every external-channel / credential env
var (overriding any ``.env`` value, since OS env beats the dotenv source) and clears the
cached settings singleton, so every test runs against a clean, offline configuration
regardless of the machine it runs on.
"""

from __future__ import annotations

import pytest

# External integrations + secrets that must never bleed in from a real .env.
_BLANKED_ENV = (
    "ENERGY_ALERT_WEBHOOK_URL",
    "ENERGY_ALERT_WEBHOOK_TOKEN",
    "ENERGY_ALERT_EMAIL_TO",
    "ENERGY_SMTP_HOST",
    "ENERGY_SMTP_USER",
    "ENERGY_SMTP_PASSWORD",
    "ENERGY_SMTP_FROM",
    "ENERGY_ENTSOE_API_TOKEN",
    "ENERGY_GME_API_USERNAME",
    "ENERGY_GME_API_PASSWORD",
    "ENERGY_AGSI_API_KEY",
    "ENERGY_EIA_API_KEY",
    "ENERGY_DASHBOARD_PASSWORD",
)


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """Blank external creds/channels and reset the cached settings, per test."""
    for key in _BLANKED_ENV:
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("ENERGY_DEMO_MODE", "true")

    from energy_prices.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    yield
    settings_mod.get_settings.cache_clear()
