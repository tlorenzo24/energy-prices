"""Hermetic delivery-path tests for energy_prices.notifications.

All network/SMTP I/O is monkeypatched — nothing leaves the box.
Settings are constructed explicitly so the autouse _hermetic_env fixture
(which blanks env vars) does not interfere with the values we need.
"""

from __future__ import annotations

import datetime as dt
import json
import smtplib
from unittest.mock import patch

from energy_prices.config.settings import Settings
from energy_prices.notifications import (
    _post_webhook,
    _redact_url,
    _send_email,
    dispatch_alerts,
)

# ---------------------------------------------------------------------------
# Shared alert fixture (matches shape used in test_smoke.py)
# ---------------------------------------------------------------------------
SAMPLE_ALERTS = [
    {
        "rule": "PUN q0.9 above 200",
        "market": "elec_dayahead",
        "zone": "PUN",
        "worst_value": 250.0,
        "worst_target": dt.datetime(2026, 5, 31, 19, tzinfo=dt.UTC),
        "n_crossings": 3,
        "run_at": dt.datetime(2026, 5, 30, tzinfo=dt.UTC),
        "raised_at": dt.datetime(2026, 5, 30, 12, tzinfo=dt.UTC),
    }
]

_WEBHOOK_URL = "https://n8n.example.com/webhook/secret-uuid-goes-here"
_WEBHOOK_TOKEN = "tok-abc123"


# ---------------------------------------------------------------------------
# _redact_url unit tests
# ---------------------------------------------------------------------------
def test_redact_url_strips_path():
    assert _redact_url(_WEBHOOK_URL) == "https://n8n.example.com"


def test_redact_url_none_returns_unset():
    assert _redact_url(None) == "<unset>"


def test_redact_url_empty_string_returns_unset():
    assert _redact_url("") == "<unset>"


# ---------------------------------------------------------------------------
# Webhook: success path
# ---------------------------------------------------------------------------
class _FakeResponse:
    status_code = 200

    def raise_for_status(self):
        pass  # no-op


def _make_webhook_settings(token: str | None = _WEBHOOK_TOKEN) -> Settings:
    return Settings(
        alert_webhook_url=_WEBHOOK_URL,
        alert_webhook_token=token,
        demo_mode=True,
    )


def test_webhook_sends_bearer_token(monkeypatch):
    """Authorization header must be 'Bearer <token>' when token is set."""
    recorded: dict = {}

    def fake_post(url, *, data, headers, timeout):
        recorded["headers"] = headers
        recorded["data"] = data
        return _FakeResponse()

    # _post_webhook does `import requests` inside the function body; patching
    # requests.post at the canonical location is sufficient.
    with patch("requests.post", side_effect=fake_post):
        result = _post_webhook({"n_alerts": 1, "alerts": SAMPLE_ALERTS}, _make_webhook_settings())

    assert result is True
    assert recorded["headers"]["Authorization"] == f"Bearer {_WEBHOOK_TOKEN}"


def test_webhook_no_auth_header_when_token_none(monkeypatch):
    """No Authorization header when token is None."""
    recorded: dict = {}

    def fake_post(url, *, data, headers, timeout):
        recorded["headers"] = headers
        return _FakeResponse()

    with patch("requests.post", side_effect=fake_post):
        _post_webhook({"n_alerts": 1, "alerts": SAMPLE_ALERTS}, _make_webhook_settings(token=None))

    assert "Authorization" not in recorded["headers"]


def test_webhook_body_is_valid_json_with_alerts(monkeypatch):
    """Posted data must be valid JSON that round-trips, including datetime serialisation."""
    from energy_prices.notifications import build_payload

    settings = _make_webhook_settings()
    payload = build_payload(SAMPLE_ALERTS, settings)
    recorded: dict = {}

    def fake_post(url, *, data, headers, timeout):
        recorded["data"] = data
        return _FakeResponse()

    with patch("requests.post", side_effect=fake_post):
        _post_webhook(payload, settings)

    parsed = json.loads(recorded["data"])
    assert parsed["n_alerts"] == 1
    # Datetime fields must have been serialised to ISO strings
    alert = parsed["alerts"][0]
    assert isinstance(alert["worst_target"], str)
    assert "2026-05-31" in alert["worst_target"]


def test_webhook_failure_does_not_propagate(monkeypatch):
    """A network error must be swallowed and return False."""

    def fake_post(url, *, data, headers, timeout):
        raise OSError("connection refused")

    with patch("requests.post", side_effect=fake_post):
        result = _post_webhook({"n_alerts": 1, "alerts": []}, _make_webhook_settings())

    assert result is False


def test_dispatch_webhook_channel_reported_on_failure(monkeypatch):
    """dispatch_alerts returns channels['webhook']=False when POST fails."""

    def fake_post(url, *, data, headers, timeout):
        raise OSError("timeout")

    settings = _make_webhook_settings()
    with patch("requests.post", side_effect=fake_post):
        result = dispatch_alerts(SAMPLE_ALERTS, settings)

    assert result["channels"]["webhook"] is False
    assert result["skipped"] is False


# ---------------------------------------------------------------------------
# Email: success path
# ---------------------------------------------------------------------------
def _make_email_settings() -> Settings:
    return Settings(
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_user="user@example.com",
        smtp_password="s3cr3t",
        smtp_from="energy-prices@zeusenergytrading.com",
        alert_email_to="a@x.com, b@y.com",
        smtp_starttls=True,
        demo_mode=True,
    )


class _FakeSMTP:
    """Context-manager SMTP stub that records calls."""

    def __init__(self):
        self.starttls_called = 0
        self.login_calls: list[tuple[str, str]] = []
        self.sent_message: object = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def starttls(self):
        self.starttls_called += 1

    def login(self, user, password):
        self.login_calls.append((user, password))

    def send_message(self, msg):
        self.sent_message = msg


def test_email_to_header(monkeypatch):
    """To header must equal the comma-separated recipients string."""
    fake = _FakeSMTP()

    with patch("smtplib.SMTP", return_value=fake):
        result = _send_email(
            __build_email_payload(_make_email_settings()),
            _make_email_settings(),
        )

    assert result is True
    assert fake.sent_message["To"] == "a@x.com, b@y.com"


def test_email_from_header(monkeypatch):
    settings = _make_email_settings()
    fake = _FakeSMTP()

    with patch("smtplib.SMTP", return_value=fake):
        _send_email(__build_email_payload(settings), settings)

    assert fake.sent_message["From"] == "energy-prices@zeusenergytrading.com"


def test_email_subject_starts_with_prefix(monkeypatch):
    settings = _make_email_settings()
    fake = _FakeSMTP()

    with patch("smtplib.SMTP", return_value=fake):
        _send_email(__build_email_payload(settings), settings)

    assert fake.sent_message["Subject"].startswith("[energy-prices]")


def test_email_starttls_and_login_called_once(monkeypatch):
    settings = _make_email_settings()
    fake = _FakeSMTP()

    with patch("smtplib.SMTP", return_value=fake):
        _send_email(__build_email_payload(settings), settings)

    assert fake.starttls_called == 1
    assert len(fake.login_calls) == 1
    assert fake.login_calls[0] == ("user@example.com", "s3cr3t")


def test_email_body_contains_alert_text(monkeypatch):
    settings = _make_email_settings()
    fake = _FakeSMTP()

    with patch("smtplib.SMTP", return_value=fake):
        _send_email(__build_email_payload(settings), settings)

    body = fake.sent_message.get_body().get_content()
    assert "250" in body  # worst_value from SAMPLE_ALERTS


def test_email_failure_does_not_propagate(monkeypatch):
    """SMTP exception must be swallowed and return False."""

    def bad_smtp(*args, **kwargs):
        raise smtplib.SMTPException("connect failed")

    settings = _make_email_settings()
    with patch("smtplib.SMTP", side_effect=bad_smtp):
        result = _send_email(__build_email_payload(settings), settings)

    assert result is False


def test_dispatch_email_channel_reported_on_failure(monkeypatch):
    """dispatch_alerts returns channels['email']=False when SMTP fails."""

    def bad_smtp(*args, **kwargs):
        raise smtplib.SMTPException("connect failed")

    settings = _make_email_settings()
    with patch("smtplib.SMTP", side_effect=bad_smtp):
        result = dispatch_alerts(SAMPLE_ALERTS, settings)

    assert result["channels"]["email"] is False
    assert result["skipped"] is False


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------
def __build_email_payload(settings: Settings) -> dict:
    from energy_prices.notifications import build_payload
    return build_payload(SAMPLE_ALERTS, settings)
