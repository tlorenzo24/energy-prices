"""Alert delivery: push triggered price-alerts to real channels.

:func:`evaluate_alerts` (in :mod:`energy_prices.alerts`) only *computes* the
triggered alerts; this module *delivers* them. Two channels, each enabled purely
by configuration (so it is a no-op stub until you set the env vars):

* **Webhook** — POST a JSON payload to an n8n (cloud or self-hosted) webhook.
  n8n then does the fan-out (email / Slack / Teams / Telegram …). This is the
  recommended path: the routing logic lives in n8n, not in this codebase.
* **Email** — direct SMTP fan-out, for when you don't want an n8n hop.

Nothing here raises on a delivery failure: a broken channel must never crash the
scheduler's daily job. Failures are logged and reported in the returned summary.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import smtplib
from email.message import EmailMessage
from typing import Any

from energy_prices.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Network timeout for the webhook POST (seconds). Short: the daily job must not
# hang on a slow/unreachable endpoint.
_WEBHOOK_TIMEOUT = 15


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
def _json_default(value: Any) -> str:
    """JSON encoder fallback for datetimes (and anything else stringifiable)."""
    if isinstance(value, dt.datetime):
        return value.isoformat()
    return str(value)


def _serialize_alert(alert: dict) -> dict:
    """Make one triggered-alert dict JSON-clean (datetimes -> ISO strings)."""
    out: dict[str, Any] = {}
    for key, val in alert.items():
        out[key] = val.isoformat() if isinstance(val, dt.datetime) else val
    return out


def build_payload(alerts: list[dict], settings: Settings | None = None) -> dict:
    """Assemble the JSON body sent to the webhook / used in the email.

    Stable shape so an n8n workflow can bind to known fields:
    ``{service, generated_at, n_alerts, alerts:[...]}``.
    """
    settings = settings or get_settings()
    return {
        "service": "energy-prices",
        "environment": "postgres" if settings.is_postgres else "sqlite",
        "demo_mode": settings.demo_mode,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "n_alerts": len(alerts),
        "alerts": [_serialize_alert(a) for a in alerts],
    }


def render_text(payload: dict) -> str:
    """Human-readable summary of the payload (email body / log line)."""
    lines = [f"⚠️  energy-prices — {payload['n_alerts']} alert prezzo attivi", ""]
    for a in payload["alerts"]:
        when = a.get("worst_target") or "—"
        lines.append(
            f"• {a.get('rule')}: {a.get('worst_value'):.1f} €/MWh "
            f"@ {when} ({a.get('n_crossings')} step)"
        )
    lines += ["", f"Generato: {payload['generated_at']} (UTC)"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------
def _post_webhook(payload: dict, settings: Settings) -> bool:
    """POST the payload to the configured n8n webhook. Returns success."""
    import requests  # local import: keeps module import cheap

    headers = {"Content-Type": "application/json"}
    if settings.alert_webhook_token:
        headers["Authorization"] = f"Bearer {settings.alert_webhook_token}"
    try:
        resp = requests.post(
            settings.alert_webhook_url,  # type: ignore[arg-type]
            data=json.dumps(payload, default=_json_default),
            headers=headers,
            timeout=_WEBHOOK_TIMEOUT,
        )
        resp.raise_for_status()
        logger.info("Webhook delivered %d alert(s) -> %s (HTTP %s)",
                    payload["n_alerts"], settings.alert_webhook_url, resp.status_code)
        return True
    except Exception as exc:  # noqa: BLE001 - never crash the caller
        logger.error("Webhook delivery failed (%s): %s", settings.alert_webhook_url, exc)
        return False


def _send_email(payload: dict, settings: Settings) -> bool:
    """Send the alerts as a plain-text email via SMTP. Returns success."""
    if not settings.smtp_host:
        logger.warning("Email dispatch requested but ENERGY_SMTP_HOST is unset; skipping.")
        return False
    recipients = settings.email_recipients
    sender = settings.smtp_from or settings.smtp_user or "energy-prices@localhost"
    msg = EmailMessage()
    msg["Subject"] = f"[energy-prices] {payload['n_alerts']} alert prezzo"
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.set_content(render_text(payload))

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=_WEBHOOK_TIMEOUT) as smtp:
            if settings.smtp_starttls:
                smtp.starttls()
            if settings.smtp_user and settings.smtp_password:
                smtp.login(settings.smtp_user, settings.smtp_password)
            smtp.send_message(msg)
        logger.info("Email delivered %d alert(s) -> %s", payload["n_alerts"], recipients)
        return True
    except Exception as exc:  # noqa: BLE001 - never crash the caller
        logger.error("Email delivery failed (%s): %s", settings.smtp_host, exc)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def dispatch_alerts(alerts: list[dict], settings: Settings | None = None) -> dict:
    """Deliver triggered alerts to every configured channel.

    Returns a summary ``{channels: {webhook|email: bool}, delivered: int,
    skipped: bool}``. No configured channel => logs the payload as a stub and
    reports ``skipped=True`` (useful before n8n/SMTP are wired up).
    """
    settings = settings or get_settings()
    if not alerts:
        logger.info("No alerts to dispatch.")
        return {"channels": {}, "delivered": 0, "skipped": False}

    payload = build_payload(alerts, settings)
    channels: dict[str, bool] = {}

    if settings.has_webhook:
        channels["webhook"] = _post_webhook(payload, settings)
    if settings.has_email:
        channels["email"] = _send_email(payload, settings)

    if not channels:
        # Stub mode: no channel configured yet. Emit the exact payload that
        # WOULD be sent, so it's ready to wire into n8n / SMTP later.
        logger.warning(
            "No alert channel configured (set ENERGY_ALERT_WEBHOOK_URL or SMTP_*). "
            "Stub payload that would be sent:\n%s",
            json.dumps(payload, indent=2, default=_json_default, ensure_ascii=False),
        )
        return {"channels": {}, "delivered": 0, "skipped": True, "payload": payload}

    delivered = sum(1 for ok in channels.values() if ok)
    return {"channels": channels, "delivered": delivered, "skipped": False, "payload": payload}


def evaluate_and_dispatch(session, rules=None, settings: Settings | None = None) -> dict:
    """Convenience: evaluate the alert rules then dispatch the triggered ones.

    Returns ``{triggered: [...], dispatch: {...}}``.
    """
    from energy_prices.alerts import evaluate_alerts

    triggered = evaluate_alerts(session, rules)
    summary = dispatch_alerts(triggered, settings)
    return {"triggered": triggered, "dispatch": summary}
