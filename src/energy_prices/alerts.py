"""Price-alert evaluation over the latest stored forecasts.

Pure-ish: reads the newest forecast run for each rule's (market, zone) and
reports threshold crossings. It does NOT send anything — sending (email, Slack,
n8n webhook) is a later, deployment-specific step; this module computes and
returns the triggered alerts so the CLI/scheduler/n8n can act on them.

A rule fires when, anywhere in the forecast horizon, the chosen quantile
(default the q0.5 median, or e.g. q0.9 for a "risk of spike" rule) crosses a
threshold in the given direction.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

from energy_prices.config import Market, Zone
from energy_prices.storage.repositories import ForecastRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlertRule:
    """One alert condition over a forecast series."""

    market: str
    zone: str | None
    threshold: float
    direction: str = "above"          # "above" | "below"
    quantile: float = 0.5             # which forecast quantile to test
    label: str | None = None

    def describe(self) -> str:
        z = f"/{self.zone}" if self.zone else ""
        return self.label or f"{self.market}{z} q{self.quantile:g} {self.direction} {self.threshold:g}"


def default_rules() -> list[AlertRule]:
    """A sensible starter set: spike risk on PUN, high/low gas, negative prices."""
    return [
        AlertRule(Market.ELEC_DAYAHEAD.value, Zone.PUN.value, 200.0, "above", 0.9,
                  label="PUN: rischio picco (>200 €/MWh, q90)"),
        AlertRule(Market.ELEC_DAYAHEAD.value, Zone.PUN.value, 0.0, "below", 0.1,
                  label="PUN: rischio prezzi negativi (q10 < 0)"),
        AlertRule(Market.GAS_DAYAHEAD.value, None, 60.0, "above", 0.5,
                  label="Gas (PSV): prezzo elevato (>60 €/MWh)"),
    ]


def evaluate_alerts(session, rules: list[AlertRule] | None = None) -> list[dict]:
    """Evaluate rules against the latest forecast run; return triggered alerts.

    Each triggered alert is a dict with the rule description, the worst crossing
    value, its target timestamp, the run timestamp, and how many horizon steps
    crossed. Rules with no stored forecast are skipped (logged at debug).
    """
    rules = rules if rules is not None else default_rules()
    repo = ForecastRepository(session)
    triggered: list[dict] = []

    for rule in rules:
        fc = repo.get_forecasts(rule.market, zone=rule.zone, latest=True)
        if fc.empty:
            logger.debug("No forecast for %s; skipping rule.", rule.describe())
            continue
        col = f"q{rule.quantile:g}"
        if col not in fc.columns:
            logger.debug("Quantile %s missing for %s; skipping.", col, rule.describe())
            continue

        series = fc[col].dropna()
        values = series.to_numpy(dtype=float)
        mask = values > rule.threshold if rule.direction == "above" else values < rule.threshold
        crossing = series[mask]
        if crossing.empty:
            continue

        # Report the most extreme crossing.
        worst_ts = crossing.idxmax() if rule.direction == "above" else crossing.idxmin()
        worst_val = float(crossing.loc[worst_ts])
        run_at = repo.latest_run_at(rule.market, zone=rule.zone)
        triggered.append(
            {
                "rule": rule.describe(),
                "market": rule.market,
                "zone": rule.zone,
                "quantile": rule.quantile,
                "direction": rule.direction,
                "threshold": rule.threshold,
                "worst_value": worst_val,
                "worst_target": worst_ts.to_pydatetime() if hasattr(worst_ts, "to_pydatetime") else worst_ts,
                "n_crossings": int(crossing.size),
                "run_at": run_at,
                "raised_at": dt.datetime.now(dt.UTC),
            }
        )
        logger.info("ALERT: %s — worst %.2f €/MWh at %s (%d steps).",
                    rule.describe(), worst_val, worst_ts, crossing.size)

    return triggered
