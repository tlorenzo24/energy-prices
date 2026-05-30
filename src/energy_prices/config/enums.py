"""Market, zone and resolution enums + the canonical Italian zone code mappings.

Grounded in the 2025 Italian market reform (verified 2026-05-30):
- Day-ahead (MGP) clears on 7 zonal prices; "PUN Index GME" is an ex-post
  volume-weighted index, no longer a settlement price.
- From 2025-10-01 the MGP resolution is 15 minutes (96 points/day); 60 minutes
  before. Code must treat both.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum

# The trading-day on which the SDAC 15-minute MTU went live (delivery 2025-10-01).
QUARTER_HOUR_GOLIVE = dt.date(2025, 10, 1)


class Market(str, Enum):
    """Logical markets we ingest and forecast."""

    ELEC_DAYAHEAD = "elec_dayahead"  # MGP, zonal + PUN Index
    GAS_DAYAHEAD = "gas_dayahead"    # MGP-GAS / PSV (IGI day-ahead index)
    TTF = "ttf"                      # Dutch TTF benchmark (front-month proxy)


class Zone(str, Enum):
    """Italian electricity bidding zones (post-2021 set) + the PUN index pseudo-zone."""

    NORD = "NORD"
    CNOR = "CNOR"
    CSUD = "CSUD"
    SUD = "SUD"
    CALA = "CALA"
    SICI = "SICI"
    SARD = "SARD"
    PUN = "PUN"  # not a physical zone: the ex-post PUN Index GME

    @property
    def is_physical(self) -> bool:
        return self is not Zone.PUN


class Resolution(int, Enum):
    """Market time unit in minutes."""

    QUARTER_HOUR = 15
    HALF_HOUR = 30
    HOUR = 60
    DAILY = 1440


# The 7 physical bidding zones (PUN excluded — it is a derived index).
MARKET_ZONES: list[Zone] = [
    Zone.NORD,
    Zone.CNOR,
    Zone.CSUD,
    Zone.SUD,
    Zone.CALA,
    Zone.SICI,
    Zone.SARD,
]

# ENTSO-E country_code strings used by entsoe-py for day-ahead price queries.
ENTSOE_ZONE_CODE: dict[Zone, str] = {
    Zone.NORD: "IT_NORD",
    Zone.CNOR: "IT_CNOR",
    Zone.CSUD: "IT_CSUD",
    Zone.SUD: "IT_SUD",
    Zone.CALA: "IT_CALA",
    Zone.SICI: "IT_SICI",
    Zone.SARD: "IT_SARD",
}

# Raw ENTSO-E EIC area codes (for direct REST calls / reference).
EIC_CODE: dict[Zone, str] = {
    Zone.NORD: "10Y1001A1001A73I",
    Zone.CNOR: "10Y1001A1001A70O",
    Zone.CSUD: "10Y1001A1001A71M",
    Zone.SUD: "10Y1001A1001A788",
    Zone.CALA: "10Y1001C--00096J",
    Zone.SICI: "10Y1001A1001A75E",
    Zone.SARD: "10Y1001A1001A74G",
}

# Approximate zonal consumption weights for reconstructing a PUN-like index when
# the official GME PUN Index is unavailable (e.g. ENTSO-E-only mode). These are
# rough load shares and should be replaced by GME's official PUN Index when the
# GME API is connected. Must sum to ~1.0.
PUN_ZONE_WEIGHTS: dict[Zone, float] = {
    Zone.NORD: 0.55,
    Zone.CNOR: 0.10,
    Zone.CSUD: 0.14,
    Zone.SUD: 0.08,
    Zone.CALA: 0.03,
    Zone.SICI: 0.06,
    Zone.SARD: 0.04,
}


def resolution_for_delivery(day: dt.date) -> Resolution:
    """Return the expected electricity MTU for a given delivery day."""
    return Resolution.QUARTER_HOUR if day >= QUARTER_HOUR_GOLIVE else Resolution.HOUR
