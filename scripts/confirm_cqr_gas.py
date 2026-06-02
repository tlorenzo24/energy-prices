"""Closing confirmations for the 2026-06-02 re-backtest (2 targeted checks).

1. Electricity coverage WITH CQR: production always CQR-wraps LightGBM. The lean
   matrix ran LightGBM raw (coverage 0.51-0.64); this re-runs LightGBM+CQR on the
   recent hourly regime to confirm calibration restores ~0.80 nominal coverage.
   CQR is ~35x the base per-window cost, so few windows.

2. Gas: backtest the ACTUAL production model PsvBasisForecaster (PSV = TTF
   forecast + mean-reverting basis), which the lean matrix did not cover (it tested
   sarimax/ensemble). Needs the TTF history wired in as exog['ttf'].

Results stream to data/confirm_results.txt with flush (observable live).
"""

from __future__ import annotations

import datetime as dt
import logging
import time
import warnings

import pandas as pd

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")

from energy_prices.config import Market  # noqa: E402
from energy_prices.forecasting.evaluation import walk_forward  # noqa: E402
from energy_prices.models.calibration import CalibratedForecaster  # noqa: E402
from energy_prices.models.lgbm import LightGBMForecaster  # noqa: E402
from energy_prices.models.psv_basis import PsvBasisForecaster  # noqa: E402
from energy_prices.storage.db import session_scope  # noqa: E402
from energy_prices.storage.repositories import PriceRepository  # noqa: E402

UTC = dt.UTC
_fh = open("data/confirm_results.txt", "w", encoding="utf-8")


def emit(line: str) -> None:
    print(line, flush=True)
    _fh.write(line + "\n")
    _fh.flush()


def _series(market: str, zone: str | None, start=None, end=None) -> pd.Series:
    with session_scope() as s:
        df = PriceRepository(s).get_prices(market, zone=zone, start=start, end=end)
    y = df["price"].astype(float).sort_index()
    return y[~y.index.duplicated(keep="last")]


emit(f"# closing confirmations — started {time.strftime('%H:%M:%S')}")
emit(f"{'check':<34}{'win':>4}{'pts':>7}{'rMAE':>8}{'MAE':>9}{'pinball':>9}{'cov/nom':>12}{'sec':>7}")
emit("-" * 90)

# --- 1. Electricity LightGBM + CQR coverage (recent hourly regime) ---
y = _series(Market.ELEC_DAYAHEAD.value, "PUN",
            dt.datetime(2025, 3, 10, tzinfo=UTC), dt.datetime(2025, 10, 1, tzinfo=UTC))
t = time.time()
r = walk_forward(y, lambda: CalibratedForecaster(LightGBMForecaster(), horizon=24),
                 horizon=24, step=24, n_windows=12)
a = r["aggregate"]
cov = f"{a['coverage']:.2f}/{a['nominal_coverage']:.2f}"
emit(f"{'elec LightGBM+CQR (recent hr)':<34}{a['n_windows']:>4}{a['n']:>7}{a['rmae']:>8.3f}"
     f"{a['mae']:>9.2f}{a['avg_pinball']:>9.3f}{cov:>12}{time.time()-t:>7.0f}")

# --- 2. Gas production model psv_basis (TTF wired as exog) ---
gas = _series(Market.GAS_DAYAHEAD.value, None)
ttf = _series("ttf", None)
# Align TTF onto a daily index covering gas; psv_basis reads exog['ttf'] history.
exog = pd.DataFrame({"ttf": ttf})
t = time.time()
rg = walk_forward(gas, PsvBasisForecaster, horizon=1, step=1, n_windows=30, exog=exog)
ag = rg["aggregate"]
covg = (f"{ag['coverage']:.2f}/{ag['nominal_coverage']:.2f}"
        if ag["coverage"] == ag["coverage"] else "n/a")
emit(f"{'gas psv_basis (production)':<34}{ag['n_windows']:>4}{ag['n']:>7}{ag['rmae']:>8.3f}"
     f"{ag['mae']:>9.2f}{ag['avg_pinball']:>9.3f}{covg:>12}{time.time()-t:>7.0f}")

emit("-" * 90)
emit(f"DONE — {time.strftime('%H:%M:%S')}")
_fh.close()
