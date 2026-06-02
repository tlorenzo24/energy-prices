"""Re-backtest matrix over the extended GME history (2026-06-02).

Settles the lightgbm-vs-ensemble decision on resolution-isolated, gap-free windows.
CQR calibration is DELIBERATELY excluded here: CalibratedForecaster does nested
rolling-origin recalibration (~35x the per-window cost of the base model), so it is
applied separately to the winner only — not across the whole grid.

Writes one result line per variant to data/rebacktest_results.txt with flush, so
progress is observable live (tail the file) even when run in the background.
"""

from __future__ import annotations

import datetime as dt
import importlib
import logging
import time
import warnings

import pandas as pd

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")

from energy_prices.config import Market  # noqa: E402
from energy_prices.forecasting.evaluation import walk_forward  # noqa: E402
from energy_prices.storage.db import session_scope  # noqa: E402
from energy_prices.storage.repositories import PriceRepository  # noqa: E402

UTC = dt.UTC
OUT = "data/rebacktest_results.txt"
_fh = open(OUT, "w", encoding="utf-8")


def emit(line: str) -> None:
    print(line, flush=True)
    _fh.write(line + "\n")
    _fh.flush()


def _factory(model: str, is_elec: bool):
    if model == "ensemble":
        from energy_prices.models.ensemble import EnsembleForecaster

        return EnsembleForecaster if is_elec else EnsembleForecaster.for_gas
    table = {
        "baseline": ("energy_prices.models.baseline", "SeasonalNaiveForecaster"),
        "lightgbm": ("energy_prices.models.lgbm", "LightGBMForecaster"),
        "sarimax": ("energy_prices.models.gas_sarimax", "SarimaxForecaster"),
    }
    mod, cls = table[model]
    return getattr(importlib.import_module(mod), cls)


def _load(market: str, zone: str | None, start: str | None, end: str | None) -> pd.Series:
    sdt = dt.datetime.fromisoformat(start).replace(tzinfo=UTC) if start else None
    edt = (
        dt.datetime.fromisoformat(end).replace(tzinfo=UTC) + dt.timedelta(days=1)
        if end else None
    )
    with session_scope() as session:
        df = PriceRepository(session).get_prices(market, zone=zone, start=sdt, end=edt)
    y = df["price"].astype(float).sort_index()
    return y[~y.index.duplicated(keep="last")]


def _horizon(y: pd.Series) -> int:
    spacing = y.index.to_series().diff().median()
    if pd.isna(spacing) or spacing <= pd.Timedelta(0):
        return 24
    return max(1, int(round(24 * 60 / (spacing / pd.Timedelta(minutes=1)))))


# label, market, zone, start, end, windows, is_elec
CASES = [
    ("PUN hourly crisis 21-22", Market.ELEC_DAYAHEAD.value, "PUN", "2021-01-01", "2022-04-25", 60, True),
    ("PUN hourly recent 25",    Market.ELEC_DAYAHEAD.value, "PUN", "2025-03-10", "2025-09-30", 60, True),
    ("PUN 15min post-reform",   Market.ELEC_DAYAHEAD.value, "PUN", "2025-10-01", None,        30, True),
    ("Gas day-ahead",           Market.GAS_DAYAHEAD.value,  None,  None,         None,        30, False),
]

emit(f"# re-backtest matrix (no CQR) — started {time.strftime('%H:%M:%S')}")
emit(f"{'case':<26}{'model':<11}{'win':>4}{'pts':>7}{'rMAE':>8}{'MAE':>9}{'pinball':>9}{'cov/nom':>12}{'sec':>7}")
emit("-" * 93)

for label, market, zone, start, end, windows, is_elec in CASES:
    y = _load(market, zone, start, end)
    if y.empty or len(y) < 50:
        emit(f"{label:<26}  (insufficient data: {len(y)} pts)")
        continue
    h = _horizon(y)
    models = ["lightgbm", "ensemble", "baseline"] if is_elec else ["sarimax", "ensemble", "baseline"]
    for model in models:
        factory = _factory(model, is_elec)
        t = time.time()
        try:
            r = walk_forward(y, factory, horizon=h, step=h, n_windows=windows)
        except Exception as exc:  # noqa: BLE001
            emit(f"{label:<26}{model:<11}  ERROR {type(exc).__name__}: {exc}")
            continue
        a = r["aggregate"]
        cov = f"{a['coverage']:.2f}/{a['nominal_coverage']:.2f}" if a["coverage"] == a["coverage"] else "n/a"
        emit(f"{label:<26}{model:<11}{a['n_windows']:>4}{a['n']:>7}{a['rmae']:>8.3f}"
             f"{a['mae']:>9.2f}{a['avg_pinball']:>9.3f}{cov:>12}{time.time() - t:>7.0f}")
    emit("-" * 93)

emit(f"MATRIX DONE — {time.strftime('%H:%M:%S')}")
_fh.close()
