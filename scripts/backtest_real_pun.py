"""Real-data backtest analysis for PUN (GME electricity, 15-min).

Runs rolling-origin walk-forward backtests on the REAL ingested PUN series and
reports rMAE / coverage / pinball for:

* horizon=24 (the literal CLI default, 6h ahead) — lightgbm with/without CQR
* horizon=96 (true day-ahead, one full local day) — lightgbm, ensemble,
  each with/without CQR
* a cal_fraction sweep for lightgbm+CQR at horizon=96 (coverage calibration)
* a Diebold-Mariano comparison of the point forecasts (ensemble vs lightgbm)

Everything is computed from the DB; no synthetic data. Output is plain text so it
can be captured and pasted into the task write-up.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from energy_prices.config.enums import Market
from energy_prices.forecasting.evaluation import diebold_mariano, walk_forward
from energy_prices.models.calibration import CalibratedForecaster
from energy_prices.models.ensemble import EnsembleForecaster
from energy_prices.models.lgbm import LightGBMForecaster
from energy_prices.storage.db import session_scope
from energy_prices.storage.repositories import PriceRepository


def load_pun():
    with session_scope() as s:
        df = PriceRepository(s).get_prices(Market.ELEC_DAYAHEAD.value, zone="PUN")
    y = df["price"].astype(float).sort_index()
    return y[~y.index.duplicated(keep="last")]


def fmt(agg: dict) -> str:
    cov = agg["coverage"]
    nom = agg["nominal_coverage"]
    covs = f"{cov:.3f}/{nom:.2f}" if cov == cov else "n/a"
    return (
        f"rMAE={agg['rmae']:.3f}  MAE={agg['mae']:.2f}  "
        f"pinball={agg['avg_pinball']:.3f}  cov/nom={covs}  "
        f"n={agg['n']} win={agg['n_windows']}"
    )


def lgbm_factory():
    return LightGBMForecaster()


def ens_factory():
    return EnsembleForecaster()


def cqr(base_factory, **kw):
    def f():
        return CalibratedForecaster(base_factory(), **kw)
    return f


def run(label, y, factory, horizon, windows, store=None):
    t0 = time.time()
    res = walk_forward(y, factory, horizon=horizon, step=horizon, n_windows=windows)
    dt = time.time() - t0
    print(f"  {label:<32} {fmt(res['aggregate'])}  ({dt:.0f}s)", flush=True)
    if store is not None:
        store[label] = res
    return res


def main():
    y = load_pun()
    print(f"PUN series: {len(y)} obs, {y.index.min()} -> {y.index.max()}", flush=True)
    print(f"price: mean={y.mean():.1f} std={y.std():.1f} "
          f"min={y.min():.1f} max={y.max():.1f}\n", flush=True)

    # --- 1) Literal CLI horizon (24 periods = 6h ahead) -------------------- #
    print("[H=24, windows=30]  (literal `energy backtest` default)", flush=True)
    run("lightgbm", y, lgbm_factory, 24, 30)
    run("lightgbm+cqr (cf=0.2)", y, cqr(lgbm_factory), 24, 30)

    # --- 2) True day-ahead (96 quarter-hours = one local day) -------------- #
    print("\n[H=96, windows=20]  (true day-ahead)", flush=True)
    store: dict = {}
    run("lightgbm", y, lgbm_factory, 96, 20, store)
    run("lightgbm+cqr (cf=0.2)", y, cqr(lgbm_factory), 96, 20, store)
    run("ensemble", y, ens_factory, 96, 20, store)
    run("ensemble+cqr (cf=0.2)", y, cqr(ens_factory), 96, 20, store)

    # --- 3) cal_fraction sweep for lightgbm+CQR @ H=96 --------------------- #
    print("\n[cal_fraction sweep — lightgbm+cqr, H=96, windows=20]", flush=True)
    for cf in (0.15, 0.25, 0.30, 0.40):
        run(f"lightgbm+cqr (cf={cf})", y, cqr(lgbm_factory, cal_fraction=cf), 96, 20)

    # --- 4) Diebold-Mariano: ensemble point vs lightgbm point @ H=96 ------- #
    print("\n[Diebold-Mariano — point forecast, H=96]", flush=True)
    lg = store.get("lightgbm")
    en = store.get("ensemble")
    if lg is not None and en is not None:
        pl = lg["predictions"]
        pe = en["predictions"]
        joined = pl[["y_true", "y_pred"]].join(
            pe[["y_pred"]], rsuffix="_ens", how="inner"
        ).dropna()
        err_lg = (joined["y_true"] - joined["y_pred"]).to_numpy()
        err_en = (joined["y_true"] - joined["y_pred_ens"]).to_numpy()
        mae_lg = float(np.mean(np.abs(err_lg)))
        mae_en = float(np.mean(np.abs(err_en)))
        # h=96: multi-step day-ahead -> use the matching DM truncation lag.
        stat, p = diebold_mariano(err_en, err_lg, h=96)
        print(f"  n={len(joined)}  MAE ensemble={mae_en:.2f}  MAE lightgbm={mae_lg:.2f}",
              flush=True)
        print(f"  DM(ensemble vs lightgbm) stat={stat:.3f} p={p:.4f}  "
              f"(stat<0 => ensemble better; p<0.05 => significant)", flush=True)
    else:
        print("  (missing runs for DM)", flush=True)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
