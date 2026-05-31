"""Real-data backtest for PSV day-ahead gas: does TTF+basis beat PSV-only?

Rolling-origin walk-forward on the REAL ingested daily series (no synthetic data),
comparing the incumbent gas models against the new cointegration forecaster
``PsvBasisForecaster`` (PSV = TTF_forecast + mean-reverting basis).

Leak-safety: TTF is passed only as *history* via the ``ttf`` exog column; the
walk-forward harness truncates it to <= origin for training. ``PsvBasisForecaster``
forecasts TTF internally and ignores the realised future TTF the harness places in
``exog_future`` — so no look-ahead. The PSV-only baselines are run with ``exog=None``
so they can never see TTF at all.

Reported per horizon: rMAE (vs seasonal-naive same-day-last-week), MAE, avg pinball,
interval coverage (q0.1-q0.9, nominal 0.80), plus a Diebold-Mariano test of the new
model's point forecast against the PSV-only SARIMAX baseline.
"""

from __future__ import annotations

import os
import sys
import time

# Cap BLAS/LightGBM threads so the many-window backtest cannot oversubscribe the box.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

from energy_prices.config.enums import Market  # noqa: E402
from energy_prices.forecasting.evaluation import diebold_mariano, walk_forward  # noqa: E402
from energy_prices.models.ensemble import EnsembleForecaster  # noqa: E402
from energy_prices.models.gas_sarimax import SarimaxForecaster  # noqa: E402
from energy_prices.models.psv_basis import PsvBasisForecaster  # noqa: E402
from energy_prices.storage.db import session_scope  # noqa: E402
from energy_prices.storage.repositories import PriceRepository  # noqa: E402


def _daily(df):
    import pandas as pd
    s = df["price"].astype(float)
    s.index = pd.to_datetime(s.index, utc=True).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.dropna()


def load_series():
    import pandas as pd
    with session_scope() as s:
        psv = _daily(PriceRepository(s).get_prices(Market.GAS_DAYAHEAD.value))
        ttf = _daily(PriceRepository(s).get_prices(Market.TTF.value))
    psv.name = "price"
    exog = pd.DataFrame({"ttf": ttf})
    return psv, exog


def fmt(agg: dict) -> str:
    cov, nom = agg["coverage"], agg["nominal_coverage"]
    covs = f"{cov:.3f}/{nom:.2f}" if cov == cov and nom is not None else "n/a"
    return (
        f"rMAE={agg['rmae']:.3f}  MAE={agg['mae']:.2f}  "
        f"pinball={agg['avg_pinball']:.3f}  cov/nom={covs}  "
        f"n={agg['n']} win={agg['n_windows']}"
    )


def run(label, y, factory, horizon, windows, exog=None, store=None):
    t0 = time.time()
    res = walk_forward(y, factory, horizon=horizon, step=horizon, n_windows=windows, exog=exog)
    print(f"  {label:<34} {fmt(res['aggregate'])}  ({time.time() - t0:.0f}s)", flush=True)
    if store is not None:
        store[label] = res
    return res


# --- model factories --------------------------------------------------------
def f_sarimax():
    return SarimaxForecaster()


def f_gas_ensemble():
    # Current production baseline: SARIMAX + LightGBM (no TTF).
    return EnsembleForecaster.for_gas()


def f_psv_basis():
    return PsvBasisForecaster()  # the shipping production gas model


def main():
    psv, exog = load_series()
    print(f"PSV gas: {len(psv)} obs, {psv.index.min().date()} -> {psv.index.max().date()}  "
          f"mean={psv.mean():.2f} std={psv.std():.2f}", flush=True)
    print(f"TTF exog: {len(exog.dropna())} obs, {exog.dropna().index.min().date()} -> "
          f"{exog.dropna().index.max().date()}\n", flush=True)

    for horizon, windows in ((1, 60), (7, 24)):
        print(f"[H={horizon} day(s), windows={windows}]", flush=True)
        store: dict = {}
        run("sarimax (PSV only) [baseline]", psv, f_sarimax, horizon, windows, exog=None, store=store)
        run("gas-ensemble SARIMAX+LGBM [old prod]", psv, f_gas_ensemble, horizon, windows, exog=None, store=store)
        run("psv_basis (TTF+basis) [NEW prod]", psv, f_psv_basis, horizon, windows, exog=exog, store=store)

        # Diebold-Mariano: new psv_basis point vs PSV-only SARIMAX baseline.
        base, new = store.get("sarimax (PSV only) [baseline]"), store.get("psv_basis (TTF+basis) [NEW prod]")
        if base is not None and new is not None:
            j = base["predictions"][["y_true", "y_pred"]].join(
                new["predictions"][["y_pred"]], rsuffix="_new", how="inner"
            ).dropna()
            if len(j) >= 2:
                err_base = (j["y_true"] - j["y_pred"]).to_numpy()
                err_new = (j["y_true"] - j["y_pred_new"]).to_numpy()
                stat, p = diebold_mariano(err_new, err_base, h=horizon)
                print(f"  DM(psv_basis vs sarimax) n={len(j)} stat={stat:.3f} p={p:.4f}  "
                      f"(stat<0 => psv_basis better; p<0.05 => significant)", flush=True)
        print("", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
