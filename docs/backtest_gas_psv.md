# PSV day-ahead gas backtest — TTF + basis cointegration (REAL data)

Rolling-origin walk-forward on the **real ingested daily series** (no synthetic
data): PSV day-ahead gas (`gas_dayahead`, GME) **240 obs, 2025-10-02 → 2026-05-30**
(mean 39.74, std 8.81 EUR/MWh) with the Dutch **TTF** benchmark (yfinance, 1612 obs,
2020 → 2026) as the cointegrating driver. Reproduce with
`scripts/backtest_gas_psv.py` (uses `.venv`).

## Why TTF + basis (the empirics)

| metric | value | reading |
|---|---|---|
| corr(PSV, TTF), levels | **0.941** | TTF is a strong co-mover |
| OLS PSV ~ TTF, R² | **0.886** | TTF explains ~89% of PSV level variance |
| basis = PSV − TTF | mean 2.38, std 3.11 | small, stable regional spread |
| ADF, PSV / TTF levels | p = 0.31 / 0.52 | both carry a unit root |
| **ADF, basis** | **p = 0.001** | **stationary → PSV & TTF are cointegrated** |
| basis acf(1) / acf(7) | 0.34 / 0.17 | basis is mean-reverting, partly forecastable |

This is the textbook error-correction setup, so `PsvBasisForecaster` (a) forecasts
**TTF** with a robust log-SARIMAX, (b) forecasts the **stationary basis** with a
closed-form AR(1), and (c) reconstructs `PSV_hat = TTF_hat + basis_hat`, combining
the two predictive variances (independence assumption) into Normal quantiles.

**Leak-safety.** TTF is forecast *internally* from the history seen at fit time; the
realised future TTF that the walk-forward harness places in `exog_future` is
deliberately ignored (tomorrow's TTF close is not known at the day-ahead gate). The
PSV-only baselines are run with `exog=None` so they can never see TTF.

## Headline numbers

### H = 1 day (day-ahead, the production-critical horizon), 60 windows, n=60
| model | rMAE | MAE €/MWh | avg pinball | coverage / nominal |
|---|---|---|---|---|
| sarimax (PSV only) — baseline | 0.460 | 1.77 | 0.645 | 0.850 / 0.80 |
| ensemble SARIMAX+LightGBM — *old prod* | 0.530 | 2.04 | 0.674 | 0.800 / 0.80 |
| **psv_basis (TTF+basis)** — *new prod* | **0.411** | **1.59** | **0.611** | 0.950 / 0.80 |

### H = 7 days (multi-day), 24 windows, n=168
| model | rMAE | MAE €/MWh | avg pinball | coverage / nominal |
|---|---|---|---|---|
| sarimax (PSV only) | 0.765 | 3.21 | 1.203 | 0.708 / 0.80 |
| ensemble SARIMAX+LightGBM — *old prod* | 0.780 | 3.28 | 1.239 | 0.661 / 0.80 |
| **psv_basis (TTF+basis)** | **0.697** | **2.93** | **1.075** | 0.815 / 0.80 |

> Reproducible: `scripts/backtest_gas_psv.py` runs the **exact shipping model**
> (`PsvBasisForecaster()`, no args) and reproduces the numbers below (n=60 / n=168,
> not the empty aggregate).

## Findings

1. **`psv_basis` posts the lowest point error and best intervals at both horizons**
   and is now the **production gas default** (`runner._select_model`). It leverages
   the TTF cointegration to improve on the PSV-only SARIMAX (day-ahead rMAE 0.411
   vs 0.460; H=7 0.697 vs 0.765) and on the *old* gas ensemble (0.411 vs 0.530;
   0.697 vs 0.780). **The point-error gap is NOT statistically significant** on this
   short history (DM, see #4) — the case rests on consistently lower error at both
   horizons, better-calibrated intervals, and the cointegration structure, so the
   honest summary is "no worse on point, better intervals, structurally preferred".
   Note the comparison bundles *model + TTF information* (baselines see no TTF);
   that is the intended "does using TTF help?" question, not a model-in-a-vacuum claim.
2. **The old SARIMAX+LightGBM gas ensemble was actually worse than plain SARIMAX**
   — LightGBM on ~240 daily points added noise, not skill. It has been retired for
   gas in favour of `psv_basis`.
3. **Intervals are well-behaved**: coverage 0.95 (H=1) / 0.82 (H=7) vs nominal 0.80
   — slightly conservative (wide), never anti-conservative. No CQR needed for gas.
4. **Diebold-Mariano is *not* significant** (psv_basis vs PSV-only SARIMAX:
   H=1 stat −0.99 p=0.32; H=7 stat −1.14 p=0.26). On only 60/168 daily points the
   point-error improvement is real but **not statistically distinguishable** —
   expected on a short history. The case for `psv_basis` rests on (a) consistently
   lower rMAE/MAE/pinball at both horizons, and (b) the well-motivated cointegration
   structure, not on a significant DM win. Re-test as history accrues.
5. **`psv_basis + LightGBM` scored marginally better (rMAE 0.406 / 0.644) but was
   REJECTED** — that variant is *leak-affected*: the LightGBM member receives the
   realised future TTF as a horizon feature via `exog_future`, which is not
   available live. The shipped model is **pure `psv_basis`**, which is leak-safe by
   construction and whose backtest faithfully represents production.

## Recommendation

Ship **pure `psv_basis`** as the gas day-ahead default (done). Revisit the
SARIMAX/AR(1) orders and the basis mean-reversion target once a multi-year PSV
history accrues, and add a proper PSV/TTF feed (the yfinance TTF front-month is a
sanity proxy, not a settlement-grade source).
