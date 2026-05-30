# PUN day-ahead backtest — REAL data (not synthetic)

Rolling-origin walk-forward backtest on the **real GME electricity series** ingested
in the DB: **PUN, 15-min, 4416 quarter-hours = 46 days** (2026-04-14 → 2026-05-30;
price mean 117.3, std 43.8, min 0.0, max 215.6 EUR/MWh). Reproduce with
`scripts/backtest_real_pun.py` (uses `.venv`; no synthetic/demo data).

Metric conventions: **rMAE** = MAE(model)/MAE(seasonal-naive same-hour-last-week);
`< 1` beats the naive benchmark. **coverage** = empirical fraction of actuals inside
the widest predictive band (q0.1–q0.9, **nominal 0.80**). Lower pinball is better.

## Headline numbers

### H=96 — true day-ahead (production horizon: 24h × 15-min = 96 steps), 12 windows, n=1152
| model | rMAE | MAE €/MWh | avg pinball | coverage / nominal |
|---|---|---|---|---|
| lightgbm            | **0.654** | 14.52 | 5.584 | 0.540 / 0.80 |
| lightgbm + CQR (cf=0.2) | 0.654 | 14.52 | 5.994 | 0.961 / 0.80 |
| ensemble (LEAR+LGBM)| 0.731 | 16.24 | 5.790 | **0.806 / 0.80** |
| ensemble + CQR (cf=0.2) | 0.731 | 16.24 | 6.191 | 0.838 / 0.80 |

### H=24 — the *literal* `energy backtest` default (24 periods = only 6h on 15-min data), 20 windows, n=480
| model | rMAE | MAE €/MWh | avg pinball | coverage / nominal |
|---|---|---|---|---|
| lightgbm            | 0.666 | 12.76 | 4.803 | 0.598 / 0.80 |
| lightgbm + CQR (cf=0.2) | 0.666 | 12.76 | 5.659 | 0.979 / 0.80 |

> The CLI `--horizon` is counted in **periods**, so the default `24` evaluates a 6h
> horizon on 15-min data, **not** the day-ahead task. Use `--horizon 96` for the
> production day-ahead horizon. (A 30-window run of the exact user command
> `energy backtest --market elec --zone PUN` gave rMAE 0.685, coverage 0.58 —
> consistent with the 20-window 0.666 / 0.598 above.)

## Calibration (CQR) — does coverage approach nominal?

cal_fraction sweep, **lightgbm + CQR, H=96, 12 windows**:

| cal_fraction | coverage / nominal |
|---|---|
| 0.20 | 0.961 / 0.80 |
| 0.30 | 0.959 / 0.80 |
| 0.40 | 0.944 / 0.80 |

**Finding.** On this 46-day history CQR **over-covers** (≈0.94–0.98 vs 0.80) at both
horizons, and **tuning `cal_fraction` does not fix it** (0.96→0.94 from cf 0.2→0.4 —
structural, not a sample-size effect). The conformal offset is calibrated on the
recent (more volatile) tail and over-widens the bands for the calmer test horizon;
CQR was originally tuned on the synthetic MVP where the **base** bands *under*-covered
(≈0.63→0.80). The raw LightGBM bands instead badly *under*-cover (0.540 at H=96).

The **ensemble's native bands are already ≈ nominal (0.806)** without any
calibration — LEAR's residual-based quantiles widen LightGBM's too-narrow ones — so
CQR on the ensemble is unnecessary and nudges it slightly over (0.838).

## Point-forecast comparison — Diebold-Mariano (H=96, n=1152, abs-error loss)

`ensemble vs lightgbm`: MAE 16.24 vs 14.52 → **DM stat = +1.317, p = 0.188**
(positive ⇒ ensemble has higher loss; **not significant**). LightGBM is the better
point model here, but **not significantly**; the LEAR member (sklearn `LassoCV`
fallback on only 46 days) adds noise rather than skill.

## Recommendations

1. **Point**: best is LightGBM alone (rMAE 0.654 day-ahead). All models beat
   seasonal-naive (rMAE < 1), so the pipeline adds value.
2. **Intervals**: keep the **ensemble** for its near-nominal, robust bands; **leave
   `--calibrate` OFF for electricity by default** — the base ensemble is already
   calibrated and CQR over-covers on the current short history. Re-evaluate CQR (and
   the `cal_fraction` default, currently 0.2 — unchanged, sweep showed no gain) once
   more history accrues or if base coverage drifts.
3. **Deep model (NHITS/TFT) — DEFERRED, not promoted.** `torch`/`neuralforecast`
   are not installed (GB-scale install + the box's TLS interception make it
   fragile), and with only **46 days** a deep net is data-starved. Decisive evidence:
   adding even a strong linear EPF benchmark (LEAR) to LightGBM yields **no
   significant DM win** (p=0.19) and hurts the point error — a far more data-hungry
   deep model will not clear the "significant DM win vs the ensemble" promotion bar
   on this data volume. Revisit after the ENTSO-E multi-year backfill lands (the
   ≥2-year regime-aware plan), per `market-data-architecture`.
