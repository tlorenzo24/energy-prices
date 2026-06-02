# PUN day-ahead backtest — REAL data (242 days, refreshed 2026-05-31)

> ## Update 2026-06-02 — extended hourly history (multi-regime, ~3y)
>
> A GME **hourly** backfill (2021-01 → 2025-09-30) added ~3 years of pre-reform
> data, enabling a backtest across distinct regimes that the 242-day 15-min sample
> could not reach. Two large gap-free blocks were used (the backfill left 6 mid-range
> gaps from GME rate-limiting). Reproduce with `scripts/rebacktest_matrix.py`
> (broad, CQR off) and `scripts/confirm_cqr_gas.py` (CQR coverage check).
>
> **LightGBM beats the ensemble in every regime** (rMAE, raw LightGBM, no CQR):
>
> | regime | LightGBM | ensemble (LEAR+LGBM) | baseline |
> |---|---|---|---|
> | hourly crisis 2021-04→2022-04 (60 win) | **0.477** | 0.611 | 1.001 |
> | hourly recent 2025-03→09 (60 win)      | **0.731** | 0.937 | 1.014 |
> | 15-min post-reform 2025-10→ (30 win)   | **0.672** | 0.826 | 1.001 |
>
> This **confirms the LightGBM decision** on far more data and regime diversity than
> the original 242-day run — the conclusion is now robust, not sample-specific.
>
> **CQR coverage (LightGBM+CQR, hourly recent, 12 win):** rMAE 0.731 (unchanged),
> MAE 6.89, **coverage 0.79 / 0.80** — i.e. on the **hourly** series CQR lands
> essentially *on* nominal, NOT over-covering. This differs from the 15-min 242-day
> finding below (CQR ≈0.96, over-covers): the calibration behaviour is
> resolution/regime-dependent, so the "structural over-cover" claim is specific to
> the calmer 15-min tail, not universal. Net: LightGBM+CQR is well-justified on both.

Rolling-origin walk-forward on the **real GME electricity series**: **PUN, 15-min,
23 232 quarter-hours = 242 days** (2025-09-30 → 2026-05-30; price mean 121.8,
std 31.7, min 0.0, max 279.8 EUR/MWh), **48 windows**. Reproduce with
`scripts/backtest_real_pun.py` (uses `.venv`; no synthetic/demo data).

> **Supersedes the prior 46-day run.** That doc was computed before the deep
> backfill brought the DB to 242 days; `get_prices` has no row cap, so the old
> numbers were simply a smaller sample. The headline conclusions mostly hold, but
> **two things changed materially on the larger sample** — see findings 1–2.

Metric conventions: **rMAE** = MAE(model)/MAE(seasonal-naive same-hour-last-week);
`< 1` beats the naive benchmark. **coverage** = empirical fraction of actuals
inside the widest predictive band (q0.1–q0.9, **nominal 0.80**). Lower pinball is
better.

## Headline numbers

### H=96 — true day-ahead (production horizon: 24h × 15-min = 96 steps), 48 windows, n=4608
| model | rMAE | MAE €/MWh | avg pinball | coverage / nominal |
|---|---|---|---|---|
| **lightgbm**            | **0.689** | **16.55** | 6.287 | 0.579 / 0.80 |
| lightgbm + CQR (cf=0.2) | 0.689 | 16.55 | 7.212 | 0.959 / 0.80 |
| ensemble (LEAR+LGBM)| 0.821 | 19.72 | 7.349 | 0.684 / 0.80 |
| ensemble + CQR (cf=0.2) | 0.821 | 19.72 | 7.460 | 0.844 / 0.80 |

### H=24 — 6h-ahead intraday reference (24 periods on 15-min data), 48 windows, n=1152
| model | rMAE | MAE €/MWh | avg pinball | coverage / nominal |
|---|---|---|---|---|
| lightgbm            | 0.582 | 12.92 | 4.845 | 0.648 / 0.80 |
| lightgbm + CQR (cf=0.2) | 0.582 | 12.92 | 6.666 | 0.976 / 0.80 |

> The CLI `--horizon` is counted in **periods of the series resolution**; its
> default is now **one full delivery day** (96 on 15-min PUN = true day-ahead, not
> 24 = 6h). Use `energy backtest` with no `--horizon` for the production task.

## Calibration (CQR) — does coverage approach nominal?

cal_fraction sweep, **lightgbm + CQR, H=96, 48 windows**:

| cal_fraction | coverage / nominal |
|---|---|
| 0.20 | 0.959 / 0.80 |
| 0.30 | 0.964 / 0.80 |
| 0.40 | 0.968 / 0.80 |

**Finding (unchanged).** CQR **over-covers** (≈0.96–0.97 vs 0.80) on lightgbm and
**tuning `cal_fraction` does not fix it** — structural, not a sample-size effect.
The conformal offset, calibrated on the recent (more volatile) tail, over-widens
the bands for the calmer test horizon.

## Point-forecast comparison — Diebold-Mariano (H=96, n=4608, abs-error loss)

`ensemble vs lightgbm`: MAE 19.72 vs 16.55 → **DM stat = +3.493, p = 0.0005**
(positive ⇒ ensemble has higher loss; **significant**). **This changed on the
larger sample:** on 46 days the gap was *not* significant (p=0.188); on 242 days
**LightGBM significantly beats the ensemble**. The LEAR member (sklearn `LassoCV`
fallback) adds noise, not skill, at this data volume.

## Recommendations

1. **Point — LightGBM alone, now decisively.** rMAE 0.689 day-ahead, and the DM
   advantage over the ensemble is **significant** (p=0.0005) on 242 days — it was
   only suggestive on 46. All models still beat seasonal-naive (rMAE < 1).
2. **Intervals — no configuration is well-calibrated on 242 days.** lightgbm base
   *under*-covers badly (0.579), lightgbm+CQR *over*-covers (0.959), ensemble base
   *under*-covers (0.684), ensemble+CQR is **closest to nominal (0.844)**. The
   46-day finding that the ensemble's *native* bands sit ≈ nominal (≈0.806) **does
   NOT replicate** on the larger, calmer sample.
3. **Production model — DECIDED: LightGBM + CQR.** The prior config (ensemble,
   CQR off) was dominated on 242 days — significantly worse on point than lightgbm
   *and* under-covering (0.684). Production now emits **LightGBM wrapped in CQR**
   for electricity (`runner._select_model` returns `LightGBMForecaster`; the runner
   forces CQR on for elec). This is the **point-first / risk-aware** choice: best
   point error (rMAE 0.689) with conservative, over-covering (~0.96) bands — the
   safe failure mode for a trading-risk tool (never *under*-state risk). Gas
   (`psv_basis`) keeps CQR off (its native bands are already ≈ nominal). The
   alternative, *calibration-first* config (ensemble + CQR, coverage 0.844) was
   not chosen because its point error is significantly worse.
4. **Deep model (NHITS/TFT) — still DEFERRED.** 242 days remains data-starved for a
   deep net, and the ensemble's extra (linear EPF) member already fails to beat a
   single GBM here. Revisit after the **ENTSO-E multi-year backfill + exogenous
   drivers** land (load/wind+solar are still absent — see `runner` exog warning),
   which is the real unlock for both deep models and regime-aware modelling.
