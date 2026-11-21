# Deployment-faithful evaluation (Phase 3)

## Question

The previous six-fold OOF used a fixed model (offline) or a rolling update that
kept the initial feature builder (online). Both are useful research protocols,
but neither exactly matches the deployed job, which rebuilds the feature
pipeline and retrains the three GBDT models every day on the last 800 realized
rows.

This phase asks a narrower question:

> If we replay the deployment logic over the six historical test folds, does the
> model still add value over Offline, Online, Buy & Hold, Constant 1.25 and
> Vol-only under the same allocation and cost rules?

It also checks whether the 111-row E14 outage that starts after the competition
period changes the picture, and whether a missing indicator is justified.

## Daily-retrain protocol

For every test day `t` in the existing six folds:

1. Select training rows with `target` non-null and `date_id < t`; keep the last
   800 rows. No future row, and no row whose target was not yet realized, enters
   the training set.
2. Rebuild `NotebookFeatureBuilder` on that window: missing-rate filtering,
   median imputation, derived features and calendar features are recomputed
   using only the past window. This is stricter than the online protocol, which
   keeps the initial builder.
3. Refit ElasticNet, XGBoost and LightGBM with the notebook hyperparameters.
4. Predict `t`, apply the notebook allocation policy and update the policy with
   the realized `t` target after the prediction.
5. Reinitialize the allocation state at the start of every fold, so folds are
   independent; smoothing never carries across a fold boundary.

The resulting OOF is saved with `train_start`, `train_end`, `n_train`,
component predictions, volatility and allocation, which makes the no-leakage
check auditable (`train_end < date_id` for every row).

## Strategies compared

| Strategy | Position source |
| --- | --- |
| Daily retrain | This phase's causal, rebuild-every-day OOF |
| Offline | Existing six-fold fixed-model OOF |
| Online | Existing six-fold rolling-update OOF |
| Buy & Hold | Constant 1.0 |
| Constant 1.25 | Constant 1.25 |
| Vol-only 1.0x | Same volatility target, smoothing and cost, no return prediction |

Model strategies and Vol-only include the notebook multiplicative transaction
cost. Buy & Hold and Constant 1.25 are static positions and therefore do not
trade every day; their turnover is a diagnostic, not a cost-matched comparison.

## Metrics

All strategies are evaluated on the same fold dates and the same labels:

- **Official Score**: the competition's adjusted Sharpe, with volatility and
  return penalties.
- **Raw Sharpe**: annualized Sharpe of the strategy return before the
  competition penalties.
- **Cumulative return**: compounded strategy return over the concatenated OOF.
- **Maximum drawdown**: worst peak-to-trough loss on the concatenated OOF.
- **Turnover**: mean absolute daily position change.

Both fold-level means and concatenated OOF metrics are reported; the former
gives each regime equal weight, the latter reflects a single capital path.

## E14 stress test

The recent E14 outage starts on 2026-06-16, after the last six-fold test day
(2025-12-03), so it cannot affect the six-fold OOF. To evaluate it, the live
panel's realized post-competition window is used as an exploratory holdout.
Three feature treatments are compared:

- `current_median`: production behavior, median fill;
- `drop_e14`: remove E14 after feature selection;
- `e14_indicator`: median fill plus a binary `E14_MISSING` feature.

The live source does not publish `forward_returns` or `risk_free_rate`, so this
window uses proxy scores (`fwd = target_metric`, `rf = 0`) and is not a
leaderboard-comparable result. The static setting is
`--retrain-every 999`; the rolling setting is `--retrain-every 20`. The combined
chart is produced by `scripts/plot_e14_comparison.py`.

## Results

Run: `models/20261121_193718_gdbt_daily_retrain_263fee` (1,488 test days;
`train_end < date_id` for every row). Comparison artifacts:
`reports/deployment_eval/`.

### Unified six-fold comparison

| Strategy | Official score (fold mean) | Official score (continuous) | Raw Sharpe (fold mean) | Raw Sharpe (continuous) | Cumulative return | Max drawdown | Turnover | Mean allocation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Daily retrain | 0.871 | 0.808 | 1.114 | 0.966 | 134.1% | -19.3% | 0.520 | 0.659 |
| Offline | 0.833 | 0.419 | 0.992 | 0.629 | 109.5% | -31.8% | 0.304 | 0.495 |
| Online | 1.253 | 1.271 | 1.435 | 1.277 | 191.7% | -20.1% | 0.514 | 0.652 |
| Buy & Hold | 1.498 | 1.280 | 1.498 | 1.280 | 203.9% | -18.0% | 0.001 | 1.000 |
| Constant 1.25 | 1.411 | 1.204 | 1.481 | 1.264 | 283.3% | -22.3% | 0.001 | 1.250 |
| Vol-only 1.0x | 1.267 | 1.151 | 1.486 | 1.329 | 354.9% | -20.6% | 0.066 | 1.200 |

Fold-level official scores:

| Fold | Daily retrain | Offline | Online | Buy & Hold | Constant 1.25 | Vol-only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2015 | 0.348 | 0.226 | 0.460 | 0.011 | -0.007 | 0.076 |
| 2017 | 0.572 | 0.316 | 0.521 | 2.784 | 2.643 | 2.942 |
| 2019 | 2.368 | 1.979 | 2.588 | 2.072 | 1.958 | 1.505 |
| 2021 | 0.386 | 2.360 | 1.205 | 2.105 | 1.989 | 1.917 |
| 2023 | 1.653 | 0.166 | 2.851 | 1.396 | 1.314 | 0.753 |
| 2025 | -0.099 | -0.049 | -0.107 | 0.618 | 0.569 | 0.405 |

The stricter protocol **does not reproduce the earlier Online result**. Daily
retrain averages 0.871 versus 1.253 for Online, and it is below Buy & Hold
(1.498), Constant 1.25 (1.411) and Vol-only (1.267). Its mean score is only 0.038
above Offline, while its continuous score (0.808) is better than Offline (0.419)
but still below Online and the simple baselines.

The difference is concentrated in 2021 (0.386 daily vs 1.205 Online) and 2023
(1.653 vs 2.851); Daily retrain only wins 2017 and, marginally, 2025 against
Online. Overall raw-prediction correlation with Online is 0.897 and allocation
correlation is 0.801, so the signals are similar; the gap is a different
allocation path, not a completely different model. Sampled 2021 windows still
contain 99 features, so the gap is not explained by features being dropped.
A plausible but unproven explanation is calibration/path drift from rebuilding
medians and refitting every day; this needs a controlled fixed-feature versus
dynamic-feature experiment before being treated as a cause.

Turnover is 0.520 for Daily retrain versus 0.514 for Online and 0.066 for
Vol-only. The model therefore buys no cost advantage from the daily rebuild.
Its maximum drawdown (-19.3%) is slightly better than Online (-20.1%), but this
does not compensate for lower return and Sharpe.

Fold-paired differences (Daily retrain minus baseline, official score) are
negative on average against every comparator: Online -0.382 (2/6 folds
positive), Buy & Hold -0.626 (3/6), Constant 1.25 -0.540 (3/6), Vol-only
-0.395 (3/6). With only six folds these are descriptive, not a significance
test; they show the loss is not driven by one outlier fold.

### E14 stress test

The recent E14 outage starts on **2026-06-16**, while the last six-fold test day
is **2025-12-03**. The overlap with the six-fold OOF is therefore **0 rows**;
the outage cannot explain any of the Phase 2 or Phase 3 OOF results.

For the post-competition live window (108 realized days, 2026-06-16 to
2026-11-17), proxy scores are:

| Setting | Variant | Proxy score | Cumulative return | Max drawdown | Turnover | Mean allocation |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Static pre-outage model | Current median | 1.864 | 7.9% | -2.6% | 0.432 | 0.332 |
| Static pre-outage model | Drop E14 | 1.647 | 8.4% | -3.0% | 0.477 | 0.602 |
| Static pre-outage model | E14 indicator | 1.864 | 7.9% | -2.6% | 0.432 | 0.332 |
| Rolling 20-day retrain | Current median | -0.155 | -1.6% | -9.5% | 0.440 | 0.932 |
| Rolling 20-day retrain | Drop E14 | -0.432 | -5.7% | -9.9% | 0.499 | 0.999 |
| Rolling 20-day retrain | E14 indicator | -0.155 | -1.6% | -9.5% | 0.440 | 0.932 |

The rolling retrain windows average 6.25% E14-missing rows; the latest 800-row
window is 13.9% missing. The static model is trained before the outage and sees
no missing E14, which is why the indicator is constant there.

Interpretation:

- Dropping E14 is worse in both settings, so median filling is preferable to
  dropping the feature in this sample.
- The indicator has **zero XGBoost importance** and produces identical
  allocations; there is no evidence that a missing indicator helps. Do not add
  it to production yet.
- The static model scores 1.864 while the rolling-retrain model scores -0.155.
  This difference is confounded by market regime, retraining and the
  post-competition period; it is **not** a causal E14 effect. The honest
  statement is that the outage is a monitoring risk, not a measured cause of the
  Phase 3 underperformance.
- These are proxy scores (`fwd = target_metric`, `rf = 0`), not official
  leaderboard scores. A controlled live A/B or a longer realized window is
  required before changing the production feature set.

## Verdict

The deployment-faithful replay **does not support entering Deep Learning on the
strength of the GBDT signal**. Daily retraining on the last 800 rows produces a
worse six-fold result than the existing Online protocol and remains below Buy &
Hold, Constant 1.25 and Vol-only. The Phase 2 conclusion is reinforced: the
model's apparent value is not robust to the exact deployment path, and no
reliable incremental value has been demonstrated.

Before any DL work, the higher-value next step is to stabilize the feature /
allocation path and test whether a fixed-feature rolling retrain (the Online
protocol) is the better deployment compromise. E14 should stay median-filled,
with missing-share monitoring and no indicator until a controlled test shows a
benefit.

## Remaining issues

- The E14 stress window is post-competition and proxy-scored; it cannot be
  compared to the six-fold official scores.
- Daily retrain has no W&B run; only local manifest/artifacts.
- The daily-versus-Online gap is not yet decomposed into feature-median drift,
  model-seed/optimizer effects and allocation path dependence.
- Buy & Hold / Constant 1.25 have no per-day trading cost; their turnover is a
  diagnostic and not a cost-matched comparison.
- The VPS deployment and three smoke tests in `deploy/README.md` are still
  unverified on the server.
- The dashboard is not implemented.


## Artifacts

- `scripts/run_daily_retrain.py`: causal daily-retrain six-fold OOF.
- `scripts/compare_deployment_modes.py`: unified comparison and plots.
- `scripts/run_e14_stress.py`: post-competition E14 feature-treatment test.
- `scripts/plot_e14_comparison.py`: combined static-vs-rolling chart.
- `models/<run>/oof_predictions.csv`, `cv_summary.json`: daily-retrain output.
- `reports/deployment_eval/summary.csv`, `fold_metrics.csv`,
  `manifest.json`, `plots/01..06`: unified comparison tables and figures.
- `reports/deployment_eval/e14_static/` and `e14_rolling20/`: E14 stress
  tables, importance and per-setting charts; the combined chart is
  `reports/deployment_eval/e14_static_vs_rolling.png`.
