# Feature-builder and retraining attribution (Phase 4)

Date: 2026-11-21  
Scope: explain the Phase 3 gap between the existing Online GBDT and the
deployment-faithful Daily Retrain. No Deep Learning was implemented.

## 1. Question and design

The hypothesis was that dynamic feature-builder reconstruction and daily model
retraining introduce calibration or prediction-path drift. Phase 4 separates the
two factors with a 2×2 experiment on the same six test folds.

| | Fixed model | Rolling daily model |
| --- | --- | --- |
| **Fixed feature builder** | **A**: existing Offline OOF | **B**: existing Online OOF |
| **Dynamic feature builder** | **C**: new Phase 4C OOF | **D**: existing Daily-Retrain OOF |

Operational definitions:

- **Fixed builder**: fit once per fold on the last 800 realized rows before the
  first test day. Its feature list, medians, missing-rate filter and
  `v1_median` stay frozen for that fold.
- **Dynamic builder**: refit on every test day on the last 800 realized rows.
  For C the fixed model keeps its original input schema; dynamic columns absent
  from that schema would be filled from the fold-start medians, and extra
  dynamic columns are ignored. In practice the fixed model had **zero missing
  fixed-schema features**; dynamic extra columns were 0–1.
- **Fixed model**: train ElasticNet + XGBoost + LightGBM once at fold start and
  never update.
- **Rolling model**:
  - B uses the existing Online implementation: append the previous test day's
    feature row and target, trim to the last 800 rows, refit all three models;
  - D uses a full daily refit on the last 800 rows, together with its dynamic
    builder.
- **Allocation**: the same `NotebookAllocationPolicy` in every arm, reset to
  zero at fold start, with the realized target fed back only after the
  prediction. Hyperparameters, multiplier, volatility window, clipping,
  smoothing and transaction cost are unchanged.

The 2×2 design gives four direct contrasts:

- `C - A`: builder effect with a fixed model;
- `D - B`: builder effect with a rolling model;
- `B - A`: retraining effect with a fixed builder;
- `D - C`: retraining effect with a dynamic builder;
- interaction: `(D - C) - (B - A)`.

## 2. Strict point-in-time protocol

For every prediction day `t`:

1. Training rows must have `date_id < t` and a non-null realized target.
2. Fixed builders use only the fold-start window; dynamic builders use the last
   800 realized rows before `t`.
3. The test row is transformed with those stored statistics; no future row is
   read.
4. The policy sees the target for `t` only after the allocation for `t` has
   been written.
5. The new C OOF was checked for `train_end < date_id` on every row; violations
   = 0, `n_train = 800`, duplicate dates = 0.

No hyperparameter or allocation value was selected using these OOF results.

## 3. Immutable OOFs

The attribution run copies every input OOF into
`reports/phase4_attribution/oof/` and records SHA-256 hashes in
`manifest.json`. The existing A/B/D files are never written to.

| Arm | Source | OOF |
| --- | --- | --- |
| A | `models/20261121_170606_gdbt_notebook_cv_offline_ae2a75` | existing Offline |
| B | `models/20261121_170851_gdbt_notebook_cv_online_0746df` | existing Online |
| C | `models/20261121_204048_gdbt_phase4_dynamic_fixed_a859e0` | new Phase 4C |
| D | `models/20261121_193718_gdbt_daily_retrain_263fee` | existing Daily Retrain |

## 4. Diagnostics

### 4.1 Feature builder and preprocessing

Fixed builders contained 97/98/98/99/99/99 features in 2015–2025. Dynamic
builders stayed in the same range (97–99), with **0 missing fixed-schema
features** and at most one extra dynamic feature. The difference is therefore
not a lost-feature artifact; it is median/imputation and feature-selection
drift.

| Fold | Fixed features | Dynamic features | Max dynamic extras | Max missing fixed | Mean median shift | Max median shift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2015 | 97 | 97–98 | 1 | 0 | 0.0872 | 1.544 |
| 2017 | 98 | 98 | 0 | 0 | 0.1056 | 3.275 |
| 2019 | 98 | 98–99 | 1 | 0 | 0.0625 | 2.879 |
| 2021 | 99 | 99 | 0 | 0 | 0.1104 | 2.860 |
| 2023 | 99 | 99 | 0 | 0 | 0.0954 | 3.012 |
| 2025 | 99 | 99 | 0 | 0 | 0.0507 | 0.667 |

`feature_medians_initial.csv` stores every fold-start median;
`feature_stats.csv` stores the daily dynamic counts and median shifts.

### 4.2 Raw predictions and calibration

Overall OOF prediction statistics (`prediction_stats.csv`, fold `all`):

| Arm | Mean | Std | Positive share | Pearson IC | Rank IC |
| --- | ---: | ---: | ---: | ---: | ---: |
| A fixed/fixed | -0.001177 | 0.004125 | 0.349 | 0.0244 | 0.0543 |
| B fixed/rolling | -0.000126 | 0.003073 | 0.500 | 0.0472 | 0.0243 |
| C dynamic/fixed | -0.001602 | 0.004213 | 0.288 | 0.0156 | 0.0346 |
| D dynamic/rolling | -0.000137 | 0.003128 | 0.510 | 0.0103 | -0.0004 |

C has the most negative mean prediction and the lowest positive share (28.8%),
which explains its low exposure. `prediction_bins.csv` contains five prediction
quantiles per fold and overall.

### 4.3 Prediction and allocation correlations

Pearson correlation across concatenated OOF:

| | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| A raw / allocation | 1 / 1 | 0.086 / 0.188 | 0.865 / 0.727 | 0.068 / 0.179 |
| B raw / allocation | 0.086 / 0.188 | 1 / 1 | 0.045 / 0.161 | 0.897 / 0.801 |
| C raw / allocation | 0.865 / 0.727 | 0.045 / 0.161 | 1 / 1 | 0.040 / 0.163 |
| D raw / allocation | 0.068 / 0.179 | 0.897 / 0.801 | 0.040 / 0.163 | 1 / 1 |

The fixed model is stable under builder reconstruction (A–C raw 0.865), while
the rolling model is stable across builder reconstruction (B–D raw 0.897). The
two model regimes are almost orthogonal (A–B 0.086). Nevertheless, a small
raw-prediction difference is amplified by clipping and smoothing into a much
different allocation path: A–C allocation correlation is 0.727 and B–D is
0.801.

`prediction_correlations.csv` also reports Pearson and Spearman correlations
against the existing Online and Daily outputs for every fold.

### 4.4 Allocation, exposure and clipping

Overall (`allocation_stats.csv`):

| Arm | Mean allocation | Zero share | Two share | Turnover |
| --- | ---: | ---: | ---: | ---: |
| A fixed/fixed | 0.495 | 18.5% | 0% | 0.304 |
| B fixed/rolling | 0.652 | 1.0% | 0% | 0.514 |
| C dynamic/fixed | 0.402 | 25.0% | 0% | 0.221 |
| D dynamic/rolling | 0.659 | 0.5% | 0% | 0.520 |

C is the low-exposure arm: one quarter of its days are at zero, versus 1.0% for
B. The dynamic builder with a fixed model is therefore not just a different
prediction; it changes the frequency of the `signal <= 0` clipping event.

### 4.5 Runtime and computational cost

| Arm | Wall time | Model fits | Model updates | Builder refits |
| --- | ---: | ---: | ---: | ---: |
| A fixed/fixed | 145.9 s | 6 | 0 | 6 |
| B fixed/rolling | 3,384.4 s | 6 | 1,482 | 6 |
| C dynamic/fixed | 176.8 s | 6 | 0 | 1,488 |
| D dynamic/rolling | 2,740.6 s | 1,488 | 0 | 1,488 |

B and D are much more expensive than A/C because they refit the ensemble every
day. A/B/D wall times are reconstructed from manifest creation to OOF write and
include setup/logging; C is measured by the Phase 4C script. The detailed table
is `runtime_costs.csv`. The extra cost does not buy a robust score improvement.

## 5. Performance and statistical comparison

### 5.1 Six-fold metrics

All rows use the same folds and labels. **Concatenated** means the six fold
return series are joined in chronological fold order and compounded from the
same starting capital. This is a synthetic OOF diagnostic, not a continuous
calendar track record: the gaps between test folds are omitted and no capital
is assumed invested in those gaps.

| Arm | Official score (fold mean) | Official score (concatenated) | Raw Sharpe (fold mean) | Raw Sharpe (concatenated) | Geometric cumulative return | Max drawdown | Turnover | Mean allocation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A fixed/fixed | 0.833 | 0.419 | 0.992 | 0.629 | 109.5% | -31.8% | 0.304 | 0.495 |
| B fixed/rolling | 1.253 | 1.271 | 1.435 | 1.277 | 191.7% | -20.1% | 0.514 | 0.652 |
| C dynamic/fixed | 0.524 | 0.329 | 1.266 | 0.551 | 87.0% | -32.0% | 0.221 | 0.402 |
| D dynamic/rolling | 0.871 | 0.808 | 1.114 | 0.966 | 134.1% | -19.3% | 0.520 | 0.659 |
| Buy & Hold | 1.498 | 1.280 | 1.498 | 1.280 | 203.9% | -18.0% | 0.001 | 1.000 |
| Constant 1.25 | 1.411 | 1.204 | 1.481 | 1.264 | 283.3% | -22.3% | 0.001 | 1.250 |
| Vol-only 1.0x | 1.267 | 1.151 | 1.486 | 1.329 | 354.9% | -20.7% | 0.066 | 1.200 |

Turnover-adjusted performance adds an explicit **10 bps linear cost per unit of
|Δposition|** on top of the notebook policy. This is a stress diagnostic, not a
replacement for the official score.

| Arm | Net Sharpe @10 bps | Net cumulative return @10 bps |
| --- | ---: | ---: |
| A fixed/fixed | -0.082 | -15.1% |
| B fixed/rolling | -0.550 | -36.8% |
| C dynamic/fixed | 0.044 | -3.1% |
| D dynamic/rolling | -0.846 | -50.2% |
| Buy & Hold | 1.495 | 203.2% |
| Constant 1.25 | 1.465 | 282.3% |
| Vol-only 1.0x | 1.344 | 274.2% |

Fold-level official scores:

| Fold | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| 2015 | 0.226 | 0.460 | 0.312 | 0.348 |
| 2017 | 0.316 | 0.521 | 0.583 | 0.572 |
| 2019 | 1.979 | 2.588 | 0.776 | 2.368 |
| 2021 | 2.360 | 1.205 | 0.653 | 0.386 |
| 2023 | 0.166 | 2.851 | 0.816 | 1.653 |
| 2025 | -0.049 | -0.107 | 0.003 | -0.099 |

### 5.2 Factorial contrasts

Point estimates of the main effects (fold-mean official score):

- fixed builder mean = (0.833 + 1.253) / 2 = **1.043**;
- dynamic builder mean = (0.524 + 0.871) / 2 = **0.697**;
- fixed model mean = (0.833 + 0.524) / 2 = **0.678**;
- rolling model mean = (1.253 + 0.871) / 2 = **1.062**.

Direct paired block-bootstrap contrasts (block = 20, 2,000 reps):

| Contrast | Estimate | 95% CI | p (unadjusted) | p (Holm) |
| --- | ---: | ---: | ---: | ---: |
| C − A: dynamic builder, fixed model | -0.310 | [-0.659, 0.398] | 0.721 | 1.000 |
| D − B: dynamic builder, rolling model | -0.382 | [-0.643, 0.009] | 0.055 | 0.220 |
| B − A: rolling model, fixed builder | +0.420 | [-0.276, 1.183] | 0.219 | 0.657 |
| D − C: rolling model, dynamic builder | +0.347 | [-0.502, 1.011] | 0.518 | 1.000 |
| Interaction `(D−C) − (B−A)` | **-0.072** | point estimate only | — | — |

The point estimates point to **dynamic builder reconstruction as the negative
driver** and **rolling retraining as a positive driver**, but none of the
individual contrasts is statistically distinguishable from zero after Holm
correction. The D−B contrast is the closest (unadjusted p = 0.055, upper CI
0.009), but its Holm p is 0.220. The experiment therefore supports the
hypothesis only as a **descriptive mechanism**, not as an established causal
finding.

### 5.3 Comparisons with the requested baselines

Key paired differences against the existing Online and Daily OOFs (full 20-row
table in `bootstrap.csv`):

| Config | vs Existing Online | 95% CI | p | vs Existing Daily | 95% CI | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | -0.420 | [-1.183, 0.276] | 0.219 | -0.038 | [-0.922, 0.596] | 0.707 |
| B | 0.000 | [0, 0] | 1.000 | +0.382 | [-0.009, 0.643] | 0.055 |
| C | -0.729 | [-1.297, 0.194] | 0.145 | -0.347 | [-1.011, 0.502] | 0.518 |
| D | -0.382 | [-0.643, 0.009] | 0.055 | 0.000 | [0, 0] | 1.000 |

All four configurations also fail to beat Buy & Hold, Constant 1.25 and
Vol-only: every fold-mean CI crosses zero, and every Holm-adjusted p-value is
1.0. The full table includes both the fold-mean and the synthetic concatenated
bootstrap; the fold-mean is the primary result.

A concrete warning about concatenation: A vs Buy & Hold has a concatenated 95%
CI of [-1.525, -0.080] (unadjusted p=0.024), while its fold-mean CI is
[-1.452, 0.133] (p=0.107). A single compounded path can look significant even
when the equal-weight six-fold result does not. Do not report the concatenated
return as an investable track record.

## 6. Decision gate

- **No configuration provides robust incremental value.** B is the best point
  estimate (1.253), but it is the existing Online OOF and its CIs against Buy &
  Hold, Constant 1.25 and Vol-only all cross zero.
- **Feature-builder reconstruction is the leading descriptive explanation for
  the B→D gap** (D−B -0.382; dynamic builder main effect approximately -0.346),
  with rolling retraining opposing it (+0.347 to +0.420). The interaction is
  small (-0.072). This is not statistically established because the six-fold
  intervals are wide.
- **The Online advantage can be reproduced only by freezing the feature builder
  and rolling the model** (B). The current Daily Retrain (D) does not reproduce
  it; C shows that rebuilding the builder with a frozen model is worse than
  either. A deployable version of B would need a frozen preprocessing reference
  and a daily model update.
- **Stop further GBDT optimization.** No arm shows a reliable gain over the
  simple baselines. Do not implement Deep Learning in this phase.
- **Common evaluation interface is ready** for GRU/TCN: same six folds, same
  labels, `paired_block_bootstrap` for fold-aware paired CIs,
  `summarize_turnover_adjusted` for turnover-cost stress, and the same
  fold-mean/concatenated distinction.

## 7. Limitations

- A/B/D are reused existing OOFs; only C was newly trained. Their wall times are
  reconstructed from manifest creation to OOF write and include setup/logging;
  only C has exact in-process timing.
- C aligns dynamic features to the fixed model's schema. It records zero missing
  fixed features, but extra dynamic features are ignored. A model that could
  consume a changing schema is a different experiment.
- Six fold-level observations give wide paired intervals; Holm correction makes
  every comparison non-significant. Absence of significance is not proof of
  equality.
- The concatenated return is a synthetic diagnostic, not a continuous investable
  series. Fold gaps are omitted and policy state is reset at fold boundaries.
- The E14 outage is post-competition and is not part of these six folds; its
  separate proxy analysis remains in `docs/deployment_eval.md`.
- The 10 bps turnover cost is an additional stress assumption, not the official
  competition scoring rule.

## 8. Artifacts

Scripts and configuration:

- `scripts/run_phase4_dynamic_fixed.py`: new C OOF.
- `scripts/run_feature_builder_attribution.py`: A/B/C/D assembly, diagnostics,
  bootstrap and plots.
- `configs/gdbt/phase4.yaml`: block length, bootstrap reps, cost stress and
  primary baselines.
- `common/evaluation.py`: `paired_block_bootstrap`, `holm_adjust`,
  `summarize_turnover_adjusted` (shared with future GRU/TCN experiments).

Results:

- `reports/phase4_attribution/summary.csv`
- `reports/phase4_attribution/fold_metrics.csv`
- `reports/phase4_attribution/factorial_effects.csv`
- `reports/phase4_attribution/bootstrap.csv`
- `reports/phase4_attribution/prediction_stats.csv`
- `reports/phase4_attribution/prediction_bins.csv`
- `reports/phase4_attribution/prediction_correlations.csv`
- `reports/phase4_attribution/pairwise_correlations.csv`
- `reports/phase4_attribution/allocation_stats.csv`
- `reports/phase4_attribution/feature_stats.csv`
- `reports/phase4_attribution/feature_medians_initial.csv`
- `reports/phase4_attribution/runtime_costs.csv`
- `reports/phase4_attribution/plots/01..09`
- `reports/phase4_attribution/oof/` and `manifest.json` (immutable copies and
  SHA-256 hashes).

Tests: `pytest -q` must remain green; the common evaluation tests cover the
bootstrap, Holm adjustment and turnover-cost summary.
