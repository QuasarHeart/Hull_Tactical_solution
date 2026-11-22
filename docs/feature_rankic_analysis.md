# Phase 8 — Feature-level RankIC Analysis

Date: 2026-11-22  
Repository: `/home/yys/Hull_Tactical_solution`  
Script: `scripts/run_phase8_feature_rankic.py`  
Config: `configs/gdbt/phase8.yaml`  
Input: six-fold OOF dates from
`models/20261121_170606_gdbt_notebook_cv_offline_ae2a75/oof_predictions.csv`

> This phase is diagnostic only. No model was trained, no OOF file was modified,
> no feature was selected, and no allocation/multiplier/smoothing parameter was
> changed.

## 1. Research Question

Does an individual feature have a directional, time-series association with
future returns, and is that association stable across OOF folds, rolling
windows and volatility regimes? How redundant are the model-input features?
Do any associations survive multiple-testing correction?

The output is a description of feature-level association, **not** evidence of
independent predictive value, tradable alpha, or a reason to add/remove a
feature.

## 2. Data & Methodology

### 2.1 Dates and labels

The six OOF test blocks are exactly the existing CV blocks:

| Fold | Test days | Test date_ids | Train days |
| --- | ---: | --- | ---: |
| 2015 | 252 | 6301–6552 | 800 |
| 2017 | 251 | 6805–7055 | 800 |
| 2019 | 252 | 7307–7558 | 800 |
| 2021 | 252 | 7812–8063 | 800 |
| 2023 | 250 | 8315–8564 | 800 |
| 2025 | 231 | 8817–9047 | 800 |

Total: **1,488 OOF dates**. The OOF file is read only; its SHA-256 is recorded
in the manifest.

`target_metric = forward_returns - risk_free_rate` is used as the future target.
The model training target is `target` (market forward excess return), but
`target_metric` is the label used by the OOF evaluation and is the relevant
future return for this analysis. This also means the rank association is not
the exact training objective.

### 2.2 Feature representations

Three related representations are reported, without treating them as
independent tests:

| Representation | Count | Definition |
| --- | ---: | --- |
| Raw features | 94 | Original D/E/I/M/P/S/V columns in `panel` |
| Derived features | 5 | `U1`, `U2`, `V1_S1`, `M11_V1`, `I9_S1` |
| Model-input features | 99 union | Fold-specific `NotebookFeatureBuilder.features` |

The model-input set is not constant: the notebook's missing-rate filter and
fold-specific training windows select **97, 98, 98, 99, 99, 99** features.
`E7` is unavailable in 2015/2017/2019; `V10` is unavailable in 2015. They remain
in the 99-feature union and are reported as missing, never silently dropped.

### 2.3 Point-in-time feature construction

For each fold:

1. Fit `NotebookFeatureBuilder` only on that fold's 800 training dates.
2. Build each test date **one at a time** with
   `build(panel, [date], with_target=False)`, matching the model's daily
   prediction path.
3. Read `target_metric` for that date only after the feature row is built.

This avoids the notebook builder's I-column backward fill crossing into a later
test date. A truncation check on two folds produced a maximum absolute feature
difference of **0.0**; the manifest records `leakage_check_passed=true`.

Raw values are read directly from the panel and derived values are row-wise
transforms of raw values. No future panel row enters either representation.

### 2.4 RankIC definitions

For one feature \(j\):

```text
IC_j = Spearman(X_j, Y)
```

This is a **time-series RankIC across OOF dates**, not a daily cross-sectional
IC. Separate correlations are computed:

- overall: all valid OOF dates for that feature;
- fold: each fold's valid dates;
- rolling: within each fold, 60- and 120-day trailing windows (no cross-fold
  window);
- regime: low/mid/high volatility using the project's existing causal
  definition: 20-day trailing volatility thresholds from each fold's training
  dates.

The summary reports fold `mean`, `median`, `std`, `ICIR = mean/std`,
`positive_fold_ratio`, and the number of available folds. The overall pooled IC
is kept separately; it is not a substitute for the fold mean.

### 2.5 Inference

- Raw p-value from `scipy.stats.spearmanr` for each feature's time-series IC.
- **Fold-stratified moving-block bootstrap**: block length 20, 2,000
  replications, resampling within each fold before concatenation. This is the
  primary uncertainty diagnostic.
- **Benjamini–Hochberg FDR** separately over the 99 model-input features and
  over the 94 raw features; q<=0.05 is reported.
- No feature is selected or ranked for model use based on these results.

### 2.6 Missing, constant and outlier handling

- `n_valid` / `missing_ratio` include fold-level unavailability.
- Constants and near-zero-variance columns are flagged, not deleted.
- Robust-z outlier counts are flagged, not winsorized or removed.

## 3. Overall Feature RankIC

Top positive and negative features by fold-mean RankIC (model-input
representation):

| Feature | Class | Mean RankIC | Median | Std | ICIR | Positive folds | Overall IC | q_BH | Bootstrap 95% CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E19 | raw | **0.1100** | 0.1166 | 0.0381 | 2.89 | 6/6 | 0.1107 | **0.0019** | [0.0599, 0.1442] |
| M1 | raw | **0.1146** | 0.1331 | 0.0584 | 1.96 | 6/6 | 0.1031 | **0.0027** | [0.0583, 0.1395] |
| V7 | raw | **0.1042** | 0.1215 | 0.0431 | 2.42 | 6/6 | 0.1018 | **0.0027** | [0.0561, 0.1344] |
| V9 | raw | **0.1024** | 0.1108 | 0.0413 | 2.48 | 6/6 | 0.0949 | **0.0061** | [0.0535, 0.1270] |
| V13 | raw | 0.0715 | 0.0677 | 0.0447 | 1.60 | 6/6 | 0.0869 | **0.0157** | [0.0468, 0.1238] |
| V10 | raw | 0.1014 | 0.0850 | 0.0428 | 2.37 | 5/5 available | 0.0839 | 0.0521 | [0.0364, 0.1194] |
| P2 | raw | 0.1182 | 0.1201 | 0.0408 | 2.90 | 6/6 | 0.0189 | 0.8224 | [-0.0174, 0.0664] |
| P11 | raw | **−0.1362** | −0.1206 | 0.0317 | −4.30 | 0/6 | −0.0328 | 0.5599 | [-0.0798, 0.0073] |
| E3 | raw | **−0.1347** | −0.1114 | 0.0449 | −3.00 | 0/6 | −0.0405 | 0.4369 | [-0.0818, 0.0025] |
| E2 | raw | **−0.1340** | −0.1197 | 0.0331 | −4.05 | 0/6 | −0.0347 | 0.5434 | [-0.0779, 0.0102] |
| P10 | raw | −0.1108 | −0.1125 | 0.0349 | −3.18 | 0/6 | −0.0685 | 0.1168 | [-0.1101, -0.0241] |

Five of 99 model-input features pass BH at q<=0.05: **E19, M1, V7, V9, V13**.
`V10` is borderline at q=0.052 in the model-input representation. The raw
representation gives six significant features (the same five plus `V10`,
q=0.0051); this representation sensitivity is itself a reason not to treat the
feature list as a fixed discovery set.

There is an important split between **fold-mean** and **pooled overall** IC:

- P2 has mean fold IC +0.118 and is positive in all six folds, but pooled overall
  IC is only +0.019 and q=0.82.
- P11 has mean fold IC −0.136 and is negative in all six folds, but pooled overall
  IC is −0.033 and q=0.56.
- E2/E3 show the same pattern.

This is not a contradiction to hide: fold composition, nonstationarity and
rank distribution changes matter. The fold-mean and pooled metrics must be
reported together; neither alone is a complete result.

No constant or zero-variance feature was found in the model-input union.
Several features have many robust-z outliers (E4=193, E13=83, E11=58, P6=54,
E12=43, M6=32), but they were retained. `E7` has 733/1,488 observations
(50.7% missing) and is marked `insufficient`; `V10` has 1,236 observations
(16.9% missing) and five available folds.

## 4. OOF Fold Stability

Stability label counts across the 99 model-input features:

| Label | Count | Definition |
| --- | ---: | --- |
| stable_positive | 22 | positive in at least `n_available-1` folds; mean > 0 |
| stable_negative | 20 | negative in at least `n_available-1` folds; mean < 0 |
| single_period | 53 | few strong folds; no stable sign |
| mixed | 3 | direction changes across folds |
| insufficient | 1 | fewer than four available folds (`E7`) |

Selected fold-level ICs:

| Feature | 2015 | 2017 | 2019 | 2021 | 2023 | 2025 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E19 | 0.136 | 0.047 | 0.146 | 0.097 | 0.082 | 0.152 |
| M1 | 0.202 | 0.032 | 0.140 | 0.136 | 0.047 | 0.131 |
| V7 | 0.146 | 0.049 | 0.146 | 0.112 | 0.042 | 0.131 |
| V9 | 0.146 | 0.059 | 0.148 | 0.097 | 0.040 | 0.124 |
| V13 | 0.161 | 0.063 | 0.077 | 0.072 | 0.029 | 0.026 |
| V10 | — | 0.085 | 0.165 | 0.076 | 0.046 | 0.135 |
| P2 | 0.105 | 0.045 | 0.180 | 0.138 | 0.113 | 0.127 |
| P11 | −0.108 | −0.120 | −0.195 | −0.161 | −0.111 | −0.121 |
| E2 | −0.107 | −0.123 | −0.202 | −0.147 | −0.109 | −0.117 |
| E3 | −0.094 | −0.099 | −0.201 | −0.192 | −0.098 | −0.124 |

The five BH-significant positive features are directionally consistent in all
available folds. This is stronger than a single pooled correlation, but it is
still six macro-period samples, not six independent market histories.

The many `single_period` labels are deliberately not promoted into a feature
shortlist. They may be real episode-specific effects, or rank noise; this phase
does not have a pre-registered follow-up test that would distinguish those.

## 5. Rolling Stability

Rolling windows are computed **inside each fold** for 60 and 120 days, so a
window never crosses a fold boundary. The full series are in
`feature_rankic_rolling.parquet`; sign runs are in
`feature_rankic_rolling_runs.csv`.

Examples:

| Feature | 120d mean | 120d positive share | 120d sign flips |
| --- | ---: | ---: | ---: |
| P11 | −0.187 | 0.000 | 0 |
| E2 | −0.187 | 0.000 | 0 |
| E3 | −0.182 | 0.000 | 0 |
| P2 | 0.163 | 1.000 | 0 |
| P10 | −0.147 | 0.016 | 12 |
| M9 | −0.134 | 0.022 | 16 |
| V9 | 0.133 | 0.982 | 1 |
| V7 | 0.129 | 0.979 | 3 |
| V10 | 0.124 | 0.977 | 3 |
| E19 | 0.123 | 1.000 | 0 |

P11/E2/E3 and P2 are almost always on one side of zero in the 120-day rolling
series, but P10/M9 show frequent sign changes. The rolling evidence is
descriptive: no window is selected as "best" and no rolling result is used to
choose a model or feature set.

## 6. Regime Dependence

The project's existing causal regime definition is reused: 20-day trailing
volatility of `target_metric`, with low/mid/high thresholds from each fold's
800-day training window. The total test-day counts are:

| Regime | Test days | Coverage |
| --- | ---: | --- |
| Low volatility | 603 | all six folds |
| Mid volatility | 593 | five folds satisfy n>=20 |
| High volatility | 292 | 2015, 2019, 2021 (7 days), 2025; 2017 and 2023 have no high-vol test days |

Because the high-vol bucket is concentrated in a few folds and 2021 has only
seven high-vol days, **all high-vol cells are marked inconclusive** (only three
folds meet the 20-day minimum). `E7` (one high-vol fold) and `V10` (two) are
especially thin.

Features with the same sign in all three regimes:

- positive: **P2, M1, E7, V10, E19, M11, V9, V7, V13, S5, M14, M13**;
- negative: **E2, E3, P11, P8, P10, M9, P5, M4, S8, M7, P3, S2**.

Examples of regime magnitude:

| Feature | Low vol | Mid vol | High vol |
| --- | ---: | ---: | ---: |
| P2 | 0.135 | 0.062 | 0.173 |
| M1 | 0.115 | 0.093 | 0.150 |
| E19 | 0.090 | 0.089 | 0.099 |
| E2 | −0.180 | −0.068 | −0.169 |
| E3 | −0.175 | −0.062 | −0.177 |
| P11 | −0.146 | −0.072 | −0.178 |

The high-vol magnitudes are often larger for the positive features, but the
sample is 292 days across only a few folds; this is a description of this OOF
sample, not a regime-timing rule. The low/mid/high labels are not re-optimized.

## 7. Feature Redundancy

Pairwise Spearman correlations use model-input values and pairwise-complete
dates. There are **4,851 pairs**; **14 pairs** have |rho| >= 0.90 and form
**10 connected clusters**:

| Cluster | Features |
| --- | --- |
| 1 | **D1, D2** (exact duplicate, rho=1.000) |
| 2 | E16, E17 |
| 3 | E3, P11 |
| 4 | I2, U1 |
| 5 | I5, I9 |
| 6 | S1, V1_S1 |
| 7 | M11, M11_V1 |
| 8 | S10, S11 |
| 9 | V1, V2, V4, V8 |
| 10 | V7, V9, V10 |

Group-level mean absolute correlations (selected):

| Group pair | Mean |rho| | Pairs |
| --- | ---: | ---: |
| derived × E | 0.520 | 5 |
| I × E | 0.426 | 9 |
| V × V | 0.400 | 78 |
| V × E | 0.374 | 13 |
| derived × V | 0.358 | 5 |

The 99 model-input features are therefore not 99 independent observations:
D1/D2 is an exact duplicate, and several volatility/derived clusters are highly
redundant. This is a statement about feature geometry, not a reason to delete
any column; no feature was removed.

## 8. Multiple Testing Results

Model-input features (99 tests, BH FDR):

| Feature | Mean RankIC | raw p | q_BH | Bootstrap 95% CI | Significant |
| --- | ---: | ---: | ---: | --- | --- |
| E19 | 0.1100 | 0.000019 | 0.00185 | [0.0599, 0.1442] | yes |
| M1 | 0.1146 | 0.000068 | 0.00274 | [0.0583, 0.1395] | yes |
| V7 | 0.1042 | 0.000083 | 0.00274 | [0.0561, 0.1344] | yes |
| V9 | 0.1024 | 0.000246 | 0.00610 | [0.0535, 0.1270] | yes |
| V13 | 0.0715 | 0.000791 | 0.01565 | [0.0468, 0.1238] | yes |
| V10 | 0.1014 | 0.003156 | 0.05207 | [0.0364, 0.1194] | borderline |
| P2 | 0.1182 | 0.465178 | 0.822 | [-0.0174, 0.0664] | no |
| P11 | −0.1362 | 0.205774 | 0.560 | [-0.0798, 0.0073] | no |

Raw representation (94 tests) adds `V10` as significant (q=0.0051), so the
model-input vs raw distinction changes one decision. Derived features (5
tests) have no BH-significant result; `M11_V1` has mean +0.078 but q=0.326.

The block-bootstrap confidence intervals for the five model-input significant
features exclude zero in this sample. That is still **not** independent
confirmation: the test is on the same OOF dates used to discover the result,
and the features are not independent of each other.

## 9. Limitations

- **Non-contiguous OOF folds.** The six test years are separated by gaps. A
  pooled 1,488-day rank correlation is not a continuous account; fold means are
  therefore reported beside the pool.
- **Feature-set drift.** Model input has 97/98/98/99/99/99 features across
  folds. `E7` and `V10` are unavailable in early folds; missingness is reported,
  but there is no single fixed 99-column input history.
- **One market observation per day.** This is a time-series RankIC, not a
  cross-sectional breadth test. It cannot tell whether a feature ranks assets
  or contributes conditional information.
- **RankIC is not model contribution.** Correlated features, nonlinear
  interactions, LightGBM/XGBoost interactions, allocation and volatility scaling
  are outside this univariate analysis.
- **Target definition.** `target_metric` is the OOF evaluation label; the model
  trains on `target` (market forward excess return). The difference is small but
  nonzero.
- **Regime imbalance.** High vol has only 292 test days and is concentrated in
  a few folds; 2017 and 2023 have no high-vol test days. Regime results are
  descriptive and often inconclusive.
- **Multiple testing is not a discovery guarantee.** BH and block bootstrap
  reduce false positives but do not create an independent sample. No post-2025
  holdout was used here.
- **Economic value is not tested.** There is no transaction-cost, capacity,
  financing or turnover analysis for a feature-based strategy; this phase does
  not construct one.

## 10. Final Findings / Research Decision

### 10.1 Are there features with relatively stable cross-fold direction?

Yes, descriptively. 22 features are `stable_positive`, 20 are
`stable_negative`, and five positive features survive BH: **E19, M1, V7, V9,
V13**. `V10` is stable but representation-sensitive (model q=0.052, raw
q=0.005). P2 and P11 have strong fold-mean signs but weak pooled IC and are not
BH-significant; they should not be called stable discoveries.

### 10.2 Is there regime dependence?

Yes. The same-sign lists and magnitude table show that some associations are
stronger in high vol, while P11/E2/E3 are negative across all three regimes.
However, high-vol coverage is only 292 days and concentrated in a few folds;
`E7`/`V10` high-vol cells are explicitly inconclusive. No regime-specific
feature rule is established.

### 10.3 How redundant are the features?

Substantially. D1/D2 are exact duplicates; 14 pairs exceed |rho|=0.90; 10
clusters exist, including V1/V2/V4/V8 and V7/V9/V10. The effective feature
geometry is smaller than 99 independent signals. This is diagnostic only; no
column was removed.

### 10.4 Does statistical significance survive multiple testing?

Five of 99 model-input features survive BH at q<=0.05, with block-bootstrap CIs
excluding zero in this sample. Raw representation has six (adding V10); derived
features have none. These are **associations on the discovery OOF**, not
independent validation and not evidence of incremental model value.

### 10.5 What does this help us understand about the model?

The model-input set contains several variables with consistent positive
time-series rank association (M1, E19, V7, V9, V13) and several consistent
negative associations (P11, E3, E2, P10). These are anonymized market variables,
so economic interpretation must remain cautious; the result is better read as
"the model has access to persistent market-state correlates" than as a named
macro factor story.

### 10.6 Is follow-up research needed?

No new model hypothesis is strong enough to justify feature selection, model
retraining, multiplier changes or a new strategy search from this phase. The
correct stopping point is:

- keep all features and the frozen GBDT pipeline;
- treat this RankIC report as a diagnostic map;
- if a future phase wants to test a feature hypothesis, pre-register it and use
  an independent window, not these OOF dates;
- do not promote the five BH-positive features into a new model without such a
  test.

**Phase 8 ends here. No feature selection, no model optimization, no strategy
change.**

## 11. Artifacts and Reproduction

```text
scripts/run_phase8_feature_rankic.py
configs/gdbt/phase8.yaml
reports/feature_rankic_summary.csv
reports/feature_rankic_by_fold.csv
reports/feature_rankic_raw_summary.csv
reports/feature_rankic_raw_by_fold.csv
reports/feature_rankic_derived_summary.csv
reports/feature_rankic_derived_by_fold.csv
reports/feature_rankic_rolling.parquet
reports/feature_rankic_rolling_runs.csv
reports/feature_correlation_matrix.csv
reports/feature_correlation_pairs.csv
reports/feature_correlation_clusters.csv
reports/feature_group_correlation.csv
reports/feature_rankic_regime.csv
reports/feature_rankic_regime_counts.csv
reports/feature_rankic_feature_sets.csv
reports/feature_rankic_manifest.json
reports/feature_rankic_figures/01_top_features_abs_mean_rankic.png
reports/feature_rankic_figures/02_feature_fold_heatmap.png
reports/feature_rankic_figures/03_rolling_rankic_selected.png
reports/feature_rankic_figures/04_regime_rankic.png
reports/feature_rankic_figures/05_feature_correlation_heatmap.png
reports/feature_rankic_figures/06_mean_vs_std_scatter.png
```

Reproduce:

```bash
python scripts/run_phase8_feature_rankic.py \
  --config configs/gdbt/phase8.yaml
```

The manifest records the OOF/panel SHA-256, 1,488 dates, feature counts,
fold feature sets, regime counts, bootstrap settings, FDR alpha and the
point-in-time leakage check.
