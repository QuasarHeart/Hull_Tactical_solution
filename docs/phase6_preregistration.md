# Phase 6 pre-registration: signal robustness and regime attribution

Frozen before running the post-2026-07-06 validation.

## Windows

- Official window: **2025-12-16 through 2026-07-06** (137 trading days).
- Independent post-window: **2026-07-07 through 2026-11-17** (95 realized
  trading days; 2026-11-18 through 2026-11-20 targets are still pending).
- No target or feature after the scored date is used for a prediction.

## Strategies frozen for the post-window

| Name | Definition | Role |
| --- | --- | --- |
| Daily Retrain | dynamic builder + full daily refit | primary |
| Fixed builder + rolling | builder frozen at 2026-07-06, model updated daily | secondary |
| Offline | builder/model frozen at 2026-07-06 | secondary |
| Vol-only 1.0x | causal volatility target, same smoothing/cost | control |
| Buy & Hold | constant 1.0 | control |
| Constant 1.25 | constant 1.25 | control |

No new model family, hyperparameter, feature, threshold or allocation rule will
be introduced for the independent window. The post-window is read only after
this protocol and `configs/gdbt/phase6.yaml` are fixed. Its results will not be
used to choose a multiplier, regime threshold, or test family.

## Signal × regime

Volatility regimes are classified with the **pre-official-window** 800-day
causal volatility distribution, using its 33rd and 67th percentiles:

- low volatility: `vol <= q33`
- mid volatility: `q33 < vol < q67`
- high volatility: `vol >= q67`

For each regime report: n, Rank IC, Pearson IC, Q1/Q2/Q3/Q4/Q5 mean
`target_metric`, and Q5−Q1. Quantiles are equal-count within the regime.

## Multiplier grid

The following grid is fixed before looking at outcomes:

```
1, 50, 100, 200, 300, 400, 500, 600, 800, 1000, 1200
```

`1` means no multiplier; `400` and `600` include the fixed-multiplier cases.
The original 600/400 switch is shown separately. For each value report mean
allocation, strategy volatility, official score, raw Sharpe, cumulative return,
maximum drawdown, turnover, and return penalty. The purpose is a robustness
curve, not selecting the best value from the test window.

## Signal concentration

For January 2026 and April 2026:

- daily contribution to the strategy return: `position × target_metric`;
- leave-one-day-out: remove the day and recompute the full-window official
  score and cumulative return;
- block deletion: remove every contiguous block of length 3 and 5 within the
  month and recompute the full-window score/return; report the distribution of
  deltas and the most influential block;
- concentration share: top 1/3/5 day contributions and the number of positive
  days needed to account for 50% of the month's positive contribution.

## Statistical tests

Pre-registered primary family (Holm correction across these six tests):

1. Daily mean strategy excess > 0;
2. Daily vs Buy & Hold mean return difference > 0;
3. Daily vs Constant 1.25 mean return difference > 0;
4. Daily vs Vol-only mean return difference > 0;
5. Rank IC > 0;
6. Q5−Q1 > 0.

Inference:

- Newey-West/HAC standard errors for mean returns and mean differences, with
  **10 lags**;
- moving-block bootstrap with **block length 20**, **2,000 replications** for
  score-difference confidence intervals;
- Holm adjustment across the six tests; report unadjusted and adjusted p-values.

A positive raw Sharpe is not treated as evidence by itself.

## Independent post-window

The same six-test family and the same descriptive diagnostics are computed on
2026-07-07 through 2026-11-17. This is a confirmatory out-of-sample readout, not
a new model-selection round. No post-window result is allowed to alter the
frozen protocol.
