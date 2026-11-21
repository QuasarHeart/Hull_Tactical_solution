# Phase 5: competition-window performance attribution

Date: 2026-11-21  
Window: 2025-12-16 through 2026-07-06 (137 trading days)  
Primary path: Daily Retrain (dynamic builder + full daily refit)  
Official score: `docs/reference/hull-competition-sharpe.ipynb` formula with
`fwd = target_metric`, `rf = 0`; the notebook was executed and reproduced the
original **2.1921191431873206**.

No model was retrained for these ablations. Every experiment starts from the
saved signal, volatility, multiplier and allocation path. The saved allocation
is reproduced exactly by `simulate_positions` (max abs diff 0.0).

## 1. Ablations

| Variant | Official score | Raw Sharpe | Cumulative return | Max drawdown | Turnover | Mean allocation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original signal + allocation | **2.1921** | **2.1921** | **12.21%** | -4.01% | 0.530 | 0.711 |
| Reverse raw signal | 0.0549 | 0.1552 | 1.28% | -11.15% | 0.463 | 0.627 |
| No multiplier (`multiplier=1`) | 0.5565 | 1.8461 | 0.34% | -0.15% | 0.016 | 0.015 |
| Fixed multiplier 600 | 2.1775 | 2.1775 | 13.96% | -4.83% | 0.571 | 0.829 |
| Fixed multiplier 400 | 2.1812 | 2.1812 | 12.13% | -4.01% | 0.529 | 0.706 |
| No smoothing (`smoothing=0`) | 2.0141 | 2.0141 | 12.81% | -4.91% | 0.733 | 0.712 |

Reading:

- **Direction matters.** Reversing the raw prediction leaves long-only
  positions on the opposite days and collapses the official score from 2.192
  to 0.055. It is not a symmetric short strategy because the position floor is
  zero.
- **The multiplier is essential for exposure.** Setting it to 1 produces a
  near-zero-exposure path (mean allocation 0.015), only 0.34% cumulative
  return, and official score 0.556. The high score is not produced by raw
  predictions alone.
- **The 600/400 switch is not the main driver.** Fixed 600 scores 2.178 and
  fixed 400 scores 2.181, both close to the original 2.192. Fixed 600 earns
  more (13.96%) but has a worse drawdown and volatility; fixed 400 is nearly
  indistinguishable from the switch.
- **Smoothing improves risk-adjusted performance.** Removing smoothing raises
  turnover from 0.530 to 0.733, lowers the official score from 2.192 to 2.014,
  and deepens the drawdown (-4.01% to -4.91%), even though it raises cumulative
  return slightly (12.21% to 12.81%). The zero-position share rises to 45.3%
  without smoothing.

## 2. Prediction shuffle

The signal is shuffled while each day's volatility and multiplier are kept,
then the same allocation policy is reapplied. This tests whether the timing of
the signal matters, not whether the allocation machinery works.

| Null | Null mean | Null p05 | Null median | Null p95 | Actual | p (one-sided) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IID shuffle, 2,000 reps | 0.750 | -0.099 | 0.521 | 2.200 | 2.192 | **0.0525** |
| Block shuffle, L=20, 1,000 reps | 0.738 | -0.085 | 0.482 | 2.192 | 2.192 | **0.0519** |

The actual score is at the edge of the null distribution but does **not** cross
the conventional 5% threshold. The point estimate suggests the original signal
timing adds value (actual 2.192 vs null median about 0.5–0.7), but this is not
statistically conclusive with 137 days and the existing signal.

## 3. Prediction bins

Five equal-count bins, low to high raw prediction:

| Bin | n | Mean target_metric | Positive share | Mean allocation |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 28 | -0.001607 | 0.429 | 0.139 |
| 2 | 28 | 0.001605 | 0.571 | 0.149 |
| 3 | 27 | -0.000398 | 0.593 | 0.721 |
| 4 | 27 | 0.002813 | 0.704 | 1.055 |
| 5 | 27 | 0.000968 | 0.444 | 1.530 |

Overall Pearson IC = 0.0848, rank IC = **0.1144**; top-minus-bottom bin
spread = 0.002575. The relation is **not monotonic**: bin 4 is the best, bin 5
falls back. High predictions do not reliably correspond to the highest future
returns in this window.

## 4. Monthly decomposition

| Month | Days | Official score | Raw Sharpe | Cumulative return | Max drawdown | Mean allocation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2025-12 | 11* | 0.722 | 0.891 | 0.27% | -1.42% | 0.905 |
| 2026-01 | 20 | 5.757 | 5.757 | 3.02% | -0.82% | 0.639 |
| 2026-02 | 19 | -0.506 | -0.506 | -0.47% | -2.06% | 0.930 |
| 2026-03 | 22 | 0.418 | 0.418 | 0.37% | -0.99% | 0.487 |
| 2026-04 | 21 | 0.388 | 7.369 | 5.86% | -1.41% | 0.695 |
| 2026-05 | 20 | 0.124 | 2.245 | 1.77% | -2.73% | 0.999 |
| 2026-06 | 21 | 1.230 | 1.230 | 1.17% | -1.91% | 0.503 |
| 2026-07 | 3* | -0.452 | -6.648 | -0.22% | -0.26% | 0.369 |

`*` Small sample; monthly scores are annualized from 3–11 days and are not
stable estimates. April has a high raw Sharpe but a low official score because
the return penalty is large; the reverse can happen in February.

## 5. Volatility-regime decomposition

Regimes are classified causally using thresholds from the 800-day pre-window
causal volatility distribution: low <= 0.0555, high >= 0.2643 (33rd/67th
percentiles).

| Regime | Days | Official score | Raw Sharpe | Cumulative return | Max drawdown | Mean allocation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Low vol | 21 | **4.788** | 4.788 | 2.70% | -1.42% | 0.806 |
| Mid vol | 21 | 0.015 | 0.015 | 0.02% | -2.06% | 0.903 |
| High vol | 95 | **2.440** | 2.440 | 9.23% | -2.73% | 0.647 |

The high-vol regime carries most of the absolute return (9.23% of 12.21%); the
low-vol regime has the best risk-adjusted score. The mid-vol bucket is
essentially flat. The 95-day high-vol bucket is large because the 2026 window
was mostly above the pre-window high threshold.

## 6. What this means

1. The original 137-day score is reproduced exactly, so the reported phase-4/5
   number is not an artifact of a different metric implementation.
2. The model's direction matters, but the **multiplier and smoothing are doing
   a large part of the work**. Without the multiplier, exposure collapses and
   the score falls to 0.556. Without smoothing, the score falls to 2.014.
3. The 600/400 multiplier switch itself is not essential: fixed 600/400 are
   within about 0.015 of the original score.
4. The signal-timing evidence is positive but weak: shuffle p ≈ 0.052, rank IC
   ≈ 0.114, and prediction bins are non-monotonic.
5. The live advantage is concentrated in January, April and the high-vol
   bucket; February is negative and mid-vol is flat. Monthly/regime scores are
   noisy because of short samples.
6. These are still allocation-level diagnostics on one realized 137-day
   window; they do not establish a stable forecasting edge.

## 7. Artifacts

- `scripts/run_phase5_window_attribution.py`
- `configs/gdbt/phase5.yaml`
- `reports/phase5_window_attribution/variant_metrics.csv`
- `reports/phase5_window_attribution/shuffle_scores.csv`
- `reports/phase5_window_attribution/shuffle_summary.json`
- `reports/phase5_window_attribution/signal_bins.csv`
- `reports/phase5_window_attribution/signal_bins_summary.json`
- `reports/phase5_window_attribution/monthly.csv`
- `reports/phase5_window_attribution/regime.csv`
- `reports/phase5_window_attribution/manifest.json`
- `reports/phase5_window_attribution/plots/01..06`

Reproduce:

```bash
python scripts/run_phase5_window_attribution.py
```
