# Prospective validation design (A/B), not results

This document describes **how the pre-contact targeting policy would be validated in
production**. It contains no results and makes no causal claims about the historical data.

## Why there are no historical A/B results

Every row in the UCI Bank Marketing dataset is a customer who *was* contacted. There is no
untreated control group, no randomization, and no assignment mechanism. Comparing contact
frequencies or campaign rounds in the historical data would be confounded by selection bias.
Consequently, no retrospective A/B test and no uplift model can be estimated from this
dataset. The project predicts **response**, not incremental **uplift**.

## Design 1 — Randomized suppression (holdback)

**Goal:** measure the incrementality of the targeting policy — do model-ranked contacts cause
subscriptions that would not have happened otherwise?

1. Rank the full eligible population with the deployable pre-contact model (uncalibrated score).
2. Randomly assign every eligible customer to **contact** (e.g. 90%) or **suppressed control** (10%),
   independent of the model score.
3. Contact only the contacted arm; the suppressed arm receives no outreach this cycle.
4. Estimate the incremental conversion rate as (contacted response rate − control response rate),
   with a two-proportion confidence interval.

The suppression must be random and score-independent or the estimate is confounded. The
control arm defines the counterfactual that the historical data cannot provide.

## Design 2 — Champion/challenger done correctly

Randomize eligible customers between two model-ranked queues (e.g. Logistic Regression vs
Random Forest) rather than comparing them on a historical holdout. This is the only framing
that answers "which model performs better operationally", because it controls for the
contact decision and the time period. Historical-holdout comparisons select on the evaluation
sample and are not valid operational evidence.

## Power and sample size

Reproducible from [`scripts/experiment_power.py`](../scripts/experiment_power.py); output in
`outputs/experiment_design/power_analysis.json`. Two-proportion z-test, two-sided
alpha = 0.05, power = 0.80, baseline conversion rate = **11.27%** (historical overall rate).

| Absolute lift | Relative lift | Sample size per arm | Eligible for a 10% holdback |
|---:|---:|---:|---:|
| +1.0 pp | +8.9% | 16,296 | 162,960 |
| +2.0 pp | +17.7% | 4,223 | 42,230 |
| +3.0 pp | +26.6% | 1,942 | 19,420 |
| +5.0 pp | +44.4% | 745 | 7,450 |

Relative-lift scenarios are also published in the JSON (`+10%`, `+20%`, `+30%` over baseline).

**Minimum detectable effect (inverting the formula):**

| Eligible records | Control arm (10%) | Minimum detectable effect |
|---:|---:|---:|
| 8,236 (this holdout) | 824 | +4.74 pp |
| 42,230 | 4,223 | +2.00 pp |
| 162,960 | 16,296 | +1.00 pp |

Interpretation: detecting a modest +1 pp incrementality gain requires roughly 163k eligible
customers for a 10% holdback — an important reminder that small effects are expensive to
prove. A +2 pp effect needs ~42k eligible customers. The holdout alone (8,236 records) can
only detect effects of **+4.74 pp or larger**, so this dataset size cannot prove subtle
targeting gains; that is precisely why a prospective holdback is required.

## Guardrail metrics

- **Contact fatigue**: complaint/opt-out rate per arm.
- **Suppression harm**: conversion rate in the suppressed arm (should not exceed expectation).
- **Ranking health**: holdout ROC-AUC / PR-AUC monitored for drift; PSI on key features.
- **Calibration drift**: Brier score and reliability, re-fit per period.

## Pre-registered decision rules

1. **Promote the challenger** only if its incremental conversion point estimate exceeds the
   champion's with the pre-registered confidence level, with no guardrail breach.
2. **Stop the policy** if the contact arm is not significantly better than the suppressed
   control (no measured incrementality), or if a guardrail breaches its threshold.
3. **No peeking**: one analysis at the pre-registered sample size (sequential looks inflate
   false positives unless an alpha-spending plan is specified).

## Explicit statement

No causal claim is made anywhere in this repository about the historical UCI data. This
document is a design for future production validation.
