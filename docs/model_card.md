# Campaign Prophet model card

## Purpose

Rank customers for term-deposit outreach using information available before a call, so a
limited contact capacity can be spent on the highest-ranked customers. This is a portfolio
analysis, not a production banking decision system.

## Selected model

**Random Forest**, selected by highest Average Precision on an internal
source-order validation slice carved out of the training period. The final temporal holdout
is reported once and never used for model selection. The random stratified split is retained
only as a benchmark.

## Validation and probability use

The first 80% of rows form the training period; the final
20% is held out. Within the training period the last
20% is used for model selection. Source order is treated as
chronological because the dataset represents successive campaigns; this is an explicit dataset
limitation.

Contact decisions are ranked on the model's **uncalibrated** score. Ranking is monotone under
calibration, so calibration cannot improve ordering (Spearman rank correlation with the calibrated
scores on the holdout: 0.9956). Isotonic calibration,
fit on out-of-fold training predictions, is applied only to produce the expected-responder counts in
the capacity table.

| Measure on temporal holdout | Uncalibrated | Isotonic | Sigmoid |
|---|---:|---:|---:|
| ROC-AUC | 0.7180 | 0.7153 | 0.7180 |
| Average Precision | 0.5151 | 0.5015 | 0.5151 |
| Brier score | 0.2257 | 0.2472 | 0.2407 |
| Expected calibration error | 0.1924 | 0.2190 | 0.2070 |
| Calibration slope | 1.579 | 1.349 | 1.319 |

Brier difference (uncalibrated − isotonic), paired bootstrap 95% CI: **-0.0215**
(-0.0296, -0.0129).

## Capacity policy

Capacity is expressed as a percent of the eligible holdout population, not a currency budget.
"Coverage" is the share of all holdout responders reached; "lift" is precision at K over the
holdout base rate. Intervals are paired bootstrap 95% CIs; the p-value is a one-sided binomial
test of precision@K against the holdout base rate.

| Capacity | Contacts | Responders | Coverage % (95% CI) | Precision@K | Lift (95% CI) | p vs base |
|---|---:|---:|---:|---:|---:|---:|
| 5% | 412 | 244 | 9.6% (8.9–10.3) | 0.592 | 1.92× (1.77–2.07) | 1.4e-32 |
| 10% | 824 | 521 | 20.5% (19.5–21.5) | 0.632 | 2.05× (1.95–2.15) | 1.6e-81 |
| 20% | 1,648 | 960 | 37.8% (36.4–39.1) | 0.583 | 1.89× (1.82–1.96) | 1.1e-116 |
| 30% | 2,471 | 1,304 | 51.4% (49.7–53.0) | 0.528 | 1.71× (1.66–1.77) | 2.5e-113 |
| 50% | 4,118 | 1,828 | 72.0% (70.5–73.5) | 0.444 | 1.44× (1.41–1.47) | 1.1e-74 |

These are holdout descriptions, not causal uplift estimates: the dataset has no control group.

## Leakage policy cost

A naive model allowed to use `duration` and `campaign` (post-contact fields) is compared with the
deployable pre-contact model on an identical random split. Random Forest ROC-AUC rises from
0.8126 (pre-contact) to 0.9480 (full information); the paired
bootstrap gap is **+0.1352** (+0.1211, +0.1503). That is the
measured operational cost of the decision-time feature policy. See `leakage_comparison.csv` and
`visuals/leakage_comparison.png`.

## Feature importance (predictive)

Permutation importance on the temporal holdout (drop in average precision) ranks
`poutcome` (0.078), `pdays` (0.045), `month` (0.038), `contact` (0.007), `default` (0.006). This is a predictive-association measure, **not a causal effect**.
See `feature_importance.csv` and `visuals/feature_importance.png`.

## Distribution shift

The training-period subscription rate is 6.4% against a holdout rate of
30.8%. Per-feature population stability indices are published in
`distribution_shift.csv` and `visuals/distribution_shift.png`.

## Impact framing

On the historical holdout, a 10% contact capacity captures **20.5% of
responders** at **2.05× a random contact**. This is descriptive, not causal
uplift: the dataset contains only contacted customers and has no control group.

## Decision status

Isotonic calibration preserves ranking on the source-order holdout but does not improve the Brier
score; the calibrator is fit in the ~6% base-rate training regime and does not transfer to the ~31%
holdout regime. Calibrated probabilities are therefore **unfit for absolute expected-value claims**
and are used only as relative expected-responder counts. Because the decision layer ranks on
uncalibrated scores, this does not contaminate the targeting outputs. The capacity table is a
relative coverage/lift description, not a financial targeting recommendation.

## Limitations

- The dataset has no observed campaign cost, deposit value, or customer lifetime value.
- Source-order temporal validation is a proxy because explicit campaign timestamps are unavailable.
- It predicts response, not incremental causal uplift; there is no untreated control group.
- Any real deployment needs external validation, fairness review, monitoring, and bank-approved economics.
