# Monitoring design (how this would be watched in production)

This is a **design**, not an implemented monitor. The historical project regenerates static
artifacts; a live deployment would need the checks below. Thresholds are conventional, not tuned.

## 1. Data / covariate drift

The pipeline already computes the Population Stability Index (PSI) between the training period
and the holdout (`distribution_shift.csv`).

| PSI | Reading | Action |
|---:|---|---|
| < 0.10 | no meaningful shift | none |
| 0.10–0.25 | moderate shift | investigate the feature, re-fit calibrator |
| > 0.25 | major shift | suspect pipeline/upstream change; consider retraining |

In the current artifacts, `euribor3m`, `nr.employed`, and `emp.var.rate` all exceed **12** — the
historical holdout is a different macro regime, so drift alarms would fire by design.

## 2. Label / target drift

Track the observed subscription rate against the training prior (**6.4%** train vs **30.8%**
holdout historically). A sustained gap means the prior has moved and any probability calibration
is invalid — which is exactly why the decision layer ranks on **uncalibrated** scores.

## 3. Ranking and capacity health

- Monitor holdout ROC-AUC / PR-AUC against the reference (temporal holdout RF: 0.7180 / 0.5151).
- Monitor top-K coverage/lift against the reference capacity table; a drop in lift at fixed K is
  the operationally meaningful alert.
- Monitor subgroup recall (`subgroup_errors.csv`) for widening gaps.

## 4. Calibration health

Track Brier score and Expected Calibration Error per period. Isotonic calibration is currently
reported as **unfit** because it does not transfer across the label shift; re-fit and re-validate
on representative data before trusting any absolute probability.

## 5. Retraining trigger

Re-fit when (a) PSI crosses 0.25 on a key feature, (b) top-K lift degrades materially for two
consecutive monitoring windows, or (c) a new label period becomes available. Always re-run the
contract tests and republish the model card.

## 6. What is explicitly out of scope

No live serving metrics, no fairness certification, no causal/uplift monitoring (the data has no
control group), and no automated retraining. Those require production infrastructure and data this
project does not have.
