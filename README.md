# Campaign Prophet

**A reproducible pre-contact decision-support portfolio analysis for retail-bank term-deposit outreach, built on the UCI Bank Marketing dataset.** It ranks customers using only information available before a call, demonstrates the cost of the leakage controls, quantifies temporal drift, and translates ranking quality into a contact-capacity policy — while separating what is demonstrated from what real deployment would need.

## 1. Data validity

- **41,176** deduplicated campaign records; **11.27%** overall historical subscription rate.
- Feature-availability policy: `duration` and `campaign` are post-contact fields and are excluded from the operational model. See the [data dictionary](docs/data_dictionary.md).
- The processed, feature-engineered CSV is tracked; raw ZIPs are deliberately excluded and rebuilt on request by [`scripts/prepare_data.py`](scripts/prepare_data.py).

## 2. The leakage demonstration

A naive model allowed to use `duration` and `campaign` is trained on the *same* random split as the deployable pre-contact model. The gap is the measured cost of the policy:

| Model (random split) | Pre-contact ROC-AUC | Naive full-information ROC-AUC | Paired gap (95% CI) | Top-decile lift pre → naive |
|---|---:|---:|---:|---:|
| Random Forest | 0.8126 | 0.9480 | **+0.1352** (0.121–0.150) | 4.66× → 5.60× |
| Logistic Regression | 0.8009 | 0.9403 | **+0.1390** (0.121–0.155) | 4.46× → 5.43× |

Full metrics and bootstrap intervals in `outputs/notebook_02/leakage_comparison.csv`.

## 3. Propensity models and ranking quality

Random Forest is the model leader, selected by Average Precision on an **internal validation slice of the training period**; the final temporal holdout is reported once and never used for model selection.

| Validation | ROC-AUC | Average Precision | Interpretation |
|---|---:|---:|---|
| Random stratified benchmark | 0.8126 | 0.4822 | Benchmark only; not the deployment estimate. |
| Source-order temporal holdout | 0.7180 | 0.5151 | Forward-looking proxy; source order is an explicit assumption. |

## 4. Calibration under drift

The source-order holdout is a different macro regime: subscription rate **6.4% → 30.8%**, `euribor3m` **4.27 → 1.03**, with PSI above 12 on `euribor3m`, `nr.employed`, and `emp.var.rate`. Calibrators are fit on out-of-fold training predictions and evaluated on the holdout:

| Method | ROC-AUC | Brier | ECE | Calibration slope |
|---|---:|---:|---:|---:|
| Uncalibrated | 0.7180 | **0.2257** | **0.1924** | 1.579 |
| Isotonic | 0.7153 | 0.2472 | 0.2190 | 1.349 |
| Sigmoid (Platt) | 0.7180 | 0.2407 | 0.2070 | 1.319 |

Ranking is preserved (Spearman 0.9956) but calibration **does not transfer** across the label shift: the Brier difference (uncalibrated − isotonic) is **−0.0215 (95% CI −0.0296, −0.0129)**, i.e. calibration is significantly *worse* on the holdout. Calibrated probabilities are therefore **unfit for absolute expected-value claims** and used only as relative expected-responder counts. The decision layer ranks on the **uncalibrated** score. See `outputs/notebook_02/distribution_shift.csv` and `visuals/distribution_shift.png`.

## 5. Capacity policy

Capacity is expressed as a percent of the eligible population, not a currency budget. "Coverage" is the share of holdout responders reached; "lift" is precision at K over the base rate. Ranking is uncalibrated. Intervals are paired bootstrap 95% CIs; the p-value is a one-sided binomial test of precision@K against the holdout base rate. This is a relative coverage/lift description, **not a financial targeting recommendation**.

| Capacity | Contacts | Responders | Coverage (95% CI) | Precision@K | Lift (95% CI) | p vs base |
|---:|---:|---:|---:|---:|---:|---:|
| 5% | 412 | 244 | 9.6% (8.9–10.3) | 0.592 | 1.92× (1.77–2.07) | 1.4e-32 |
| 10% | 824 | 521 | 20.5% (19.5–21.5) | 0.632 | 2.05× (1.95–2.15) | 1.6e-81 |
| 20% | 1,648 | 960 | 37.8% (36.4–39.1) | 0.583 | 1.89× (1.82–1.96) | 1.1e-116 |
| 30% | 2,471 | 1,304 | 51.4% (49.7–53.0) | 0.528 | 1.71× (1.66–1.77) | 2.5e-113 |
| 50% | 4,118 | 1,828 | 72.0% (70.5–73.5) | 0.444 | 1.44× (1.41–1.47) | 1.1e-74 |

**Does the model choice matter operationally?** Almost not at all. At top-10% capacity, Random Forest captures **521** responders (2.05× lift) versus Logistic Regression's **514** (2.02× lift) — a **7-responder / 0.3 pp** difference despite RF's higher ranking AUC. This is why no champion/challenger architecture was added: operational decisions are insensitive to the model choice. See `capacity_table_by_model.csv` and `visuals/capacity_lift_by_model.png`.

## 6. Recommendation and prospective validation

- [`scripts/recommend.py`](scripts/recommend.py) turns a requested capacity into a contact-queue summary:
  `python scripts/recommend.py --capacity 20`.
- The historical dataset contains only contacted customers, so **no retrospective A/B or uplift analysis is possible**. [`docs/experiment_design.md`](docs/experiment_design.md) specifies the randomized-holdback design that would close that gap, with reproducible power arithmetic in [`scripts/experiment_power.py`](scripts/experiment_power.py) (two-proportion z-test, α=0.05, 80% power, 11.27% baseline):
  - detecting a **+2 pp** gain needs ~**4,223 per arm** (~42,230 eligible for a 10% holdback);
  - with only the **8,236** records the holdout can offer, the **minimum detectable effect is +4.74 pp** — i.e. this dataset size cannot prove subtle targeting gains.

## What the project answers

| Question | Evidence |
|---|---|
| Which historic segments converted more often? | Notebook 01 summary tables and scenario calculations. |
| What did the leakage controls cost? | Naive vs pre-contact comparison on an identical split. |
| Does a random split overstate future usefulness? | Random benchmark vs source-order holdout, plus the PSI shift summary. |
| How much responder coverage does a contact capacity buy? | Top-5/10/20/30/50% capacity table (coverage, precision@K, lift). |
| Are calibrated probabilities required for targeting? | No — ranking is invariant to monotone calibration; calibrated probabilities are used only for relative expected-responder counts. |
| How would this be validated in production? | Randomised-holdback experiment design with power analysis (no retrospective claims). |

## Reproducible workflow

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Recreate Notebook 01 handoffs from the tracked processed data.
.venv/bin/python scripts/prepare_data.py

# Optional: rebuild the processed data from the public UCI archive.
.venv/bin/python scripts/prepare_data.py --refresh-data

# Train, validate, calibrate, quantify shift, and create decision artifacts.
.venv/bin/python scripts/run_modeling.py

# Capacity recommendation for a requested contact budget.
.venv/bin/python scripts/recommend.py --capacity 20

# Reproducible experiment sample-size arithmetic.
.venv/bin/python scripts/experiment_power.py

# Verify the data, artifact, and decision-layer contracts.
.venv/bin/python -m pytest
```

On macOS, XGBoost requires the OpenMP runtime. Install it with `brew install libomp` before running the modeling script if you want the XGBoost comparison. Without it the pipeline runs a formally scoped **two-model comparison** (Logistic Regression and Random Forest).

The two notebooks are lightweight, runnable entry points for the same scripts. Run them from the repository root; Google Colab/Drive is no longer required.

## Repository map

```text
config/roi_scenarios.json        Transparent scenario assumptions
data/processed/                  Tracked modeling input
docs/                            Data dictionary, model card, experiment design
notebooks/                       Local-first notebook entry points
outputs/notebook_01/             Recreated analytical handoff tables
outputs/notebook_02/             Model, leakage, calibration, shift, capacity artifacts
outputs/experiment_design/       Reproducible power analysis output
scripts/                         Preparation, modeling, recommendation, power scripts
tests/                           Data, artifact, and decision-layer contract checks
visuals/                         Generated evaluation charts
```

## Limitations

This is a historical-response portfolio analysis, not a deployed banking system. The dataset contains no actual cost, deposit value, customer lifetime value, treatment/control assignment, or complete timestamps. Source order is only a proxy for time; calibrators fit in the training regime do not transfer to the shifted holdout regime; external validation, fairness review, monitoring, and approved economics are required before any deployment.

## License

MIT. See [LICENSE](LICENSE).
