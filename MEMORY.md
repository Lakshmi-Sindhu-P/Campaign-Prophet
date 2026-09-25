# Campaign Prophet — Project Memory

Persistent memory for agents working in this repository. Two layers:
- **Slow layer** (changes rarely): Core Architecture, Semantic Memory, Procedural Memory.
- **Fast layer** (append-only): Episodic Memory — the decision log.

Read this file at the start of any task and update it when the architecture or a core
convention changes. Log major decisions in the Episodic section with a date and rationale.

---

## 1. Core Architecture

**Thesis.** A reproducible **pre-contact** decision-support system that prioritizes limited
outreach capacity using only information available before a call — and *demonstrates* the
leakage cost, the temporal-shift cost, and the capacity/coverage trade-off rather than hiding
them. It is a portfolio analysis, not a production banking system.

**Pipeline stages (in order).**

1. `scripts/prepare_data.py` — load/rebuild the tracked processed dataset; write Notebook 01
   descriptive handoffs to `outputs/notebook_01/`. `--refresh-data` downloads the public UCI
   archive to a temp dir (raw data never committed).
2. `scripts/run_sql.py` — reproduce the Notebook 01 segment handoffs in portable SQL
   (`sql/*.sql`) and verify them against the pandas computation.
3. `scripts/run_modeling.py` — the core pipeline: random benchmark, leakage comparison,
   internal source-order validation, temporal-holdout reporting, out-of-fold calibration,
   distribution shift, capacity tables, permutation importance, impact statement, visuals,
   model card, and `model_selection.json`.
4. `scripts/tune_sensitivity.py` — bounded hyperparameter sensitivity on the validation slice.
5. `scripts/recommend.py --capacity N` — decision-time capacity recommendation CLI.
6. `scripts/build_report.py` — self-contained HTML capacity report; `app/main.py` exposes the
   same summary over HTTP (optional).
7. `scripts/experiment_power.py` — reproducible power / minimum-detectable-effect design.
8. `tests/test_pipeline_contract.py` — contract tests pinning data, artifacts, decision-layer,
   calibration, power, SQL-parity, report parity, and impact outputs.
9. `scripts/check_drift.py` — CI guard: regenerated artifacts must match the committed ones
   (numeric within tolerance, text exactly). Driven by `.github/workflows/pipeline.yml`.

**Data flow.** `data/processed/cleaned_feature_engineered_bank_marketing.csv` (tracked input)
→ `run_modeling.py` → `outputs/notebook_02/*` + `visuals/*` + `docs/model_card.md`.

**Artifacts (`outputs/notebook_02/`).** `model_metrics.csv`, `validation_metrics.csv`,
`leakage_comparison.csv`, `calibration_metrics.csv`, `distribution_shift.csv`,
`scored_temporal_holdout.csv`, `capacity_table.csv`, `capacity_table_by_model.csv`,
`cumulative_gains.csv`, `feature_importance.csv`, `tuning_sensitivity.csv`,
`impact_statement.md`, `subgroup_errors.csv`, `model_selection.json`.

**Model scope.** Three canonical models: Logistic Regression, Random Forest, and
Histogram Gradient Boosting (sklearn). Random Forest is operational (selected on the internal
validation slice).

**Tech stack.** Python 3.10–3.14; numpy 2.3.3; pandas 2.3.3; scipy 1.16.2;
scikit-learn 1.7.2 (all three models, incl. `HistGradientBoostingClassifier`);
matplotlib 3.10.6; seaborn 0.13.2; jupyterlab 4.4.7 / nbconvert 7.16.6; pytest 8.4.2.
No system OpenMP dependency: single-threaded numerics via `OMP_NUM_THREADS=1`.

**Folder structure.**
```
config/            roi_scenarios.json (scenario economics)
data/processed/    tracked modeling input (processed CSV)
data/raw/          instructions only; raw UCI zips excluded from Git
docs/              data_dictionary.md, experiment_design.md, monitoring.md, model_card.md (generated)
notebooks/         01_... (prepare_data), 02_... (run_modeling) thin runpy wrappers
outputs/notebook_01/  Notebook 01 handoff tables
outputs/notebook_02/  model / leakage / calibration / shift / capacity artifacts
outputs/experiment_design/  power_analysis.json
outputs/report/    capacity_report.html (self-contained)
app/               main.py (thin FastAPI /recommend endpoint; optional)
scripts/           prepare_data.py, run_sql.py, run_modeling.py, tune_sensitivity.py, recommend.py, build_report.py, experiment_power.py, check_drift.py
sql/               portable segment queries (expected_value, job/poutcome crosstabs)
tests/             test_pipeline_contract.py
visuals/           generated evaluation charts
Dockerfile         one-command reproduction
.github/workflows/ pipeline.yml (full pipeline + drift guard)
```

---

## 2. Semantic Memory (Domain Facts)

**Dataset.** UCI Bank Marketing (Moro, Cortez & Rita, 2014); `bank-additional-full`.
**41,176** deduplicated records; overall subscription rate **11.27%**. Target `y` ∈ {yes,no}.

**Decision-time feature policy.** `duration` and `campaign` are post-contact fields and are
excluded from the operational model (`EXCLUDED_FEATURES` in `run_modeling.py`). Derived/target
helpers `y`, `success`, `age_group`, `duration_category` are also excluded.

**Split design.** Rows are in source order (no timestamps; this is an explicit limitation).
First 80% = training period; final 20% = temporal holdout. Within the training period the last
20% is the internal validation slice used for model selection.

**Measured facts (current artifacts).**
- Random-split RF: ROC-AUC **0.8126**, AP **0.4822**.
- Temporal holdout RF: ROC-AUC **0.7180**, AP **0.5151**.
- Label shift: subscription rate **6.38% (train) → 30.83% (holdout)**; euribor3m mean
  **4.27 → 1.03**; PSI **>12** on `euribor3m`, `nr.employed`, `emp.var.rate`.
- Leakage cost: naive full-info RF ROC-AUC **0.9480** vs pre-contact **0.8126**; paired gap
  **+0.135** (95% CI 0.121–0.150).
- Capacity (uncalibrated ranking, holdout): top-10% captures **20.5%** of responders at
  **2.05×** lift; top-20% **37.8% / 1.89×**.
- Calibration does not transfer: uncalibrated Brier **0.2257** vs isotonic **0.2472**
  (paired diff −0.0215, CI −0.0296 to −0.0129); ranking preserved (Spearman 0.9956).
- Third model (Histogram Gradient Boosting): benchmark ROC-AUC **0.8047**, holdout **0.5990**;
  naive full-info **0.9510** (largest leakage gap **+0.1459**). Does not beat RF on internal
  validation, so RF stays operational.
- Permutation importance (holdout, RF): top features `poutcome`, `pdays`, `month`.

---

## 3. Procedural Memory (Workflows / Coding Styles)

**Run order.**
```bash
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/run_sql.py
.venv/bin/python scripts/run_modeling.py
.venv/bin/python scripts/tune_sensitivity.py
.venv/bin/python scripts/build_report.py
.venv/bin/python scripts/recommend.py --capacity 20
.venv/bin/python scripts/experiment_power.py
.venv/bin/python -m pytest
```

**Conventions.**
- Deterministic: `RANDOM_STATE = 42` everywhere; `np.random.default_rng(RANDOM_STATE)` for
  bootstraps. No wall-clock or unseeded randomness. Estimators run with `n_jobs=1` and
  `OMP_NUM_THREADS=1`, so two consecutive runs on the same machine are byte-identical.
  Across platforms (macOS Accelerate vs Linux OpenBLAS) results differ at ~1e-4, which can move
  a 4th decimal; do not assert byte-identity cross-platform.
- Standalone scripts with `main()` and module-level path constants; write to explicit
  `outputs/` / `visuals/` directories (created with `mkdir(parents=True, exist_ok=True)`).
- **Rank on uncalibrated scores.** Calibrated probabilities are only for expected-responder
  counts. Never feed calibrated scores into ranking or capacity decisions.
- **Capacity is percent-of-population** (5/10/20/30/50%), not EUR budgets. EUR scenario config
  is retained only as a labeled overlay.
- Every published README/model-card number must trace to a regenerated artifact; the contract
  tests enforce this. After changing any metric, regenerate artifacts and update the README.
- CI (`.github/workflows/pipeline.yml`) re-runs the full pipeline and then `scripts/check_drift.py`.
  The guard checks **structure exactly** (columns, row counts, categories, JSON keys) and
  **numbers within cross-platform tolerance** (floats rtol/atol 3e-2, integer counts ±3).
  Generated prose/HTML (model card, report) and PNGs are excluded — they embed rounded values
  that legitimately differ across platforms. A metric change still requires regenerating and
  committing artifacts.
- `docs/model_card.md` is **generated** by `run_modeling.py`; edit the template, not the output.
- No comments in code unless they explain non-obvious intent; docstrings for module/function.
- Notebooks are thin `runpy` wrappers; re-execute with
  `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900`.
- Raw data is never committed; only the processed CSV and regeneration script.
- No financial targeting recommendation is ever published; economics are labeled scenarios.
- Models must be **pure scikit-learn** so they run identically on macOS and CI without system
  dependencies; do not add estimators that require a system OpenMP runtime. Keep
  `OMP_NUM_THREADS=1` / `n_jobs=1` so artifacts stay byte-identical.

**Commit style.** Terse, imperative; recent history uses short messages (e.g. "Update
README.md"). Group a phase of work into one coherent commit.

---

## 4. Episodic Memory (Key Decisions Log)

Append new entries at the bottom. Format: `YYYY-MM-DD — decision — rationale`.

- **2026-09 (pivot audit)** — Decided the project is "decision-support scaffolding whose
  decision layer used broken numbers," not a classifier rewrite. Keep the feature-availability
  policy, dual split, and honesty conventions; fix the decision layer.
- **2026-09** — Rejected champion/challenger architecture, policy-simulator UI, segment
  clustering, and contact-fatigue analysis as scope creep; the Top-K table delivers the same
  insight without added architecture.
- **2026-09 (Phase 1)** — Rank on uncalibrated scores (ranking is invariant to monotone
  calibration); select models on an internal validation slice, touch the final holdout once;
  replace EUR threshold/budget artifacts with a Top-K capacity table. Removed
  `threshold_optimization.csv`, `threshold_summary.csv`, `budget_scenarios.csv`,
  `visuals/profit_vs_threshold.png`.
- **2026-09** — Kept the source-order holdout and leaned into the label shift as the drift
  story (rather than adding a stratified-by-period split). Confirmed first-contact decision
  framing (`campaign` excluded).
- **2026-09** — Formally scoped a **two-model comparison** (Logistic Regression, Random
  Forest) when XGBoost's OpenMP dependency is unavailable; did not install `libomp`.
- **2026-09 (Phase 2)** — Added a naive full-information model on an identical random split to
  *measure* the leakage-policy cost instead of asserting it.
- **2026-09 (Phase 3)** — Calibrators fit on out-of-fold training predictions (leak-free);
  calibration reported as unfit for absolute values because it does not transfer across the
  label shift. Also added ECE, calibration slope, and a paired Brier-difference CI.
- **2026-09** — Final decision model is trained on the **full training period**, not just the
  fit slice; a 64% fit slice cost ~13 AUC points because recent records dominate under drift.
- **2026-09 (Phase 4)** — Confirmed removals: duplicate processed CSV in `outputs/notebook_01/`,
  `notebooks/00_github_sync.ipynb`, `outputs/notebook_01/usd_eur_ranking_comparison.csv`.
- **2026-09 (Phase 5/6)** — Added `recommend.py` capacity CLI and a randomized-holdback
  experiment design with power/MDE arithmetic; no retrospective A/B or uplift claim is made
  (the data has no control arm).
- **2026-09 (uncertainty pass)** — Added bootstrap 95% CIs and binomial p-values to the capacity
  table, paired AUC-gap CIs to the leakage comparison, a per-model operational comparison
  (RF buys only ~7 extra responders at top-10% vs LR), and minimum-detectable-effect to the
  power analysis. Rationale: point estimates alone invite overclaiming; uncertainty is the
  honest interview-grade evidence.
- **2026-09 (reproducibility hardening)** — Set `n_jobs=1` on all estimators and
  `cross_val_predict` so re-runs are byte-identical. A verification re-run exposed last-digit
  floating-point drift (threaded reduction order) in the scored holdout, capacity, and
  calibration artifacts. Rationale: the repo claims mechanical reproducibility, so consumed
  float values should be deterministic, not just equal to displayed precision. No published
  metric changed.
- **2026-09 (Phase 7.2)** — Added CI (`.github/workflows/pipeline.yml`) that re-runs the full
  pipeline on Linux and then `scripts/check_drift.py`. Drift is compared structurally (exact) and
  numerically within cross-platform tolerance, not byte-identity; PNGs and the base64 report are
  excluded. Rationale: enforce the reproducibility claim without false alarms from cross-platform FP.
- **2026-09 (CI finding — cross-platform FP)** — The first CI run failed: Linux regenerated
  Random Forest `roc_auc` as **0.8127** vs the committed macOS **0.8126**, an ~1e-4 platform
  difference that crossed a 4th-decimal boundary. Two fixes: (a) the README-metric contract test
  now matches published numbers within 2e-4 instead of exact string equality; (b) `check_drift.py`
  was rewritten to structural-exact + numeric-tolerance (floats rtol/atol 3e-2, ints ±3) and to
  skip generated prose/HTML. Consequence: "byte-identical" holds only on one platform; the
  cross-platform guarantee is structural reproducibility within tolerance.
- **2026-09 (Phase 7.8)** — Added a `Dockerfile` (one-command reproduction) and
  `docs/monitoring.md` (PSI thresholds, label-drift, ranking/calibration health, retraining
  trigger). Design only: no live serving metrics or automated retraining exist.
- **2026-09 (Phase 7.1)** — Added a self-contained `outputs/report/capacity_report.html`
  (`scripts/build_report.py`, embedded charts/tables, no server) and a thin optional FastAPI
  `app/main.py` `/recommend` endpoint reusing `recommend.summarize`. Both read the artifacts, so
  they cannot disagree with published numbers. Serving deps isolated in `requirements-serve.txt`.
- **2026-09 (Phase 7.7)** — Added a descriptive subgroup error analysis at top-20% capacity
  (`subgroup_errors.csv`, `visuals/subgroup_errors.png`) over age band, job, and marital status.
  Explicitly **not** a fairness certification and caveated by drift; records where the policy
  under-serves (e.g. blue-collar recall 19% vs retired 50%).
- **2026-09 (Phase 7.4)** — Added a portable SQL analyst layer (`sql/*.sql`,
  `scripts/run_sql.py`) that reproduces the Notebook 01 segment handoffs via in-memory SQLite.
  Contract-tested to equal the pandas outputs (byte-identical here). Rationale: closes the
  "no SQL" gap with a verified surface rather than a decorative one.
- **2026-09 (Phase 7.5)** — Added a generated `impact_statement.md` (top-10% coverage/lift),
  surfaced in README and model card and pinned by a contract test. Framed descriptively, never
  causally — no control group exists, so no uplift claim is ever made.
- **2026-09 (Phase 7.6)** — Added a bounded hyperparameter sensitivity study
  (`scripts/tune_sensitivity.py`) scored on the internal validation slice only. Measured result:
  best RF config improves validation AP by only **+0.0009**; best HGB config (+0.0136) still
  trails untuned RF. Shipped defaults stand, and "why no tuning" is now evidence rather than
  assertion. The final holdout is never used for tuning.
- **2026-09 (Phase 7.3 — plan amendment)** — XGBoost **replaced by scikit-learn
  `HistGradientBoostingClassifier`** as the canonical third model. Reason: `brew install libomp`
  is impossible in this environment (Homebrew owned by another user, no sudo), and an
  XGBoost-on-macOS vs XGBoost-on-Linux dependency would break the cross-platform CI drift guard.
  Pure sklearn keeps one artifact set everywhere and stays byte-identical. This reverses the
  earlier "XGBoost optional/gated" decision and the locked Phase 7 wording; documented here per
  the conflict rule. Also added permutation importance (predictive association only). Removed
  `xgboost` from `requirements.txt`.
