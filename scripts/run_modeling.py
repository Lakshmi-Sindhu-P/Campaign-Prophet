#!/usr/bin/env python3
"""Train, validate, calibrate, and score Campaign Prophet's pre-contact models.

Decision-layer contract:
- Models are selected on an internal source-order validation slice carved out of the
  training period. The final temporal holdout is reported once and never used to pick
  a model.
- Ranking (contact capacity decisions) is driven by the selected model's *uncalibrated*
  scores. Isotonic calibration, fit on out-of-fold training predictions, is used only to
  estimate expected responder counts.
- Capacity is expressed as a percent of the eligible population, not a currency budget,
  and is published with bootstrap confidence intervals and a binomial test against the
  base rate.
- A naive full-information model (including `duration` and `campaign`) is trained on the
  same random split to quantify the cost of the decision-time feature policy.
"""

from __future__ import annotations

import os

# Pin single-threaded numerics before sklearn/numpy import so histogram and BLAS
# reductions are byte-identical across runs (required by the CI drift guard).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.calibration import calibration_curve


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "processed" / "cleaned_feature_engineered_bank_marketing.csv"
CONFIG_PATH = REPO_ROOT / "config" / "roi_scenarios.json"
OUTPUT_DIR = REPO_ROOT / "outputs" / "notebook_02"
VISUAL_DIR = REPO_ROOT / "visuals"
RANDOM_STATE = 42

TEMPORAL_TRAIN_FRACTION = 0.80
INTERNAL_VALIDATION_FRACTION = 0.20
CAPACITY_LEVELS = (5, 10, 20, 30, 50)
BOOTSTRAP_SAMPLES = 500
BOOTSTRAP_BRIER_SAMPLES = 1000

# `duration` and `campaign` are unavailable before the call. The target and explanatory
# convenience columns are excluded so every model uses only operationally available inputs.
EXCLUDED_FEATURES = {"y", "success", "age_group", "duration_category", "duration", "campaign"}
# Deliberately leaked into the "naive" comparison model to demonstrate their cost.
LEAKY_FEATURES = {"duration", "campaign"}
NAIVE_EXCLUDED_FEATURES = EXCLUDED_FEATURES - LEAKY_FEATURES

SHIFT_NUMERIC = ("euribor3m", "nr.employed", "pdays", "cons.conf.idx", "emp.var.rate")
SHIFT_CATEGORICAL = ("month", "contact", "poutcome")
CALIBRATION_METHODS = ("Uncalibrated", "Isotonic", "Sigmoid")


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def model_factories(positive_weight: float):
    factories = {
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=5000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
    }
    factories["Histogram Gradient Boosting"] = lambda: HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.0,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )
    return factories


def metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "precision_at_0_50": precision_score(y_true, predicted, zero_division=0),
        "recall_at_0_50": recall_score(y_true, predicted, zero_division=0),
        "f1_at_0_50": f1_score(y_true, predicted, zero_division=0),
        "brier_score": brier_score_loss(y_true, probabilities),
    }


def build_pipeline(features: pd.DataFrame, model) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor(features)), ("model", model)])


def evaluate_split(
    split_name: str,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    positive_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    rows = []
    probabilities = {}
    for model_name, factory in model_factories(positive_weight).items():
        pipeline = build_pipeline(x_train, factory())
        pipeline.fit(x_train, y_train)
        prediction = pipeline.predict_proba(x_test)[:, 1]
        probabilities[model_name] = prediction
        rows.append({"split": split_name, "model": model_name, **metrics(y_test, prediction)})
    return pd.DataFrame(rows), probabilities


# --------------------------------------------------------------------------------------
# Uncertainty and calibration statistics
# --------------------------------------------------------------------------------------


def _top_k_metrics(y_sorted: np.ndarray, capacity: int) -> tuple[int, int, float, float]:
    contacts = max(1, int(np.ceil(len(y_sorted) * capacity / 100)))
    contacts = min(contacts, len(y_sorted))
    cumulative = np.cumsum(y_sorted)
    responders = int(cumulative[contacts - 1])
    total = int(cumulative[-1])
    precision = responders / contacts
    coverage = 100 * responders / total if total else np.nan
    lift = precision / y_sorted.mean() if y_sorted.mean() else np.nan
    return contacts, responders, coverage, (precision, lift)


def build_capacity_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Top-K capacity table ranked by the uncalibrated decision score, with uncertainty."""
    y = scored["actual_subscription"].to_numpy()
    score = scored["ranking_score"].to_numpy()
    order = np.argsort(-score, kind="stable")
    y_sorted = y[order]
    base_precision = float(y.mean())
    total_responders = int(y.sum())

    bootstrap = {capacity: {"coverage": [], "precision": [], "lift": []} for capacity in CAPACITY_LEVELS}
    rng = np.random.default_rng(RANDOM_STATE)
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, len(y), len(y))
        resampled = y[idx][np.argsort(-score[idx], kind="stable")]
        for capacity in CAPACITY_LEVELS:
            _, _, coverage, (precision, lift) = _top_k_metrics(resampled, capacity)
            bootstrap[capacity]["coverage"].append(coverage)
            bootstrap[capacity]["precision"].append(precision)
            bootstrap[capacity]["lift"].append(lift)

    rows = []
    for capacity in CAPACITY_LEVELS:
        contacts, responders, coverage, (precision, lift) = _top_k_metrics(y_sorted, capacity)
        p_value = binomtest(responders, contacts, base_precision, alternative="greater").pvalue
        rows.append(
            {
                "capacity_percent": capacity,
                "contacts": contacts,
                "responders_captured": responders,
                "coverage_percent": coverage,
                "coverage_ci_low": np.percentile(bootstrap[capacity]["coverage"], 2.5),
                "coverage_ci_high": np.percentile(bootstrap[capacity]["coverage"], 97.5),
                "precision_at_k": precision,
                "precision_ci_low": np.percentile(bootstrap[capacity]["precision"], 2.5),
                "precision_ci_high": np.percentile(bootstrap[capacity]["precision"], 97.5),
                "lift": lift,
                "lift_ci_low": np.percentile(bootstrap[capacity]["lift"], 2.5),
                "lift_ci_high": np.percentile(bootstrap[capacity]["lift"], 97.5),
                "precision_p_value_vs_base_rate": p_value,
                "base_rate": base_precision,
                "expected_responders": float(
                    scored.iloc[:contacts]["calibrated_probability"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def write_impact_statement(capacity: pd.DataFrame) -> str:
    """One honest impact sentence, generated from the capacity table."""
    row = capacity[capacity["capacity_percent"] == 10].iloc[0]
    statement = (
        f"On the historical holdout, a 10% contact capacity captures **{row.coverage_percent:.1f}% of "
        f"responders** at **{row.lift:.2f}× a random contact**. This is descriptive, not causal uplift: "
        "the dataset contains only contacted customers and has no control group.\n"
    )
    (OUTPUT_DIR / "impact_statement.md").write_text(statement, encoding="utf-8")
    return statement


def build_capacity_by_model(y_test: pd.Series, probabilities: dict[str, np.ndarray]) -> pd.DataFrame:
    """Operational model comparison at each capacity (does the AUC edge buy responders?)."""
    y = y_test.to_numpy()
    rows = []
    for model_name, scores in probabilities.items():
        order = np.argsort(-scores, kind="stable")
        y_sorted = y[order]
        for capacity in CAPACITY_LEVELS:
            contacts, responders, coverage, (precision, lift) = _top_k_metrics(y_sorted, capacity)
            rows.append(
                {
                    "model": model_name,
                    "capacity_percent": capacity,
                    "contacts": contacts,
                    "responders_captured": responders,
                    "coverage_percent": coverage,
                    "precision_at_k": precision,
                    "lift": lift,
                }
            )
    return pd.DataFrame(rows)


def build_feature_importance(pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Permutation importance on the holdout — predictive association, not causation."""
    result = permutation_importance(
        pipeline,
        x_test,
        y_test,
        scoring="average_precision",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    frame = pd.DataFrame(
        {
            "feature": list(x_test.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return frame.sort_values("importance_mean", ascending=False, kind="stable").reset_index(drop=True)


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> float:
    edges = np.unique(np.quantile(probabilities, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return float(abs(y_true.mean() - probabilities.mean()))
    bin_index = np.clip(np.digitize(probabilities, edges[1:-1]), 0, len(edges) - 2)
    error = 0.0
    for bucket in range(len(edges) - 1):
        mask = bin_index == bucket
        if mask.sum() == 0:
            continue
        error += (mask.sum() / len(probabilities)) * abs(y_true[mask].mean() - probabilities[mask].mean())
    return float(error)


def calibration_slope_intercept(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    epsilon = 1e-6
    clipped = np.clip(probabilities, epsilon, 1 - epsilon)
    log_odds = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    fitted = LogisticRegression(max_iter=1000).fit(log_odds, y_true)
    return float(fitted.coef_[0][0]), float(fitted.intercept_[0])


def bootstrap_brier_difference(
    y_true: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, samples: int = BOOTSTRAP_BRIER_SAMPLES
) -> tuple[float, float, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    differences = np.empty(samples)
    for sample in range(samples):
        idx = rng.integers(0, len(y_true), len(y_true))
        differences[sample] = brier_score_loss(y_true[idx], baseline[idx]) - brier_score_loss(
            y_true[idx], candidate[idx]
        )
    return float(differences.mean()), float(np.percentile(differences, 2.5)), float(np.percentile(differences, 97.5))


def _population_stability_index(
    train_values: pd.Series, holdout_values: pd.Series, numeric: bool
) -> float:
    epsilon = 1e-6
    if numeric:
        edges = np.unique(np.quantile(train_values, np.linspace(0, 1, 11)))
        if len(edges) < 3:
            edges = np.unique(np.linspace(train_values.min(), train_values.max(), 11))
        edges = edges.astype(float)
        edges[0], edges[-1] = -np.inf, np.inf
        train_counts = np.histogram(train_values, bins=edges)[0]
        holdout_counts = np.histogram(holdout_values, bins=edges)[0]
    else:
        categories = sorted(set(train_values.unique()) | set(holdout_values.unique()))
        train_counts = train_values.value_counts().reindex(categories, fill_value=0).to_numpy()
        holdout_counts = holdout_values.value_counts().reindex(categories, fill_value=0).to_numpy()
    train_share = np.clip(train_counts / max(train_counts.sum(), 1), epsilon, None)
    holdout_share = np.clip(holdout_counts / max(holdout_counts.sum(), 1), epsilon, None)
    return float(np.sum((holdout_share - train_share) * np.log(holdout_share / train_share)))


def build_distribution_shift(data: pd.DataFrame, target: pd.Series, split_at: int) -> pd.DataFrame:
    train = data.iloc[:split_at]
    holdout = data.iloc[split_at:]
    rows = [
        {
            "feature": "subscription_rate",
            "kind": "target",
            "train_value": round(float(target.iloc[:split_at].mean()), 4),
            "holdout_value": round(float(target.iloc[split_at:].mean()), 4),
            "psi": np.nan,
        }
    ]
    for column in SHIFT_NUMERIC:
        rows.append(
            {
                "feature": column,
                "kind": "numeric",
                "train_value": round(float(train[column].mean()), 4),
                "holdout_value": round(float(holdout[column].mean()), 4),
                "psi": round(_population_stability_index(train[column], holdout[column], numeric=True), 4),
            }
        )
    for column in SHIFT_CATEGORICAL:
        train_shares = train[column].value_counts(normalize=True)
        holdout_shares = holdout[column].value_counts(normalize=True)
        top = train_shares.idxmax()
        rows.append(
            {
                "feature": column,
                "kind": "categorical",
                "train_value": f"{top} ({train_shares[top]:.1%})",
                "holdout_value": f"{top} ({holdout_shares.get(top, 0.0):.1%})",
                "psi": round(_population_stability_index(train[column], holdout[column], numeric=False), 4),
            }
        )
    return pd.DataFrame(rows)


def build_leakage_comparison(
    random_results: pd.DataFrame,
    naive_table: pd.DataFrame,
    y_test: pd.Series,
    pre_probabilities: dict[str, np.ndarray],
    naive_probabilities: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Add paired bootstrap CIs, AUC gaps, and top-decile operational lift."""
    y = y_test.to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)
    resamples = [rng.integers(0, len(y), len(y)) for _ in range(BOOTSTRAP_SAMPLES)]

    def auc_ci(scores: np.ndarray) -> tuple[float, float]:
        values = [roc_auc_score(y[idx], scores[idx]) for idx in resamples]
        return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))

    def top_decile_lift(scores: np.ndarray) -> float:
        order = np.argsort(-scores, kind="stable")
        contacts = max(1, int(np.ceil(len(y) * 0.10)))
        return float(y[order][:contacts].mean() / y.mean())

    frame = pd.concat(
        [random_results.assign(feature_set="pre_contact"), naive_table.assign(feature_set="naive_full_information")],
        ignore_index=True,
    )
    frame["roc_auc_ci_low"] = np.nan
    frame["roc_auc_ci_high"] = np.nan
    frame["roc_auc_gap_vs_pre_contact"] = np.nan
    frame["roc_auc_gap_ci_low"] = np.nan
    frame["roc_auc_gap_ci_high"] = np.nan
    frame["top_decile_lift"] = np.nan
    for model_name in pre_probabilities:
        pre = pre_probabilities[model_name]
        naive = naive_probabilities[model_name]
        low, high = auc_ci(pre)
        pre_mask = (frame["model"] == model_name) & (frame["feature_set"] == "pre_contact")
        frame.loc[pre_mask, ["roc_auc_ci_low", "roc_auc_ci_high", "top_decile_lift"]] = [
            low,
            high,
            top_decile_lift(pre),
        ]
        low, high = auc_ci(naive)
        naive_mask = (frame["model"] == model_name) & (frame["feature_set"] == "naive_full_information")
        frame.loc[naive_mask, ["roc_auc_ci_low", "roc_auc_ci_high", "top_decile_lift"]] = [
            low,
            high,
            top_decile_lift(naive),
        ]
        gaps = [roc_auc_score(y[idx], naive[idx]) - roc_auc_score(y[idx], pre[idx]) for idx in resamples]
        frame.loc[naive_mask, ["roc_auc_gap_vs_pre_contact", "roc_auc_gap_ci_low", "roc_auc_gap_ci_high"]] = [
            float(np.mean(gaps)),
            float(np.percentile(gaps, 2.5)),
            float(np.percentile(gaps, 97.5)),
        ]
    return frame


def make_visuals(
    y_test: pd.Series,
    probabilities: dict[str, np.ndarray],
    selected_name: str,
    raw_selected: np.ndarray,
    calibrated: np.ndarray,
    cumulative_gains: pd.DataFrame,
    capacity_by_model: pd.DataFrame,
    leakage: pd.DataFrame,
    feature_importance: pd.DataFrame,
    shift: pd.DataFrame,
) -> None:
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, prediction in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_test, prediction)
        ax.plot(recall, precision, label=f"{model_name} (AP={average_precision_score(y_test, prediction):.3f})")
    ax.axhline(y_test.mean(), color="black", linestyle="--", linewidth=1, label="Baseline")
    ax.set(xlabel="Recall", ylabel="Precision", title="Temporal holdout: precision-recall comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VISUAL_DIR / "precision_recall_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, prediction in [("Uncalibrated (decision score)", raw_selected), ("Calibrated", calibrated)]:
        observed, predicted = calibration_curve(y_test, prediction, n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, marker="o", label=label)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed conversion rate", title=f"{selected_name}: calibration on temporal holdout")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VISUAL_DIR / "calibration_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cumulative_gains["contact_percent"], cumulative_gains["subscriber_capture_percent"])
    ax.plot([0, 100], [0, 100], color="black", linestyle="--", linewidth=1, label="No-model baseline")
    ax.set(xlabel="Customers contacted (%)", ylabel="Subscribers captured (%)", title="Temporal holdout cumulative gains (uncalibrated ranking)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VISUAL_DIR / "cumulative_gains.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, group in capacity_by_model.groupby("model"):
        ax.plot(group["capacity_percent"], group["lift"], marker="o", label=model_name)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="No-model lift")
    ax.set(xlabel="Contact capacity (%)", ylabel="Lift vs base rate", title="Operational capacity comparison by model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VISUAL_DIR / "capacity_lift_by_model.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    means = leakage.pivot(index="model", columns="feature_set", values="roc_auc")
    lower = leakage.pivot(index="model", columns="feature_set", values="roc_auc_ci_low")
    upper = leakage.pivot(index="model", columns="feature_set", values="roc_auc_ci_high")
    error = (upper - lower) / 2
    means.plot.bar(ax=ax, yerr=error, capsize=3)
    ax.set(xlabel="Model", ylabel="ROC-AUC", title="Random-split ROC-AUC: pre-contact vs naive full-information")
    ax.legend(title="Feature set")
    fig.tight_layout()
    fig.savefig(VISUAL_DIR / "leakage_comparison.png", dpi=160)
    plt.close(fig)

    top_features = feature_importance.head(10).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_features["feature"], top_features["importance_mean"], xerr=top_features["importance_std"])
    ax.set(xlabel="Drop in average precision when permuted", title=f"{selected_name}: permutation importance (holdout)")
    fig.tight_layout()
    fig.savefig(VISUAL_DIR / "feature_importance.png", dpi=160)
    plt.close(fig)

    psi_rows = shift.dropna(subset=["psi"]).sort_values("psi")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(psi_rows["feature"], psi_rows["psi"])
    ax.axvline(0.1, color="black", linestyle="--", linewidth=1, label="0.10 (moderate shift)")
    ax.axvline(0.25, color="red", linestyle="--", linewidth=1, label="0.25 (major shift)")
    ax.set(xlabel="Population Stability Index", title="Train-period vs holdout distribution shift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VISUAL_DIR / "distribution_shift.png", dpi=160)
    plt.close(fig)


def write_model_card(
    selection: dict,
    calibration: pd.DataFrame,
    capacity: pd.DataFrame,
    shift: pd.DataFrame,
    leakage: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> None:
    path = REPO_ROOT / "docs" / "model_card.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    table_rows = "\n".join(
        f"| {row.capacity_percent}% | {int(row.contacts):,} | {int(row.responders_captured):,} | "
        f"{row.coverage_percent:.1f}% ({row.coverage_ci_low:.1f}–{row.coverage_ci_high:.1f}) | "
        f"{row.precision_at_k:.3f} | {row.lift:.2f}× ({row.lift_ci_low:.2f}–{row.lift_ci_high:.2f}) | "
        f"{row.precision_p_value_vs_base_rate:.1e} |"
        for row in capacity.itertuples()
    )
    base_rate = shift.loc[shift["feature"] == "subscription_rate"].iloc[0]
    forest_rows = leakage[leakage["model"] == "Random Forest"].set_index("feature_set")
    leakage_forest_pre = float(forest_rows.loc["pre_contact", "roc_auc"])
    leakage_forest_naive = float(forest_rows.loc["naive_full_information", "roc_auc"])
    leakage_gap = float(forest_rows.loc["naive_full_information", "roc_auc_gap_vs_pre_contact"])
    leakage_gap_low = float(forest_rows.loc["naive_full_information", "roc_auc_gap_ci_low"])
    leakage_gap_high = float(forest_rows.loc["naive_full_information", "roc_auc_gap_ci_high"])
    brier_diff = selection["calibration"]["brier_difference_uncalibrated_minus_isotonic"]
    top_features = ", ".join(
        f"`{row.feature}` ({row.importance_mean:.3f})" for row in feature_importance.head(5).itertuples()
    )
    capacity_10 = capacity[capacity["capacity_percent"] == 10].iloc[0]
    card = f"""# Campaign Prophet model card

## Purpose

Rank customers for term-deposit outreach using information available before a call, so a
limited contact capacity can be spent on the highest-ranked customers. This is a portfolio
analysis, not a production banking decision system.

## Selected model

**{selection['selected_model']}**, selected by highest Average Precision on an internal
source-order validation slice carved out of the training period. The final temporal holdout
is reported once and never used for model selection. The random stratified split is retained
only as a benchmark.

## Validation and probability use

The first {int(round(TEMPORAL_TRAIN_FRACTION * 100))}% of rows form the training period; the final
{int(round((1 - TEMPORAL_TRAIN_FRACTION) * 100))}% is held out. Within the training period the last
{int(round(INTERNAL_VALIDATION_FRACTION * 100))}% is used for model selection. Source order is treated as
chronological because the dataset represents successive campaigns; this is an explicit dataset
limitation.

Contact decisions are ranked on the model's **uncalibrated** score. Ranking is monotone under
calibration, so calibration cannot improve ordering (Spearman rank correlation with the calibrated
scores on the holdout: {selection['calibration']['ranking_spearman_uncalibrated_vs_calibrated']:.4f}). Isotonic calibration,
fit on out-of-fold training predictions, is applied only to produce the expected-responder counts in
the capacity table.

| Measure on temporal holdout | Uncalibrated | Isotonic | Sigmoid |
|---|---:|---:|---:|
| ROC-AUC | {calibration.loc['Uncalibrated', 'roc_auc']:.4f} | {calibration.loc['Isotonic', 'roc_auc']:.4f} | {calibration.loc['Sigmoid', 'roc_auc']:.4f} |
| Average Precision | {calibration.loc['Uncalibrated', 'average_precision']:.4f} | {calibration.loc['Isotonic', 'average_precision']:.4f} | {calibration.loc['Sigmoid', 'average_precision']:.4f} |
| Brier score | {calibration.loc['Uncalibrated', 'brier_score']:.4f} | {calibration.loc['Isotonic', 'brier_score']:.4f} | {calibration.loc['Sigmoid', 'brier_score']:.4f} |
| Expected calibration error | {calibration.loc['Uncalibrated', 'expected_calibration_error']:.4f} | {calibration.loc['Isotonic', 'expected_calibration_error']:.4f} | {calibration.loc['Sigmoid', 'expected_calibration_error']:.4f} |
| Calibration slope | {calibration.loc['Uncalibrated', 'calibration_slope']:.3f} | {calibration.loc['Isotonic', 'calibration_slope']:.3f} | {calibration.loc['Sigmoid', 'calibration_slope']:.3f} |

Brier difference (uncalibrated − isotonic), paired bootstrap 95% CI: **{brier_diff:+.4f}**
({selection['calibration']['brier_difference_ci_low']:+.4f}, {selection['calibration']['brier_difference_ci_high']:+.4f}).

## Capacity policy

Capacity is expressed as a percent of the eligible holdout population, not a currency budget.
"Coverage" is the share of all holdout responders reached; "lift" is precision at K over the
holdout base rate. Intervals are paired bootstrap 95% CIs; the p-value is a one-sided binomial
test of precision@K against the holdout base rate.

| Capacity | Contacts | Responders | Coverage % (95% CI) | Precision@K | Lift (95% CI) | p vs base |
|---|---:|---:|---:|---:|---:|---:|
{table_rows}

These are holdout descriptions, not causal uplift estimates: the dataset has no control group.

## Leakage policy cost

A naive model allowed to use `duration` and `campaign` (post-contact fields) is compared with the
deployable pre-contact model on an identical random split. Random Forest ROC-AUC rises from
{leakage_forest_pre:.4f} (pre-contact) to {leakage_forest_naive:.4f} (full information); the paired
bootstrap gap is **{leakage_gap:+.4f}** ({leakage_gap_low:+.4f}, {leakage_gap_high:+.4f}). That is the
measured operational cost of the decision-time feature policy. See `leakage_comparison.csv` and
`visuals/leakage_comparison.png`.

## Feature importance (predictive)

Permutation importance on the temporal holdout (drop in average precision) ranks
{top_features}. This is a predictive-association measure, **not a causal effect**.
See `feature_importance.csv` and `visuals/feature_importance.png`.

## Distribution shift

The training-period subscription rate is {base_rate['train_value']:.1%} against a holdout rate of
{base_rate['holdout_value']:.1%}. Per-feature population stability indices are published in
`distribution_shift.csv` and `visuals/distribution_shift.png`.

## Impact framing

On the historical holdout, a 10% contact capacity captures **{capacity_10['coverage_percent']:.1f}% of
responders** at **{capacity_10['lift']:.2f}× a random contact**. This is descriptive, not causal
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
"""
    path.write_text(card, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    target = (data["y"] == "yes").astype(int)
    features = data.drop(columns=sorted(EXCLUDED_FEATURES))
    naive_features = data.drop(columns=sorted(NAIVE_EXCLUDED_FEATURES))

    indices = np.arange(len(data))
    random_train_idx, random_test_idx = train_test_split(
        indices, test_size=0.20, random_state=RANDOM_STATE, stratify=target
    )
    x_random_train = features.iloc[random_train_idx]
    x_random_test = features.iloc[random_test_idx]
    y_random_train = target.iloc[random_train_idx]
    y_random_test = target.iloc[random_test_idx]
    random_results, random_probabilities = evaluate_split(
        "random_stratified_benchmark", x_random_train, x_random_test, y_random_train, y_random_test
    )

    # Phase 2: quantify the cost of excluding post-contact fields on the same random split.
    naive_table, naive_probabilities = evaluate_split(
        "naive_full_information_random",
        naive_features.iloc[random_train_idx],
        naive_features.iloc[random_test_idx],
        y_random_train,
        y_random_test,
    )
    leakage = build_leakage_comparison(random_results, naive_table, y_random_test, random_probabilities, naive_probabilities)
    leakage.to_csv(OUTPUT_DIR / "leakage_comparison.csv", index=False)

    # Training period / final temporal holdout, kept in source order.
    split_at = int(len(data) * TEMPORAL_TRAIN_FRACTION)
    x_temporal_train, x_temporal_test = features.iloc[:split_at], features.iloc[split_at:]
    y_temporal_train, y_temporal_test = target.iloc[:split_at], target.iloc[split_at:]

    # Model selection uses a held-out slice within the training period.
    inner_split = int(len(x_temporal_train) * (1 - INTERNAL_VALIDATION_FRACTION))
    x_fit, y_fit = x_temporal_train.iloc[:inner_split], y_temporal_train.iloc[:inner_split]
    x_val, y_val = x_temporal_train.iloc[inner_split:], y_temporal_train.iloc[inner_split:]
    validation_results, _ = evaluate_split("internal_source_order_validation", x_fit, x_val, y_fit, y_val)
    validation_results.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)
    ranked_validation = validation_results.sort_values(["average_precision", "roc_auc"], ascending=False, kind="stable")
    selected_name = ranked_validation.iloc[0]["model"]

    # The final holdout is touched exactly once, for reporting only.
    temporal_results, temporal_probabilities = evaluate_split(
        "source_order_temporal_holdout", x_temporal_train, x_temporal_test, y_temporal_train, y_temporal_test
    )
    all_metrics = pd.concat([random_results, temporal_results], ignore_index=True)
    all_metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    # Final decision model uses the full training period.
    weight = float((y_temporal_train == 0).sum() / (y_temporal_train == 1).sum())
    decision_pipeline = build_pipeline(x_temporal_train, model_factories(weight)[selected_name]())
    decision_pipeline.fit(x_temporal_train, y_temporal_train)
    uncalibrated = decision_pipeline.predict_proba(x_temporal_test)[:, 1]

    # Phase 3: calibrators fit on out-of-fold training predictions, never on the holdout.
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    out_of_fold = cross_val_predict(
        build_pipeline(x_temporal_train, model_factories(weight)[selected_name]()),
        x_temporal_train,
        y_temporal_train,
        cv=folds,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(out_of_fold, y_temporal_train)
    isotonic_prediction = isotonic.predict(uncalibrated)
    sigmoid = LogisticRegression(max_iter=1000).fit(out_of_fold.reshape(-1, 1), y_temporal_train)
    sigmoid_prediction = sigmoid.predict_proba(uncalibrated.reshape(-1, 1))[:, 1]

    prediction_by_method = {
        "Uncalibrated": uncalibrated,
        "Isotonic": isotonic_prediction,
        "Sigmoid": sigmoid_prediction,
    }
    calibration_rows = []
    for method in CALIBRATION_METHODS:
        probabilities = prediction_by_method[method]
        slope, intercept = calibration_slope_intercept(y_temporal_test.to_numpy(), probabilities)
        calibration_rows.append(
            {
                "method": method,
                **metrics(y_temporal_test, probabilities),
                "expected_calibration_error": expected_calibration_error(
                    y_temporal_test.to_numpy(), probabilities
                ),
                "calibration_slope": slope,
                "calibration_intercept": intercept,
            }
        )
    calibration = pd.DataFrame(calibration_rows).set_index("method")
    calibration.to_csv(OUTPUT_DIR / "calibration_metrics.csv")
    ranking_spearman = float(spearmanr(uncalibrated, isotonic_prediction).statistic)
    calibration_acceptable = bool(ranking_spearman > 0.99)
    brier_mean, brier_low, brier_high = bootstrap_brier_difference(
        y_temporal_test.to_numpy(), uncalibrated, isotonic_prediction
    )

    # Phase 3: train-period vs holdout distribution shift.
    shift = build_distribution_shift(data, target, split_at)
    shift.to_csv(OUTPUT_DIR / "distribution_shift.csv", index=False)

    economics = json.loads(CONFIG_PATH.read_text())["operational_scenario"]

    # Decision ranking is the uncalibrated score; calibrated probabilities only feed expected counts.
    scored = pd.DataFrame(
        {
            "source_row": data.index[split_at:],
            "actual_subscription": y_temporal_test.to_numpy(),
            "ranking_score": uncalibrated,
            "calibrated_probability": isotonic_prediction,
        }
    ).sort_values("ranking_score", ascending=False, kind="stable").reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored) + 1)
    scored.to_csv(OUTPUT_DIR / "scored_temporal_holdout.csv", index=False)

    capacity = build_capacity_table(scored)
    capacity.to_csv(OUTPUT_DIR / "capacity_table.csv", index=False)
    impact_statement = write_impact_statement(capacity)
    capacity_by_model = build_capacity_by_model(y_temporal_test, temporal_probabilities)
    capacity_by_model.to_csv(OUTPUT_DIR / "capacity_table_by_model.csv", index=False)

    feature_importance = build_feature_importance(decision_pipeline, x_temporal_test, y_temporal_test)
    feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    total_subscribers = scored["actual_subscription"].sum()
    gain_rows = []
    for percent in range(1, 101):
        contacts = max(1, int(np.ceil(len(scored) * percent / 100)))
        capture = 100 * scored.iloc[:contacts]["actual_subscription"].sum() / total_subscribers
        gain_rows.append(
            {
                "contact_percent": percent,
                "contacts": contacts,
                "subscriber_capture_percent": capture,
                "lift": capture / percent,
            }
        )
    gains = pd.DataFrame(gain_rows)
    gains.to_csv(OUTPUT_DIR / "cumulative_gains.csv", index=False)

    selection = {
        "selected_model": selected_name,
        "selection_metric": "average_precision",
        "selection_split": "internal_source_order_validation",
        "internal_validation_rule": (
            f"Fit on the first {int(round((1 - INTERNAL_VALIDATION_FRACTION) * 100))}% of the training period; "
            f"select on the final {int(round(INTERNAL_VALIDATION_FRACTION * 100))}% of the training period."
        ),
        "final_holdout_used_for_selection": False,
        "decision_ranking_score": "uncalibrated",
        "calibrated_probability_use": "expected_responder_counts_only",
        "capacity_levels_percent": list(CAPACITY_LEVELS),
        "capacity_bootstrap_samples": BOOTSTRAP_SAMPLES,
        "validation_split": "source_order_temporal_holdout",
        "source_order_assumption": "Rows are treated as successive campaign observations; explicit timestamps are unavailable.",
        "feature_policy": {
            "excluded_features": sorted(EXCLUDED_FEATURES),
            "pre_contact_feature_count": len(features.columns),
            "naive_added_features": sorted(LEAKY_FEATURES),
        },
        "leakage_demonstration": {
            "evaluation_split": "random_stratified_benchmark",
            "artifact": "leakage_comparison.csv",
        },
        "calibration": {
            "method": "isotonic",
            "fit_split": "out_of_fold_training_predictions",
            "evaluated_on": "source_order_temporal_holdout",
            "ranking_spearman_uncalibrated_vs_calibrated": round(ranking_spearman, 6),
            "calibration_acceptable_for_ranking": calibration_acceptable,
            "brier_difference_uncalibrated_minus_isotonic": round(brier_mean, 6),
            "brier_difference_ci_low": round(brier_low, 6),
            "brier_difference_ci_high": round(brier_high, 6),
        },
        "operational_scenario": economics,
        "model_scope": "three-model comparison: Logistic Regression, Random Forest, Histogram Gradient Boosting",
        "permutation_importance": {
            "scope": "predictive_association_only",
            "evaluated_on": "source_order_temporal_holdout",
        },
    }
    (OUTPUT_DIR / "model_selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    make_visuals(
        y_temporal_test,
        temporal_probabilities,
        selected_name,
        uncalibrated,
        isotonic_prediction,
        gains,
        capacity_by_model,
        leakage,
        feature_importance,
        shift,
    )
    write_model_card(selection, calibration, capacity, shift, leakage, feature_importance)

    print(f"Selected model on internal validation: {selected_name}")
    print(f"Holdout uncalibrated ROC-AUC: {calibration.loc['Uncalibrated', 'roc_auc']:.4f}")
    print(f"Calibration ranking Spearman (gate > 0.99): {ranking_spearman:.4f}")
    print(f"Brier diff (uncal - isotonic): {brier_mean:+.4f} [{brier_low:+.4f}, {brier_high:+.4f}]")
    print(f"Artifacts: {OUTPUT_DIR.relative_to(REPO_ROOT)} and {VISUAL_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
