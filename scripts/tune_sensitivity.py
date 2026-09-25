#!/usr/bin/env python3
"""Bounded hyperparameter sensitivity on the internal validation slice.

Answers "why no tuning" with evidence rather than assertion. The final temporal holdout is
never touched here: if a config does not materially improve validation ranking, the shipped
defaults stand; if it did, the main pipeline would be re-run with it (one holdout read).

Output: outputs/notebook_02/tuning_sensitivity.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_modeling import (  # noqa: E402  (path injection must precede import)
    DATA_PATH,
    EXCLUDED_FEATURES,
    INTERNAL_VALIDATION_FRACTION,
    OUTPUT_DIR,
    RANDOM_STATE,
    TEMPORAL_TRAIN_FRACTION,
    build_pipeline,
)


# A small, explicit grid. The first entry of each list is the shipped default.
RF_GRID = [
    {"name": "default", "n_estimators": 300, "max_depth": 12, "min_samples_leaf": 10},
    {"name": "depth8", "n_estimators": 300, "max_depth": 8, "min_samples_leaf": 10},
    {"name": "depthNone", "n_estimators": 300, "max_depth": None, "min_samples_leaf": 10},
    {"name": "leaf5", "n_estimators": 300, "max_depth": 12, "min_samples_leaf": 5},
    {"name": "leaf20", "n_estimators": 300, "max_depth": 12, "min_samples_leaf": 20},
    {"name": "trees200", "n_estimators": 200, "max_depth": 12, "min_samples_leaf": 10},
]
HGB_GRID = [
    {"name": "default", "max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31},
    {"name": "lr0.1", "max_iter": 300, "learning_rate": 0.1, "max_leaf_nodes": 31},
    {"name": "iter200", "max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 31},
    {"name": "leaves63", "max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 63},
    {"name": "small", "max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 15},
]
LR_GRID = [
    {"name": "default", "C": 1.0},
    {"name": "C0.1", "C": 0.1},
    {"name": "C10", "C": 10.0},
]


def evaluate(pipeline, x_fit: pd.DataFrame, y_fit: pd.Series, x_val: pd.DataFrame, y_val: pd.Series) -> dict:
    pipeline.fit(x_fit, y_fit)
    scores = pipeline.predict_proba(x_val)[:, 1]
    return {
        "roc_auc": roc_auc_score(y_val, scores),
        "average_precision": average_precision_score(y_val, scores),
    }


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    target = (data["y"] == "yes").astype(int)
    features = data.drop(columns=sorted(EXCLUDED_FEATURES))

    split_at = int(len(data) * TEMPORAL_TRAIN_FRACTION)
    train = features.iloc[:split_at]
    y_train = target.iloc[:split_at]
    inner = int(len(train) * (1 - INTERNAL_VALIDATION_FRACTION))
    x_fit, y_fit = train.iloc[:inner], y_train.iloc[:inner]
    x_val, y_val = train.iloc[inner:], y_train.iloc[inner:]

    rows = []
    for config in RF_GRID:
        estimator = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            class_weight="balanced",
            n_jobs=1,
            random_state=RANDOM_STATE,
        )
        rows.append(
            {
                "model": "Random Forest",
                "config": config["name"],
                **evaluate(build_pipeline(x_fit, estimator), x_fit, y_fit, x_val, y_val),
            }
        )
    for config in HGB_GRID:
        estimator = HistGradientBoostingClassifier(
            max_iter=config["max_iter"],
            learning_rate=config["learning_rate"],
            max_leaf_nodes=config["max_leaf_nodes"],
            early_stopping=False,
            random_state=RANDOM_STATE,
        )
        rows.append(
            {
                "model": "Histogram Gradient Boosting",
                "config": config["name"],
                **evaluate(build_pipeline(x_fit, estimator), x_fit, y_fit, x_val, y_val),
            }
        )
    for config in LR_GRID:
        estimator = LogisticRegression(
            C=config["C"],
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        )
        rows.append(
            {
                "model": "Logistic Regression",
                "config": config["name"],
                **evaluate(build_pipeline(x_fit, estimator), x_fit, y_fit, x_val, y_val),
            }
        )

    frame = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DIR / "tuning_sensitivity.csv", index=False)

    for model_name, group in frame.groupby("model"):
        default = group[group["config"] == "default"]["average_precision"].iloc[0]
        best = group.sort_values("average_precision", ascending=False).iloc[0]
        delta = best["average_precision"] - default
        print(
            f"{model_name}: best='{best['config']}' AP={best['average_precision']:.4f} "
            f"(default {default:.4f}, delta {delta:+.4f})"
        )
    print(f"Artifacts: outputs/notebook_02/tuning_sensitivity.csv")


if __name__ == "__main__":
    main()
