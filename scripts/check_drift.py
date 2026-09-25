#!/usr/bin/env python3
"""Fail if regenerated artifacts drift from the committed ones.

CI regenerates every artifact on Linux while the committed reference was produced on macOS
(Accelerate vs OpenBLAS). Measurements from the first real cross-platform run: Random Forest
metrics differ by ~5e-4, top-K responder counts by up to ~13, coverage percentages by <1pp,
and the permutation-importance *row order* changes (it is sorted by near-zero floats).

The guard therefore checks:
- **structure exactly** (columns, row counts, set of categorical values),
- rows aligned by key (tables sorted by their key column before comparison), and
- **numbers within a gross cross-platform tolerance** (floats atol 2.0 + rtol 5e-2, ints +/-25).

This catches stale/unregenerated artifacts, missing columns, and model-set changes without
false alarms from platform floating point. It is a staleness guard, not a bitwise contract;
generated prose/HTML and PNGs are excluded and covered by the contract tests.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
FLOAT_RTOL = 5e-2
FLOAT_ATOL = 2.0
INT_ATOL = 25

CSV_ARTIFACTS = [
    "outputs/notebook_02/model_metrics.csv",
    "outputs/notebook_02/validation_metrics.csv",
    "outputs/notebook_02/leakage_comparison.csv",
    "outputs/notebook_02/calibration_metrics.csv",
    "outputs/notebook_02/capacity_table.csv",
    "outputs/notebook_02/capacity_table_by_model.csv",
    "outputs/notebook_02/cumulative_gains.csv",
    "outputs/notebook_02/feature_importance.csv",
    "outputs/notebook_02/tuning_sensitivity.csv",
    "outputs/notebook_02/subgroup_errors.csv",
    "outputs/notebook_02/distribution_shift.csv",
]
JSON_ARTIFACTS = [
    "outputs/notebook_02/model_selection.json",
    "outputs/experiment_design/power_analysis.json",
]


def committed_text(path: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"HEAD:{path}"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    return result.stdout if result.returncode == 0 else None


def compare_csv(path: str) -> list[str]:
    text = committed_text(path)
    if text is None:
        return []
    committed = pd.read_csv(io.StringIO(text))
    current = pd.read_csv(REPO_ROOT / path)
    problems: list[str] = []
    if list(current.columns) != list(committed.columns) or len(current) != len(committed):
        return [f"{path}: structure differs (columns/rows)"]

    key_columns = list(current.columns)
    committed = committed.sort_values(key_columns, kind="stable").reset_index(drop=True)
    current = current.sort_values(key_columns, kind="stable").reset_index(drop=True)

    for column in current.columns:
        left, right = committed[column], current[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            left_values, right_values = left.astype(float).to_numpy(), right.astype(float).to_numpy()
            if pd.api.types.is_integer_dtype(left) or pd.api.types.is_integer_dtype(right):
                tolerance = INT_ATOL
            else:
                tolerance = FLOAT_ATOL + FLOAT_RTOL * np.maximum(np.abs(left_values), np.abs(right_values))
            diff = np.abs(left_values - right_values)
            if len(diff) and np.any(diff > tolerance):
                problems.append(f"{path}[{column}]: max diff {diff.max():.3g}")
        elif not left.astype(str).equals(right.astype(str)):
            problems.append(f"{path}[{column}]: category values differ")
    return problems


def _diff_json(committed, current, path: str = "") -> list[str]:
    problems: list[str] = []
    if isinstance(committed, dict) and isinstance(current, dict):
        if committed.keys() != current.keys():
            problems.append(f"{path}: keys differ")
            return problems
        for key in committed:
            problems.extend(_diff_json(committed[key], current[key], f"{path}.{key}"))
    elif isinstance(committed, list) and isinstance(current, list):
        if len(committed) != len(current):
            problems.append(f"{path}: length differs")
        else:
            for index, (left, right) in enumerate(zip(committed, current)):
                problems.extend(_diff_json(left, right, f"{path}[{index}]"))
    elif isinstance(committed, bool) or isinstance(current, bool) or not isinstance(committed, (int, float)) or not isinstance(current, (int, float)):
        if committed != current:
            problems.append(f"{path}: {committed!r} != {current!r}")
    elif abs(committed - current) > FLOAT_ATOL + FLOAT_RTOL * max(abs(committed), abs(current)):
        problems.append(f"{path}: {committed} != {current}")
    return problems


def main() -> None:
    problems: list[str] = []
    for path in CSV_ARTIFACTS:
        problems.extend(compare_csv(path))
    for path in JSON_ARTIFACTS:
        text = committed_text(path)
        if text is None:
            continue
        problems.extend(_diff_json(json.loads(text), json.loads((REPO_ROOT / path).read_text()), path))

    if problems:
        print("Artifact drift detected:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    print(f"No structural drift across {len(CSV_ARTIFACTS) + len(JSON_ARTIFACTS)} artifacts; metrics within cross-platform tolerance.")


if __name__ == "__main__":
    main()
