#!/usr/bin/env python3
"""Fail if regenerated artifacts drift from the committed ones beyond tolerance.

Used by CI after re-running the pipeline: the committed artifacts under ``outputs/`` and
``docs/model_card.md`` must still be reproducible. Numeric comparisons use rtol/atol so that
cross-platform BLAS differences do not produce false alarms; text artifacts are compared exactly.
Visual PNGs and the base64 report are intentionally excluded from the numeric contract.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RTOL = 1e-6
ATOL = 1e-9

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
TEXT_ARTIFACTS = [
    "outputs/notebook_02/impact_statement.md",
    "docs/model_card.md",
]


def committed_text(path: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"HEAD:{path}"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    return result.stdout if result.returncode == 0 else None


def _diff_json(committed, current, path: str = "") -> list[str]:
    problems: list[str] = []
    if isinstance(committed, dict) and isinstance(current, dict):
        for key in committed.keys() | current.keys():
            if key not in committed or key not in current:
                problems.append(f"{path}.{key} missing on one side")
            else:
                problems.extend(_diff_json(committed[key], current[key], f"{path}.{key}"))
    elif isinstance(committed, list) and isinstance(current, list):
        if len(committed) != len(current):
            problems.append(f"{path} length {len(committed)} != {len(current)}")
        else:
            for index, (left, right) in enumerate(zip(committed, current)):
                problems.extend(_diff_json(left, right, f"{path}[{index}]"))
    elif isinstance(committed, bool) or isinstance(current, bool):
        if committed != current:
            problems.append(f"{path} {committed!r} != {current!r}")
    elif isinstance(committed, (int, float)) and isinstance(current, (int, float)):
        if abs(committed - current) > ATOL + RTOL * max(abs(committed), abs(current)):
            problems.append(f"{path} {committed} != {current}")
    elif committed != current:
        problems.append(f"{path} {committed!r} != {current!r}")
    return problems


def main() -> None:
    problems: list[str] = []
    for path in CSV_ARTIFACTS:
        text = committed_text(path)
        if text is None:
            continue
        try:
            pd.testing.assert_frame_equal(
                pd.read_csv(REPO_ROOT / path),
                pd.read_csv(io.StringIO(text)),
                check_dtype=False,
                rtol=RTOL,
                atol=ATOL,
            )
        except AssertionError as exc:
            problems.append(f"{path}: {str(exc).splitlines()[0]}")
    for path in JSON_ARTIFACTS:
        text = committed_text(path)
        if text is None:
            continue
        problems.extend(_diff_json(json.loads(text), json.loads((REPO_ROOT / path).read_text()), path))
    for path in TEXT_ARTIFACTS:
        text = committed_text(path)
        if text is None:
            continue
        if text != (REPO_ROOT / path).read_text():
            problems.append(f"{path}: text differs from committed artifact")

    if problems:
        print("Artifact drift detected:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    print(f"No drift across {len(CSV_ARTIFACTS) + len(JSON_ARTIFACTS) + len(TEXT_ARTIFACTS)} artifacts.")


if __name__ == "__main__":
    main()
