#!/usr/bin/env python3
"""Decision-time capacity recommendation from the scored holdout artifacts.

Example:
    python scripts/recommend.py --capacity 20

This reads the artifacts produced by scripts/run_modeling.py and reports how much
responder coverage a given contact capacity buys, ranked on the uncalibrated decision
score. It makes no causal or financial claim and operates on the evaluated holdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "notebook_02"


def load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame]:
    scored_path = OUTPUT_DIR / "scored_temporal_holdout.csv"
    capacity_path = OUTPUT_DIR / "capacity_table.csv"
    if not scored_path.exists() or not capacity_path.exists():
        raise SystemExit("Run scripts/run_modeling.py before requesting a recommendation.")
    return pd.read_csv(scored_path), pd.read_csv(capacity_path)


def summarize(scored: pd.DataFrame, capacity_percent: int) -> dict:
    total_contacts = len(scored)
    contacts = max(1, int(np.ceil(total_contacts * capacity_percent / 100)))
    top = scored.iloc[:contacts]
    total_responders = int(scored["actual_subscription"].sum())
    responders = int(top["actual_subscription"].sum())
    base_precision = total_responders / total_contacts
    precision_at_k = responders / contacts
    return {
        "capacity_percent": capacity_percent,
        "contacts": contacts,
        "responders_captured": responders,
        "coverage_percent": 100 * responders / total_responders,
        "precision_at_k": precision_at_k,
        "lift": precision_at_k / base_precision,
        "expected_responders": float(top["calibrated_probability"].sum()),
        "total_holdout_records": total_contacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=int, default=20, help="Contact capacity as a percent of the eligible population.")
    parser.add_argument("--json", action="store_true", help="Emit the summary as JSON.")
    args = parser.parse_args()

    scored, capacity_table = load_artifacts()
    if args.capacity <= 0 or args.capacity > 100:
        raise SystemExit("--capacity must be between 1 and 100.")

    summary = summarize(scored, args.capacity)
    matched = capacity_table[capacity_table["capacity_percent"] == args.capacity]
    if not matched.empty:
        row = matched.iloc[0]
        summary.update(
            {
                "coverage_ci_low": float(row["coverage_ci_low"]),
                "coverage_ci_high": float(row["coverage_ci_high"]),
                "lift_ci_low": float(row["lift_ci_low"]),
                "lift_ci_high": float(row["lift_ci_high"]),
                "precision_p_value_vs_base_rate": float(row["precision_p_value_vs_base_rate"]),
            }
        )
    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print(f"Capacity policy: top {summary['capacity_percent']}% of {summary['total_holdout_records']:,} eligible records")
    print(f"  Contacts             : {summary['contacts']:,}")
    print(f"  Responders captured  : {summary['responders_captured']:,} ({summary['coverage_percent']:.1f}% of all responders)")
    if "coverage_ci_low" in summary:
        print(
            f"    95% bootstrap CI   : {summary['coverage_ci_low']:.1f}–{summary['coverage_ci_high']:.1f}%"
        )
    print(f"  Precision@K          : {summary['precision_at_k']:.3f}")
    print(f"  Lift vs base rate    : {summary['lift']:.2f}x", end="")
    if "lift_ci_low" in summary:
        print(
            f" (95% CI {summary['lift_ci_low']:.2f}–{summary['lift_ci_high']:.2f}, "
            f"p={summary['precision_p_value_vs_base_rate']:.1e} vs base rate)"
        )
    else:
        print()
    print(f"  Expected responders  : {summary['expected_responders']:.0f} (calibrated, relative only)")
    print("  Note: ranking is uncalibrated; no causal or financial claim is made.")


if __name__ == "__main__":
    main()
