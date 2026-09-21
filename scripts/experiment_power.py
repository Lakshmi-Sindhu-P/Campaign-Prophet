#!/usr/bin/env python3
"""Reproducible sample-size and minimum-detectable-effect arithmetic for the prospective
randomized-holdback design.

This is design arithmetic only. The historical UCI dataset contains only contacted
customers and has no control arm, so no retrospective A/B or uplift analysis is possible.

Sizing uses a two-proportion z-test (two-sided alpha, 80% power).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from scipy.stats import norm


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "processed" / "cleaned_feature_engineered_bank_marketing.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "experiment_design"

ALPHA = 0.05
POWER = 0.80
HOLDBACK_FRACTION = 0.10
# Uplift scenarios expressed as absolute percentage-point gains over the baseline rate.
LIFT_SCENARIOS = (0.01, 0.02, 0.03, 0.05)
# Uplift scenarios expressed as relative gains over the baseline rate.
RELATIVE_LIFT_SCENARIOS = (0.10, 0.20, 0.30)
# Eligibility sizes at which to invert the calculation into a minimum detectable effect.
MDE_ELIGIBLE_SIZES = (8236, 42230, 162960)


def baseline_rate() -> float:
    import pandas as pd

    data = pd.read_csv(DATA_PATH)
    return float((data["y"] == "yes").mean())


def sample_size_per_arm(p1: float, p2: float, alpha: float = ALPHA, power: float = POWER) -> int:
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    pooled = (p1 + p2) / 2
    numerator = (
        z_alpha * math.sqrt(2 * pooled * (1 - pooled)) + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return math.ceil(numerator / (p2 - p1) ** 2)


def minimum_detectable_lift(
    control_n: int, p1: float, alpha: float = ALPHA, power: float = POWER
) -> float:
    """Smallest absolute lift detectable with the given (smaller, control) arm size."""
    low, high = 1e-5, 0.5
    for _ in range(60):
        mid = (low + high) / 2
        if sample_size_per_arm(p1, p1 + mid, alpha, power) <= control_n:
            high = mid
        else:
            low = mid
    return high


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = baseline_rate()

    absolute_scenarios = []
    for lift in LIFT_SCENARIOS:
        treatment = base + lift
        per_arm = sample_size_per_arm(base, treatment)
        eligible = math.ceil(per_arm / HOLDBACK_FRACTION)
        absolute_scenarios.append(
            {
                "baseline_rate": round(base, 4),
                "treatment_rate": round(treatment, 4),
                "absolute_lift_pp": round(lift * 100, 2),
                "relative_lift_percent": round(100 * lift / base, 1),
                "sample_size_per_arm": per_arm,
                "total_records_needed": 2 * per_arm,
                "eligible_for_10pct_holdback": eligible,
            }
        )

    relative_scenarios = []
    for relative in RELATIVE_LIFT_SCENARIOS:
        treatment = base * (1 + relative)
        per_arm = sample_size_per_arm(base, treatment)
        relative_scenarios.append(
            {
                "relative_lift_percent": round(relative * 100, 1),
                "treatment_rate": round(treatment, 4),
                "absolute_lift_pp": round((treatment - base) * 100, 2),
                "sample_size_per_arm": per_arm,
                "eligible_for_10pct_holdback": math.ceil(per_arm / HOLDBACK_FRACTION),
            }
        )

    mde = []
    for eligible in MDE_ELIGIBLE_SIZES:
        control_n = max(1, int(eligible * HOLDBACK_FRACTION))
        lift = minimum_detectable_lift(control_n, base)
        mde.append(
            {
                "eligible_records": eligible,
                "control_arm": control_n,
                "minimum_detectable_absolute_lift_pp": round(lift * 100, 2),
                "minimum_detectable_treatment_rate": round(base + lift, 4),
            }
        )

    payload = {
        "test": "two_proportion_z_test",
        "alpha_two_sided": ALPHA,
        "power": POWER,
        "holdback_fraction": HOLDBACK_FRACTION,
        "note": (
            "Design arithmetic only. The historical dataset has no untreated control group, "
            "so these sizes describe a future randomized holdback, not the UCI data."
        ),
        "absolute_lift_scenarios": absolute_scenarios,
        "relative_lift_scenarios": relative_scenarios,
        "minimum_detectable_effect": mde,
    }
    (OUTPUT_DIR / "power_analysis.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"Baseline rate: {base:.4f}")
    for scenario in absolute_scenarios:
        print(
            f"  +{scenario['absolute_lift_pp']:.2f}pp lift -> "
            f"{scenario['sample_size_per_arm']:,} per arm, "
            f"{scenario['eligible_for_10pct_holdback']:,} eligible for a 10% holdback"
        )
    for entry in mde:
        print(
            f"  with {entry['eligible_records']:,} eligible -> MDE "
            f"+{entry['minimum_detectable_absolute_lift_pp']:.2f}pp"
        )
    print(f"Artifacts: {(OUTPUT_DIR / 'power_analysis.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
