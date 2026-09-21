#!/usr/bin/env python3
"""Create reproducible Notebook 01 data and handoff artifacts.

The script uses the tracked processed data by default. With --refresh-data it downloads
the public UCI archive into a temporary cache, rebuilds the cleaned dataset, and never
commits the raw archive. All configuration handoffs are JSON rather than pickle files.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "cleaned_feature_engineered_bank_marketing.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "notebook_01"
CONFIG_PATH = REPO_ROOT / "config" / "roi_scenarios.json"
UCI_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"


def add_engineered_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["success"] = (frame["y"] == "yes").astype(int)
    frame["age_group"] = pd.cut(
        frame["age"],
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=["<25", "25-35", "35-45", "45-55", "55-65", "65+"],
    )
    frame["duration_category"] = pd.cut(
        frame["duration"],
        bins=[-1, 60, 180, 360, float("inf")],
        labels=["<1min", "1-3min", "3-6min", "6min+"],
    )
    frame["had_previous_contact"] = (frame["previous"] > 0).astype(int)
    return frame


def download_source_dataset() -> pd.DataFrame:
    """Download and read the nested UCI CSV without retaining raw data in Git."""
    with tempfile.TemporaryDirectory(prefix="campaign-prophet-") as temp_dir:
        archive_path = Path(temp_dir) / "bank_marketing.zip"
        urllib.request.urlretrieve(UCI_ARCHIVE_URL, archive_path)
        with zipfile.ZipFile(archive_path) as outer:
            nested_name = next(name for name in outer.namelist() if name.endswith("bank-additional.zip"))
            nested_bytes = outer.read(nested_name)
        with zipfile.ZipFile(io.BytesIO(nested_bytes)) as inner:
            csv_name = next(name for name in inner.namelist() if name.endswith("bank-additional-full.csv"))
            with inner.open(csv_name) as csv_file:
                return pd.read_csv(csv_file, sep=";")


def load_dataset(refresh_data: bool) -> pd.DataFrame:
    if refresh_data or not PROCESSED_PATH.exists():
        raw = download_source_dataset()
        data = add_engineered_columns(raw.drop_duplicates().reset_index(drop=True))
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(PROCESSED_PATH, index=False)
        return data

    data = pd.read_csv(PROCESSED_PATH)
    required = {"y", "success", "age_group", "duration_category", "had_previous_contact"}
    missing = required.difference(data.columns)
    if missing:
        data = add_engineered_columns(data)
        data.to_csv(PROCESSED_PATH, index=False)
    return data


def expected_value_by_job(data: pd.DataFrame, scenario: dict[str, float]) -> pd.DataFrame:
    summary = (
        data.groupby("job", observed=False)
        .agg(
            total_contacts=("success", "size"),
            successful_contacts=("success", "sum"),
            success_rate=("success", "mean"),
            avg_duration=("duration", "mean"),
        )
        .reset_index()
    )
    cost = scenario["cost_per_contact"]
    revenue = scenario["revenue_per_success"]
    summary["expected_value_per_contact"] = summary["success_rate"] * revenue - cost
    summary["total_cost"] = summary["total_contacts"] * cost
    summary["total_revenue"] = summary["successful_contacts"] * revenue
    summary["total_profit"] = summary["total_revenue"] - summary["total_cost"]
    summary["roi_percent"] = 100 * summary["total_profit"] / summary["total_cost"]
    return summary.sort_values("expected_value_per_contact", ascending=False).reset_index(drop=True)


def write_handoff_artifacts(data: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = json.loads(CONFIG_PATH.read_text())
    usd_base = expected_value_by_job(data, scenarios["usd"]["Base Case"])
    eur_base = expected_value_by_job(data, scenarios["eur"]["Base Case"])

    usd_base.to_csv(OUTPUT_DIR / "job_summary.csv", index=False)
    usd_base.to_csv(OUTPUT_DIR / "expected_value_usd_base.csv", index=False)
    eur_base.to_csv(OUTPUT_DIR / "expected_value_eur_base.csv", index=False)

    job_subscription = pd.crosstab(data["job"], data["y"]).reset_index()
    job_subscription.to_csv(OUTPUT_DIR / "job_subscription_summary.csv", index=False)
    pd.crosstab(data["poutcome"], data["y"]).reset_index().to_csv(
        OUTPUT_DIR / "poutcome_subscription_summary.csv", index=False
    )
    data.groupby("y", observed=False)["duration"].agg(["count", "mean", "std"]).reset_index().to_csv(
        OUTPUT_DIR / "duration_summary.csv", index=False
    )
    data.groupby("y", observed=False)["age"].agg(["count", "mean", "std"]).reset_index().to_csv(
        OUTPUT_DIR / "age_summary.csv", index=False
    )
    shutil.copy2(CONFIG_PATH, OUTPUT_DIR / "roi_scenarios.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-data", action="store_true", help="Download and rebuild the processed dataset from UCI.")
    args, _ = parser.parse_known_args()
    data = load_dataset(args.refresh_data)
    write_handoff_artifacts(data)
    print(f"Prepared {len(data):,} rows at {PROCESSED_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote transparent Notebook 01 handoff artifacts to {OUTPUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
