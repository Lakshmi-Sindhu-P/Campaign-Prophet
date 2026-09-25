#!/usr/bin/env python3
"""SQL analyst layer: reproduce the Notebook 01 segment handoffs in portable SQL.

The queries in ``sql/`` run against the tracked processed CSV loaded into in-memory
SQLite. Each result is verified against the equivalent pandas computation before it is
written, so the SQL surface is proven equivalent, not decorative.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_data import CONFIG_PATH, PROCESSED_PATH, expected_value_by_job  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = REPO_ROOT / "sql"
OUTPUT_DIR = REPO_ROOT / "outputs" / "notebook_01"


def _connect() -> sqlite3.Connection:
    data = pd.read_csv(PROCESSED_PATH)
    connection = sqlite3.connect(":memory:")
    data.to_sql("campaign", connection, index=False)
    return connection


def build_sql_frames(connection: sqlite3.Connection | None = None) -> dict[str, pd.DataFrame]:
    connection = connection or _connect()
    scenarios = json.loads(CONFIG_PATH.read_text())

    def scenario_params(name: str) -> dict[str, float]:
        scenario = scenarios[name]["Base Case"]
        return {"cost": scenario["cost_per_contact"], "revenue": scenario["revenue_per_success"]}

    return {
        "usd": pd.read_sql_query(
            (SQL_DIR / "expected_value_by_job.sql").read_text(),
            connection,
            params=scenario_params("usd"),
        ),
        "eur": pd.read_sql_query(
            (SQL_DIR / "expected_value_by_job.sql").read_text(),
            connection,
            params=scenario_params("eur"),
        ),
        "job": pd.read_sql_query((SQL_DIR / "job_subscription_summary.sql").read_text(), connection),
        "poutcome": pd.read_sql_query((SQL_DIR / "poutcome_subscription_summary.sql").read_text(), connection),
    }


def build_pandas_frames() -> dict[str, pd.DataFrame]:
    data = pd.read_csv(PROCESSED_PATH)
    scenarios = json.loads(CONFIG_PATH.read_text())
    return {
        "usd": expected_value_by_job(data, scenarios["usd"]["Base Case"]),
        "eur": expected_value_by_job(data, scenarios["eur"]["Base Case"]),
        "job": pd.crosstab(data["job"], data["y"]).reset_index(),
        "poutcome": pd.crosstab(data["poutcome"], data["y"]).reset_index(),
    }


def compare_frames() -> dict[str, bool]:
    sql_frames = build_sql_frames()
    pandas_frames = build_pandas_frames()
    results: dict[str, bool] = {}
    for key, sql_frame in sql_frames.items():
        try:
            pd.testing.assert_frame_equal(
                sql_frame.reset_index(drop=True),
                pandas_frames[key].reset_index(drop=True),
                check_dtype=False,
                check_names=False,
                rtol=1e-9,
                atol=1e-9,
            )
            results[key] = True
        except AssertionError:
            results[key] = False
    return results


def main() -> None:
    results = compare_frames()
    if not all(results.values()):
        raise SystemExit(f"SQL/pandas equivalence failed: {results}")
    frames = build_sql_frames()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames["usd"].to_csv(OUTPUT_DIR / "job_summary.csv", index=False)
    frames["usd"].to_csv(OUTPUT_DIR / "expected_value_usd_base.csv", index=False)
    frames["eur"].to_csv(OUTPUT_DIR / "expected_value_eur_base.csv", index=False)
    frames["job"].to_csv(OUTPUT_DIR / "job_subscription_summary.csv", index=False)
    frames["poutcome"].to_csv(OUTPUT_DIR / "poutcome_subscription_summary.csv", index=False)
    print(f"SQL layer verified against pandas; wrote handoffs to {OUTPUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
