import json
from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_processed_dataset_has_operational_model_contract():
    data = pd.read_csv(ROOT / "data" / "processed" / "cleaned_feature_engineered_bank_marketing.csv")
    required = {"y", "duration", "campaign", "success", "age_group", "duration_category", "had_previous_contact"}
    assert len(data) == 41176
    assert required.issubset(data.columns)
    assert set(data["y"].unique()) == {"yes", "no"}


def test_economics_are_transparent_and_have_positive_breakeven_logic():
    scenarios = json.loads((ROOT / "config" / "roi_scenarios.json").read_text())
    operational = scenarios["operational_scenario"]
    assert operational["cost_per_contact"] > 0
    assert operational["revenue_per_success"] > operational["cost_per_contact"]
    assert operational["currency"] == "EUR"


def test_published_readme_metrics_match_generated_artifacts():
    metrics_path = ROOT / "outputs" / "notebook_02" / "model_metrics.csv"
    assert metrics_path.exists(), "Run scripts/run_modeling.py before publishing the README."
    metrics = pd.read_csv(metrics_path)
    forest = metrics[
        (metrics["split"] == "random_stratified_benchmark") & (metrics["model"] == "Random Forest")
    ].iloc[0]
    temporal = metrics[
        (metrics["split"] == "source_order_temporal_holdout") & (metrics["model"] == "Random Forest")
    ].iloc[0]
    readme = (ROOT / "README.md").read_text()
    assert f"{forest['roc_auc']:.4f}" in readme
    assert f"{forest['average_precision']:.4f}" in readme
    assert f"{temporal['roc_auc']:.4f}" in readme
    assert f"{temporal['average_precision']:.4f}" in readme
    assert re.search(r"not a financial targeting recommendation", readme, re.IGNORECASE)


def test_generated_artifact_set_is_complete():
    expected = {
        "model_metrics.csv",
        "validation_metrics.csv",
        "calibration_metrics.csv",
        "model_selection.json",
        "scored_temporal_holdout.csv",
        "capacity_table.csv",
        "cumulative_gains.csv",
        "leakage_comparison.csv",
        "distribution_shift.csv",
        "feature_importance.csv",
        "tuning_sensitivity.csv",
        "impact_statement.md",
        "subgroup_errors.csv",
    }
    produced = {path.name for path in (ROOT / "outputs" / "notebook_02").glob("*")}
    assert expected.issubset(produced)


def test_model_comparison_is_three_model():
    metrics = pd.read_csv(ROOT / "outputs" / "notebook_02" / "model_metrics.csv")
    models = set(metrics["model"])
    assert {"Logistic Regression", "Random Forest", "Histogram Gradient Boosting"}.issubset(models)
    selection = json.loads((ROOT / "outputs" / "notebook_02" / "model_selection.json").read_text())
    assert "three-model" in selection["model_scope"]


def test_subgroup_error_analysis_is_descriptive():
    subgroups = pd.read_csv(ROOT / "outputs" / "notebook_02" / "subgroup_errors.csv")
    assert set(subgroups["dimension"]) == {"age_group", "job", "marital"}
    assert {"records", "responders", "selection_rate", "precision", "recall", "coverage"}.issubset(subgroups.columns)
    assert subgroups["recall"].between(0, 1).all()
    for _, group in subgroups.groupby("dimension"):
        assert abs(group["coverage"].sum() - 100) < 1e-6
    selection = json.loads((ROOT / "outputs" / "notebook_02" / "model_selection.json").read_text())
    assert "no_fairness_certification" in selection["subgroup_analysis"]["scope"]


def test_sql_layer_matches_pandas_handoffs():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import run_sql

    results = run_sql.compare_frames()
    assert results and all(results.values()), results


def test_impact_statement_is_generated_and_non_causal():
    statement = (ROOT / "outputs" / "notebook_02" / "impact_statement.md").read_text()
    assert "not causal uplift" in statement
    capacity = pd.read_csv(ROOT / "outputs" / "notebook_02" / "capacity_table.csv")
    row = capacity[capacity["capacity_percent"] == 10].iloc[0]
    assert f"{row.coverage_percent:.1f}%" in statement
    assert f"{row.lift:.2f}" in statement
    assert "not causal uplift" in (ROOT / "README.md").read_text()


def test_tuning_sensitivity_shows_defaults_stand():
    tuning = pd.read_csv(ROOT / "outputs" / "notebook_02" / "tuning_sensitivity.csv")
    assert {"model", "config", "roc_auc", "average_precision"}.issubset(tuning.columns)
    assert set(tuning["config"]).issuperset({"default"})
    assert {"Logistic Regression", "Random Forest", "Histogram Gradient Boosting"}.issubset(set(tuning["model"]))
    for _, group in tuning.groupby("model"):
        default = group[group["config"] == "default"]["average_precision"].iloc[0]
        # Tuning must not materially change validation ranking; else defaults should not stand.
        assert group["average_precision"].max() - default < 0.02


def test_feature_importance_is_predictive_only():
    importance = pd.read_csv(ROOT / "outputs" / "notebook_02" / "feature_importance.csv")
    assert {"feature", "importance_mean", "importance_std"}.issubset(importance.columns)
    assert importance["importance_mean"].is_monotonic_decreasing
    assert len(importance) > 0
    selection = json.loads((ROOT / "outputs" / "notebook_02" / "model_selection.json").read_text())
    assert selection["permutation_importance"]["scope"] == "predictive_association_only"


def test_leakage_comparison_quantifies_the_feature_policy_cost():
    leakage = pd.read_csv(ROOT / "outputs" / "notebook_02" / "leakage_comparison.csv")
    forest = leakage[leakage["model"] == "Random Forest"].set_index("feature_set")
    assert forest.loc["naive_full_information", "roc_auc"] > forest.loc["pre_contact", "roc_auc"]


def test_distribution_shift_reports_label_shift():
    shift = pd.read_csv(ROOT / "outputs" / "notebook_02" / "distribution_shift.csv")
    base_rate = shift[shift["feature"] == "subscription_rate"].iloc[0]
    assert base_rate["holdout_value"] > base_rate["train_value"]
    euribor = shift[shift["feature"] == "euribor3m"].iloc[0]
    assert euribor["psi"] > 0.25


def test_calibration_gate_is_recorded():
    selection = json.loads((ROOT / "outputs" / "notebook_02" / "model_selection.json").read_text())
    calibration = selection["calibration"]
    assert calibration["method"] == "isotonic"
    assert calibration["fit_split"] == "out_of_fold_training_predictions"
    assert calibration["ranking_spearman_uncalibrated_vs_calibrated"] > 0.99


def test_power_analysis_is_reproducible_and_present():
    payload = json.loads((ROOT / "outputs" / "experiment_design" / "power_analysis.json").read_text())
    assert payload["alpha_two_sided"] == 0.05
    assert payload["power"] == 0.80
    two_pp = next(row for row in payload["absolute_lift_scenarios"] if row["absolute_lift_pp"] == 2.0)
    assert two_pp["sample_size_per_arm"] == 4223
    assert {entry["eligible_records"] for entry in payload["minimum_detectable_effect"]} == {8236, 42230, 162960}


def test_decision_layer_ranks_on_uncalibrated_scores():
    scored = pd.read_csv(ROOT / "outputs" / "notebook_02" / "scored_temporal_holdout.csv")
    assert scored["ranking_score"].is_monotonic_decreasing
    assert scored["rank"].is_monotonic_increasing
    assert (scored["ranking_score"] >= 0).all()

    selection = json.loads((ROOT / "outputs" / "notebook_02" / "model_selection.json").read_text())
    assert selection["final_holdout_used_for_selection"] is False
    assert selection["decision_ranking_score"] == "uncalibrated"
    assert selection["calibrated_probability_use"] == "expected_responder_counts_only"


def test_capacity_table_is_consistent_with_ranked_holdout():
    capacity = pd.read_csv(ROOT / "outputs" / "notebook_02" / "capacity_table.csv")
    scored = pd.read_csv(ROOT / "outputs" / "notebook_02" / "scored_temporal_holdout.csv")
    assert list(capacity["capacity_percent"]) == [5, 10, 20, 30, 50]
    total_responders = scored["actual_subscription"].sum()
    for row in capacity.itertuples():
        top = scored.iloc[: int(row.contacts)]
        assert int(row.responders_captured) == int(top["actual_subscription"].sum())
        assert abs(row.coverage_percent - 100 * row.responders_captured / total_responders) < 1e-6
        assert row.coverage_ci_low <= row.coverage_percent <= row.coverage_ci_high
        assert row.lift_ci_low <= row.lift <= row.lift_ci_high
        assert row.lift > 1.0
        assert row.precision_p_value_vs_base_rate < 0.05


def test_capacity_model_comparison_is_published():
    by_model = pd.read_csv(ROOT / "outputs" / "notebook_02" / "capacity_table_by_model.csv")
    assert {"Logistic Regression", "Random Forest", "Histogram Gradient Boosting"}.issubset(set(by_model["model"]))
    top10 = by_model[by_model["capacity_percent"] == 10].set_index("model")
    # The higher-AUC model must not change the operational decision materially.
    assert abs(top10.loc["Random Forest", "responders_captured"] - top10.loc["Logistic Regression", "responders_captured"]) <= 50


def test_leakage_comparison_has_bootstrap_uncertainty():
    leakage = pd.read_csv(ROOT / "outputs" / "notebook_02" / "leakage_comparison.csv")
    assert {"roc_auc_ci_low", "roc_auc_ci_high", "roc_auc_gap_vs_pre_contact", "top_decile_lift"}.issubset(leakage.columns)
    forest = leakage[(leakage["model"] == "Random Forest")].set_index("feature_set")
    assert forest.loc["pre_contact", "roc_auc_ci_low"] <= forest.loc["pre_contact", "roc_auc"] <= forest.loc["pre_contact", "roc_auc_ci_high"]
    assert forest.loc["naive_full_information", "roc_auc_gap_ci_low"] > 0


def test_calibration_methods_and_diagnostics_are_published():
    calibration = pd.read_csv(ROOT / "outputs" / "notebook_02" / "calibration_metrics.csv").set_index("method")
    assert {"Uncalibrated", "Isotonic", "Sigmoid"}.issubset(set(calibration.index))
    assert {"expected_calibration_error", "calibration_slope", "calibration_intercept"}.issubset(calibration.columns)
    selection = json.loads((ROOT / "outputs" / "notebook_02" / "model_selection.json").read_text())
    calibration_block = selection["calibration"]
    assert calibration_block["brier_difference_ci_high"] < 0  # isotonic is significantly worse than uncalibrated
