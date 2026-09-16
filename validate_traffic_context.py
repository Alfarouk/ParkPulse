from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline import chronological_split, prepare_dataset
from src.traffic_context import (
    add_traffic_mapping_columns,
    validate_mapping_boundaries,
    validate_timestamp_mapping,
)


ARTIFACT_DIR = Path("artifacts/traffic_context_experiments")
PARKING_PATH = Path("data/dataset.csv")
TELEMATICS_PATH = Path("data/Year_2016.csv")
MODEL_NAMES = {"logistic", "random_forest", "catboost"}


def main():
    required = [
        ARTIFACT_DIR / "traffic_context_lookup.csv",
        ARTIFACT_DIR / "model_comparison.csv",
        ARTIFACT_DIR / "selected_model_comparison.csv",
        ARTIFACT_DIR / "threshold_search.csv",
        ARTIFACT_DIR / "experiment_metrics.json",
        ARTIFACT_DIR / "mapping_validation.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run `python compare_traffic_context.py` first. Missing: " + ", ".join(missing)
        )

    boundary_checks = validate_mapping_boundaries()
    if boundary_checks["validated_weekdays"] != 7 or boundary_checks["validated_hours"] != 23:
        raise AssertionError(f"Incomplete mapping boundary checks: {boundary_checks}")

    parking = prepare_dataset(PARKING_PATH)
    split = chronological_split(parking)
    observed_mapping_checks = {}
    for split_name, frame in (("train", split.train), ("valid", split.valid), ("test", split.test)):
        mapped = add_traffic_mapping_columns(frame)
        observed_mapping_checks[split_name] = validate_timestamp_mapping(mapped)

    lookup = pd.read_csv(ARTIFACT_DIR / "traffic_context_lookup.csv")
    if len(lookup) != 35 or lookup["traffic_context_key"].nunique() != 35:
        raise AssertionError("Traffic lookup must contain 35 unique day/hour contexts.")
    for prefix in ("traffic_speed", "traffic_acceleration"):
        segment_totals = (
            lookup[f"{prefix}_valid_segments"]
            + lookup[f"{prefix}_missing_segments"]
        )
        if not (segment_totals == 54280).all():
            raise AssertionError(f"{prefix} aggregation did not account for every road segment.")
    telematics_columns = set(pd.read_csv(TELEMATICS_PATH, nrows=0).columns)
    referenced_columns = set(lookup["traffic_speed_source_column"]) | set(
        lookup["traffic_acceleration_source_column"]
    )
    if not referenced_columns.issubset(telematics_columns):
        raise AssertionError("Traffic lookup references columns absent from Year_2016.csv.")

    comparison = pd.read_csv(ARTIFACT_DIR / "model_comparison.csv")
    expected_experiments = {"parking_only", "parking_plus_traffic"}
    if set(comparison["experiment"]) != expected_experiments:
        raise AssertionError("Comparison artifact does not contain both experiments.")
    if set(comparison["model"]) != MODEL_NAMES or set(comparison["split"]) != {"validation", "test"}:
        raise AssertionError("Comparison artifact must contain all three models and both splits.")

    thresholds = pd.read_csv(ARTIFACT_DIR / "threshold_search.csv")
    with open(ARTIFACT_DIR / "experiment_metrics.json", "r", encoding="utf-8") as file:
        results = json.load(file)

    for split_name in ("train", "valid", "test"):
        row_counts = {
            results[experiment]["classifier_rows"][split_name]
            for experiment in expected_experiments
        }
        positive_counts = {
            results[experiment]["positive_rows"][split_name]
            for experiment in expected_experiments
        }
        if len(row_counts) != 1 or len(positive_counts) != 1:
            raise AssertionError(
                f"Experiments do not use identical {split_name} classifier rows and targets."
            )

    for experiment_name in expected_experiments:
        selected = results[experiment_name]["selected_model"]
        validation_results = results[experiment_name]
        expected_selected = max(
            MODEL_NAMES,
            key=lambda model: (
                validation_results[model]["validation"]["f1"],
                validation_results[model]["validation"]["pr_auc"],
                -validation_results[model]["validation"]["brier"],
            ),
        )
        if selected != expected_selected:
            raise AssertionError(f"{experiment_name} was not selected from validation metrics.")

        for model_name in MODEL_NAMES:
            table = thresholds[
                (thresholds["experiment"] == experiment_name)
                & (thresholds["model"] == model_name)
            ]
            best_threshold = float(table.loc[table["f1"].idxmax(), "threshold"])
            stored_threshold = float(results[experiment_name][model_name]["threshold"])
            if not np.isclose(best_threshold, stored_threshold):
                raise AssertionError(
                    f"{experiment_name}/{model_name} threshold is not validation maximum F1."
                )

    print(json.dumps({
        "boundary_mapping_checks": boundary_checks,
        "parking_timestamp_checks": observed_mapping_checks,
        "lookup_contexts": len(lookup),
        "comparison_rows": len(comparison),
        "selected_models": {
            experiment: results[experiment]["selected_model"]
            for experiment in sorted(expected_experiments)
        },
    }, indent=2))
    print("\nPASS: traffic-context mapping and validation-only selection checks passed.")


if __name__ == "__main__":
    main()
