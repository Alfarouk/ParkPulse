from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from catboost import Pool

from src.pipeline import (
    FEATURES,
    build_catboost_classifier,
    build_logistic_classifier,
    build_random_forest_classifier,
    choose_threshold_for_f1,
    chronological_split,
    classification_metrics,
    early_warning_candidates,
    prepare_dataset,
)
from src.traffic_context import (
    TRAFFIC_FEATURES,
    attach_traffic_context,
    build_traffic_context_lookup,
    validate_mapping_boundaries,
    validate_timestamp_mapping,
)


PARKING_PATH = Path("data/dataset.csv")
TELEMATICS_PATH = Path("data/Year_2016.csv")
ARTIFACT_DIR = Path("artifacts/traffic_context_experiments")
MODEL_DIR = Path("models/traffic_context_experiments")

EXPERIMENT_FEATURES = {
    "parking_only": FEATURES,
    "parking_plus_traffic": [*FEATURES, *TRAFFIC_FEATURES],
}
MODEL_NAMES = ("logistic", "random_forest", "catboost")
METRIC_NAMES = ("precision", "recall", "f1", "pr_auc", "roc_auc", "brier")


def timed_fit(name, fit_fn):
    start = time.perf_counter()
    fitted = fit_fn()
    return fitted, time.perf_counter() - start


def feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    matrix = df[feature_columns].copy()
    matrix["parking_id"] = matrix["parking_id"].astype(str)
    return matrix


def validation_selection_key(model_result: dict) -> tuple[float, float, float]:
    metrics = model_result["validation"]
    return metrics["f1"], metrics["pr_auc"], -metrics["brier"]


def train_experiment(name, feature_columns, split_frames):
    candidates = {
        split_name: early_warning_candidates(frame)
        for split_name, frame in split_frames.items()
    }
    matrices = {
        split_name: feature_matrix(frame, feature_columns)
        for split_name, frame in candidates.items()
    }
    targets = {
        split_name: frame["will_become_saturated_soon"].astype(int)
        for split_name, frame in candidates.items()
    }

    fitted_models = {}
    results = {}
    threshold_tables = []

    logistic, seconds = timed_fit(
        f"{name}/logistic",
        lambda: build_logistic_classifier(feature_columns=feature_columns).fit(
            matrices["train"], targets["train"]
        ),
    )
    fitted_models["logistic"] = logistic
    logistic_valid = logistic.predict_proba(matrices["valid"])[:, 1]
    threshold, table = choose_threshold_for_f1(targets["valid"], logistic_valid)
    results["logistic"] = {
        "fit_seconds": seconds,
        "threshold": threshold,
        "validation": classification_metrics(targets["valid"], logistic_valid, threshold),
    }
    threshold_tables.append(table.assign(experiment=name, model="logistic"))

    random_forest, seconds = timed_fit(
        f"{name}/random_forest",
        lambda: build_random_forest_classifier(feature_columns=feature_columns).fit(
            matrices["train"], targets["train"]
        ),
    )
    fitted_models["random_forest"] = random_forest
    forest_valid = random_forest.predict_proba(matrices["valid"])[:, 1]
    threshold, table = choose_threshold_for_f1(targets["valid"], forest_valid)
    results["random_forest"] = {
        "fit_seconds": seconds,
        "threshold": threshold,
        "validation": classification_metrics(targets["valid"], forest_valid, threshold),
    }
    threshold_tables.append(table.assign(experiment=name, model="random_forest"))

    train_pool = Pool(matrices["train"], targets["train"], cat_features=["parking_id"])
    valid_pool = Pool(matrices["valid"], targets["valid"], cat_features=["parking_id"])
    catboost, seconds = timed_fit(
        f"{name}/catboost",
        lambda: build_catboost_classifier().fit(
            train_pool,
            eval_set=valid_pool,
            early_stopping_rounds=80,
        ),
    )
    fitted_models["catboost"] = catboost
    catboost_valid = catboost.predict_proba(valid_pool)[:, 1]
    threshold, table = choose_threshold_for_f1(targets["valid"], catboost_valid)
    results["catboost"] = {
        "fit_seconds": seconds,
        "threshold": threshold,
        "best_iteration": int(catboost.get_best_iteration()),
        "validation": classification_metrics(targets["valid"], catboost_valid, threshold),
    }
    threshold_tables.append(table.assign(experiment=name, model="catboost"))

    # Selection is complete before any test probabilities or metrics are calculated.
    selected_model = max(MODEL_NAMES, key=lambda model: validation_selection_key(results[model]))

    test_probabilities = {}
    for model_name, model in fitted_models.items():
        if model_name == "catboost":
            test_pool = Pool(matrices["test"], targets["test"], cat_features=["parking_id"])
            probabilities = model.predict_proba(test_pool)[:, 1]
            model.save_model(MODEL_DIR / f"{name}_{model_name}.cbm")
        else:
            probabilities = model.predict_proba(matrices["test"])[:, 1]
            joblib.dump(model, MODEL_DIR / f"{name}_{model_name}.joblib")
        test_probabilities[model_name] = probabilities
        results[model_name]["test"] = classification_metrics(
            targets["test"],
            probabilities,
            results[model_name]["threshold"],
        )

    results["selected_model"] = selected_model
    results["selection_rule"] = (
        "highest validation F1; validation PR-AUC then lower validation Brier as tie-breakers"
    )
    results["feature_count"] = len(feature_columns)
    results["classifier_rows"] = {
        split_name: len(frame) for split_name, frame in candidates.items()
    }
    results["positive_rows"] = {
        split_name: int(target.sum()) for split_name, target in targets.items()
    }

    selected_predictions = candidates["test"][[
        "parking_id",
        "timestamp",
        "occupancy_ratio",
        "future_occupancy_ratio",
        "will_become_saturated_soon",
    ]].copy()
    selected_predictions.insert(0, "experiment", name)
    selected_predictions.insert(1, "selected_model", selected_model)
    selected_predictions["threshold"] = results[selected_model]["threshold"]
    selected_predictions["early_warning_probability"] = test_probabilities[selected_model]

    return results, pd.concat(threshold_tables, ignore_index=True), selected_predictions


def comparison_rows(experiment_results):
    rows = []
    for experiment_name, result in experiment_results.items():
        for model_name in MODEL_NAMES:
            for split_name in ("validation", "test"):
                metrics = result[model_name][split_name]
                rows.append({
                    "experiment": experiment_name,
                    "model": model_name,
                    "selected_model": model_name == result["selected_model"],
                    "split": split_name,
                    "feature_count": result["feature_count"],
                    "row_count": result["classifier_rows"][
                        "valid" if split_name == "validation" else "test"
                    ],
                    "positive_rows": result["positive_rows"][
                        "valid" if split_name == "validation" else "test"
                    ],
                    "fit_seconds": result[model_name]["fit_seconds"],
                    **metrics,
                })
    return pd.DataFrame(rows)


def selected_model_summary(experiment_results):
    rows = []
    for experiment_name, result in experiment_results.items():
        model_name = result["selected_model"]
        row = {
            "experiment": experiment_name,
            "selected_model": model_name,
            "feature_count": result["feature_count"],
            "threshold": result[model_name]["threshold"],
        }
        for split_name in ("validation", "test"):
            for metric_name in METRIC_NAMES:
                row[f"{split_name}_{metric_name}"] = result[model_name][split_name][metric_name]
        rows.append(row)

    summary = pd.DataFrame(rows)
    baseline = summary.loc[summary["experiment"] == "parking_only"].iloc[0]
    for metric_name in METRIC_NAMES:
        summary[f"test_{metric_name}_delta_vs_parking_only"] = (
            summary[f"test_{metric_name}"] - baseline[f"test_{metric_name}"]
        )
    return summary


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    parking = prepare_dataset(PARKING_PATH)
    split = chronological_split(parking)
    parking_splits = {"train": split.train, "valid": split.valid, "test": split.test}

    lookup = build_traffic_context_lookup(TELEMATICS_PATH)
    lookup.to_csv(ARTIFACT_DIR / "traffic_context_lookup.csv", index=False)
    traffic_splits = {
        split_name: attach_traffic_context(frame, lookup)
        for split_name, frame in parking_splits.items()
    }

    mapping_checks = {
        "boundary_rules": validate_mapping_boundaries(),
        "parking_timestamps": {
            split_name: validate_timestamp_mapping(frame)
            for split_name, frame in traffic_splits.items()
        },
        "unique_parking_context_keys": sorted({
            key
            for frame in traffic_splits.values()
            for key in frame["traffic_context_key"].unique()
        }),
        "lookup_context_count": len(lookup),
    }
    with open(ARTIFACT_DIR / "mapping_validation.json", "w", encoding="utf-8") as file:
        json.dump(mapping_checks, file, indent=2)

    experiment_results = {}
    threshold_tables = []
    selected_predictions = []
    for experiment_name, feature_columns in EXPERIMENT_FEATURES.items():
        frames = traffic_splits if experiment_name == "parking_plus_traffic" else parking_splits
        result, thresholds, predictions = train_experiment(
            experiment_name,
            feature_columns,
            frames,
        )
        experiment_results[experiment_name] = result
        threshold_tables.append(thresholds)
        selected_predictions.append(predictions)

    comparison = comparison_rows(experiment_results)
    summary = selected_model_summary(experiment_results)
    comparison.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)
    summary.to_csv(ARTIFACT_DIR / "selected_model_comparison.csv", index=False)
    pd.concat(threshold_tables, ignore_index=True).to_csv(
        ARTIFACT_DIR / "threshold_search.csv",
        index=False,
    )
    pd.concat(selected_predictions, ignore_index=True).to_csv(
        ARTIFACT_DIR / "selected_model_test_predictions.csv",
        index=False,
    )
    with open(ARTIFACT_DIR / "experiment_metrics.json", "w", encoding="utf-8") as file:
        json.dump(experiment_results, file, indent=2)

    print(summary.to_string(index=False))
    print(f"\nComparison artifacts: {ARTIFACT_DIR}")
    print(f"Experiment models: {MODEL_DIR}")


if __name__ == "__main__":
    main()
