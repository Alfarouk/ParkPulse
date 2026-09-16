import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.pipeline import SATURATION_THRESHOLD

CLASSIFIER_PRED_PATH = Path(
    "artifacts/traffic_context_experiments/selected_model_test_predictions.csv"
)
REGRESSION_PRED_PATH = Path("artifacts/catboost_test_predictions.csv")
METRICS_PATH = Path("artifacts/metrics.json")
THRESHOLD_PATH = Path("artifacts/traffic_context_experiments/threshold_search.csv")
DEPLOYED_RF_PATH = Path("models/random_forest_classifier.joblib")


def assert_metrics_match(actual, expected, label):
    for name, value in actual.items():
        if not np.isclose(value, float(expected[name]), rtol=1e-9, atol=1e-12):
            raise AssertionError(
                f"{label} metric mismatch for {name}: calculated {value}, stored {expected[name]}"
            )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    required_paths = [
        CLASSIFIER_PRED_PATH,
        REGRESSION_PRED_PATH,
        METRICS_PATH,
        THRESHOLD_PATH,
        DEPLOYED_RF_PATH,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required ParkPulse artifacts: {missing}")

    with METRICS_PATH.open("r", encoding="utf-8") as file_handle:
        metrics = json.load(file_handle)
    deployment = metrics["deployed_classifier"]
    if deployment["model"] != "random_forest":
        raise AssertionError("The deployed early-warning classifier must be Random Forest.")
    if deployment["experiment"] != "parking_only":
        raise AssertionError("The deployed classifier must use the parking-only experiment.")
    if deployment["model_file"] != DEPLOYED_RF_PATH.name:
        raise AssertionError("The deployed classifier model filename is inconsistent.")
    threshold = float(deployment["threshold"])

    deployed_hash = sha256(DEPLOYED_RF_PATH)

    threshold_table = pd.read_csv(THRESHOLD_PATH)
    rf_validation = threshold_table.loc[
        (threshold_table["experiment"] == "parking_only")
        & (threshold_table["model"] == "random_forest")
    ].copy()
    if rf_validation.empty:
        raise AssertionError("Parking-only Random Forest threshold search results are missing.")
    best_f1_threshold = float(
        rf_validation.loc[rf_validation["f1"].idxmax(), "threshold"]
    )
    if not np.isclose(threshold, best_f1_threshold):
        raise AssertionError(
            f"Stored deployment threshold {threshold} is not the validation maximum-F1 "
            f"threshold {best_f1_threshold}."
        )

    classifier_pred = pd.read_csv(CLASSIFIER_PRED_PATH)
    classifier_pred = classifier_pred.loc[
        (classifier_pred["experiment"] == "parking_only")
        & (classifier_pred["selected_model"] == "random_forest")
    ].copy()
    if classifier_pred.empty:
        raise AssertionError("Parking-only Random Forest test predictions are missing.")
    if not (classifier_pred["occupancy_ratio"] < SATURATION_THRESHOLD).all():
        raise AssertionError(
            "Currently saturated rows must be excluded from classifier evaluation, not labeled negative."
        )
    expected_target = (
        classifier_pred["future_occupancy_ratio"] >= SATURATION_THRESHOLD
    ).astype(int)
    if not classifier_pred["will_become_saturated_soon"].astype(int).equals(expected_target):
        raise AssertionError("Early-warning targets do not match the eligible-row target definition.")
    if not np.isclose(classifier_pred["threshold"].astype(float), threshold).all():
        raise AssertionError("Classifier prediction thresholds do not match deployment metadata.")

    probabilities = classifier_pred["early_warning_probability"].astype(float).to_numpy()
    if not np.logical_and(probabilities >= 0.0, probabilities <= 1.0).all():
        raise AssertionError("Classifier probabilities must be within [0, 1].")
    y_true = classifier_pred["will_become_saturated_soon"].astype(int).to_numpy()
    y_pred = (probabilities >= threshold).astype(int)
    classification = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
    }
    assert_metrics_match(classification, deployment["test"], "classification")

    regression_pred = pd.read_csv(REGRESSION_PRED_PATH)
    if not regression_pred["target_offset_minutes"].between(20, 40).all():
        raise AssertionError("Found regression targets outside the 20-40 minute window.")
    expected_saturated = regression_pred["occupancy_ratio"] >= SATURATION_THRESHOLD
    if not regression_pred["currently_saturated"].astype(bool).equals(expected_saturated):
        raise AssertionError("Exported saturation flags do not match occupancy_ratio >= 0.90.")
    actual_occupancy = regression_pred["future_occupancy"].astype(float).to_numpy()
    predicted_occupancy = regression_pred["predicted_future_occupancy"].astype(float).to_numpy()
    regression = {
        "mae": float(mean_absolute_error(actual_occupancy, predicted_occupancy)),
        "rmse": float(np.sqrt(mean_squared_error(actual_occupancy, predicted_occupancy))),
        "r2": float(r2_score(actual_occupancy, predicted_occupancy)),
    }
    assert_metrics_match(regression, metrics["catboost_regressor"]["test"], "regression")

    print(json.dumps({
        "deployed_classifier": "random_forest",
        "classifier_model_file": str(DEPLOYED_RF_PATH),
        "classifier_sha256": deployed_hash,
        "classifier_feature_set": deployment["experiment"],
        "validation_max_f1_threshold": threshold,
        "classifier_test_rows": len(classifier_pred),
        "classification": classification,
        "deployed_regressor": "catboost",
        "regressor_model_file": "models/catboost_regressor.cbm",
        "regression_test_rows": len(regression_pred),
        "regression": regression,
    }, indent=2))
    print(
        "\nPASS: ParkPulse deploys the fair-comparison parking-only Random Forest "
        "for early warning and keeps CatBoost for regression."
    )


if __name__ == "__main__":
    main()
