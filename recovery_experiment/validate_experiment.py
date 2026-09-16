from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import FEATURES, SATURATION_THRESHOLD, chronological_split


EXPERIMENT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts"
REQUIRED_FILES = [
    EXPERIMENT_DIR / "run_experiment.py",
    EXPERIMENT_DIR / "README.md",
    ARTIFACT_DIR / "metrics.json",
    ARTIFACT_DIR / "model_comparison.csv",
    ARTIFACT_DIR / "threshold_search.csv",
    ARTIFACT_DIR / "selected_model_test_predictions.csv",
]
EXPECTED_COUNTS = {
    "train": {"rows": 1604, "recoveries": 240},
    "validation": {"rows": 484, "recoveries": 76},
    "test": {"rows": 362, "recoveries": 65},
}
INTEGRATION_OWNED_FILES = {
    "app.py",
    "validate_project.py",
    "artifacts/model_summary.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=1e-10, atol=1e-12):
        raise AssertionError(f"{label}: {actual} != {expected}")


def main() -> None:
    missing = [str(path) for path in REQUIRED_FILES if not path.exists()]
    if missing:
        raise AssertionError(f"Missing required experiment files: {missing}")

    with (ARTIFACT_DIR / "metrics.json").open(encoding="utf-8") as handle:
        metrics = json.load(handle)

    if metrics["features"] != FEATURES:
        raise AssertionError("Recovery features differ from ParkPulse parking-only FEATURES.")
    prohibited = [
        feature
        for feature in metrics["features"]
        if "future" in feature.lower() or "target" in feature.lower()
    ]
    if prohibited:
        raise AssertionError(f"Future/target leakage features found: {prohibited}")
    if metrics["leakage_controls"]["telematics_used"]:
        raise AssertionError("Telematics must not be used in the initial recovery experiment.")
    if not metrics["selection"]["frozen_before_test_evaluation"]:
        raise AssertionError("Selection was not marked frozen before test evaluation.")

    modeled = pd.read_csv(
        PROJECT_ROOT / "artifacts" / "modeled_dataset.csv",
        parse_dates=["timestamp", "future_timestamp"],
    )
    split_data = chronological_split(modeled)
    raw_splits = {
        "train": split_data.train,
        "validation": split_data.valid,
        "test": split_data.test,
    }
    for name, part in raw_splits.items():
        saturated = part.loc[
            part["occupancy_ratio"].astype(float) >= SATURATION_THRESHOLD
        ].copy()
        target = (
            saturated["future_occupancy_ratio"].astype(float) < SATURATION_THRESHOLD
        ).astype(int)
        expected = EXPECTED_COUNTS[name]
        if len(saturated) != expected["rows"] or int(target.sum()) != expected["recoveries"]:
            raise AssertionError(f"Unexpected {name} recovery population.")
        reported = metrics["class_prevalence"][name]
        if int(reported["rows"]) != len(saturated):
            raise AssertionError(f"Reported {name} row count is incorrect.")
        if int(reported["recovery_rows"]) != int(target.sum()):
            raise AssertionError(f"Reported {name} recovery count is incorrect.")

    candidates = metrics["candidates"]
    selection_rows = []
    for candidate_id, candidate in candidates.items():
        validation = candidate["validation"]
        selection_rows.append(
            {
                "candidate_id": candidate_id,
                "selection_score": (validation["f1"] + validation["pr_auc"]) / 2.0,
                "f1": validation["f1"],
                "pr_auc": validation["pr_auc"],
                "balance": min(validation["precision"], validation["recall"]),
                "brier": validation["brier"],
            }
        )
    expected_selected = str(
        pd.DataFrame(selection_rows)
        .sort_values(
            ["selection_score", "f1", "pr_auc", "balance", "brier", "candidate_id"],
            ascending=[False, False, False, False, True, True],
        )
        .iloc[0]["candidate_id"]
    )
    selected_candidate = metrics["selection"]["selected_candidate"]
    if selected_candidate != expected_selected:
        raise AssertionError("Selected candidate is not the validation-only winner.")

    comparison = pd.read_csv(ARTIFACT_DIR / "model_comparison.csv")
    if comparison["selected"].astype(bool).sum() != 1:
        raise AssertionError("Model comparison must mark exactly one selected candidate.")
    selected_comparison = comparison.loc[comparison["selected"].astype(bool)].iloc[0]
    if selected_comparison["candidate_id"] != selected_candidate:
        raise AssertionError("Comparison selected candidate disagrees with metrics JSON.")

    threshold_table = pd.read_csv(ARTIFACT_DIR / "threshold_search.csv")
    for candidate_id, candidate in candidates.items():
        selected_rows = threshold_table.loc[
            (threshold_table["candidate_id"] == candidate_id)
            & threshold_table["is_selected_threshold"].astype(bool)
        ]
        if len(selected_rows) != 1:
            raise AssertionError(f"{candidate_id} must have one selected threshold.")
        assert_close(
            float(selected_rows.iloc[0]["threshold"]),
            float(candidate["threshold"]),
            f"{candidate_id} threshold",
        )

    predictions = pd.read_csv(
        ARTIFACT_DIR / "selected_model_test_predictions.csv",
        parse_dates=["timestamp", "future_timestamp"],
    )
    if len(predictions) != EXPECTED_COUNTS["test"]["rows"]:
        raise AssertionError("Selected-model test prediction row count is incorrect.")
    if not (predictions["occupancy_ratio"].astype(float) >= SATURATION_THRESHOLD).all():
        raise AssertionError("Test predictions contain non-saturated current rows.")
    expected_target = (
        predictions["future_occupancy_ratio"].astype(float) < SATURATION_THRESHOLD
    ).astype(int)
    if not expected_target.equals(predictions["recovery"].astype(int)):
        raise AssertionError("Exported recovery targets do not match the definition.")

    y_true = predictions["recovery"].astype(int).to_numpy()
    probabilities = predictions["recovery_probability"].astype(float).to_numpy()
    threshold = float(metrics["selection"]["selected_threshold"])
    predicted = (probabilities >= threshold).astype(int)
    if not np.array_equal(predicted, predictions["predicted_recovery"].astype(int)):
        raise AssertionError("Exported recovery decisions do not match the selected threshold.")
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    recalculated = {
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "pr_auc": average_precision_score(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "brier": brier_score_loss(y_true, probabilities),
    }
    for name, value in recalculated.items():
        assert_close(float(value), float(metrics["selected_test"][name]), f"test {name}")
    if metrics["selected_test"]["confusion_matrix"] != {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }:
        raise AssertionError("Reported test confusion matrix is incorrect.")

    changed = []
    for relative_path, expected_hash in metrics["protected_files_sha256"].items():
        # These files are expected to change only when a validated experiment is
        # deliberately integrated. Models, data, metrics, training code, and the
        # shared feature pipeline remain protected below.
        if relative_path in INTEGRATION_OWNED_FILES:
            continue
        path = PROJECT_ROOT / relative_path
        if sha256(path) != expected_hash:
            changed.append(relative_path)
    if changed:
        raise AssertionError(f"Protected deployed files changed: {changed}")

    print(json.dumps({
        "validation": "passed",
        "selected_candidate": selected_candidate,
        "selected_threshold": threshold,
        "test_recoveries_detected": int(tp),
        "test_false_recovery_alerts": int(fp),
        "protected_files_unchanged": True,
    }, indent=2))


if __name__ == "__main__":
    main()
