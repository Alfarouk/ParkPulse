from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts"
SUBMISSION_MUTABLE_FILES = {"app.py", "artifacts\\model_summary.csv"}
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(target: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(target, prediction)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probability)),
        "pr_auc": float(average_precision_score(target, probability)),
        "brier": float(brier_score_loss(target, probability)),
    }


def assert_metrics(actual: dict[str, float], expected: dict[str, float], label: str) -> None:
    for name, value in actual.items():
        if not np.isclose(value, float(expected[name]), rtol=1e-9, atol=1e-12):
            raise AssertionError(
                f"{label} {name} mismatch: calculated {value}, stored {expected[name]}"
            )


def main() -> None:
    with (ARTIFACT_DIR / "experiment_results.json").open(encoding="utf-8") as handle:
        results = json.load(handle)
    with (ARTIFACT_DIR / "frozen_selection.json").open(encoding="utf-8") as handle:
        frozen = json.load(handle)
    with (ARTIFACT_DIR / "deployment_preservation_check.json").open(encoding="utf-8") as handle:
        preservation = json.load(handle)

    if not frozen["frozen_before_test_evaluation"]:
        raise AssertionError("Winning configuration was not frozen before test evaluation.")
    comparison = pd.read_csv(ARTIFACT_DIR / "rolling_model_comparison.csv")
    if comparison.iloc[0]["candidate_id"] != frozen["selected_candidate"]:
        raise AssertionError("Frozen candidate is not the rolling-validation winner.")
    if comparison.iloc[0]["feature_set"] != frozen["feature_set"]:
        raise AssertionError("Frozen feature set is not the rolling-validation winner's feature set.")

    folds = pd.read_csv(ARTIFACT_DIR / "rolling_fold_summary.csv", parse_dates=["validation_end"])
    predictions = pd.read_csv(
        ARTIFACT_DIR / "untouched_test_predictions.csv", parse_dates=["timestamp"]
    )
    if not (folds["validation_end"].max() < predictions["timestamp"].min()):
        raise AssertionError("Rolling validation overlaps the untouched test period.")

    target = predictions["will_become_saturated_soon"].astype(int).to_numpy()
    baseline_actual = metrics(
        target,
        predictions["baseline_probability"].astype(float).to_numpy(),
        float(predictions["baseline_threshold"].iloc[0]),
    )
    winner_actual = metrics(
        target,
        predictions["winning_probability"].astype(float).to_numpy(),
        float(predictions["winning_threshold"].iloc[0]),
    )
    assert_metrics(baseline_actual, results["untouched_test"]["baseline"], "baseline test")
    assert_metrics(winner_actual, results["untouched_test"]["winner"], "winner test")

    protected_hashes = {
        relative_path: expected_hash
        for relative_path, expected_hash in preservation["protected_hashes_after"].items()
        if relative_path not in SUBMISSION_MUTABLE_FILES
    }
    current_hashes = {
        relative_path: sha256(ROOT / relative_path)
        for relative_path in protected_hashes
    }
    if current_hashes != protected_hashes:
        raise AssertionError("A protected deployment file changed after the experiment.")
    if not preservation["all_unchanged"]:
        raise AssertionError("Experiment reports modified deployment files.")

    feasibility = results["target_feasibility"]
    if feasibility["supports_crosses_within_30_target"]:
        raise AssertionError("Sparse-cadence target conclusion is inconsistent.")
    if results["recommend_replacement"]:
        raise AssertionError("Replacement recommendation contradicts held-out results.")

    print(json.dumps({
        "selected_candidate": frozen["selected_candidate"],
        "feature_set": frozen["feature_set"],
        "threshold": frozen["threshold"],
        "test_rows": len(predictions),
        "baseline_test": baseline_actual,
        "winner_test": winner_actual,
        "recommend_replacement": results["recommend_replacement"],
        "protected_deployment_unchanged": True,
    }, indent=2))
    print("\nPASS: optimization experiment is internally consistent and deployment is unchanged.")


if __name__ == "__main__":
    main()
