from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import FEATURES, SATURATION_THRESHOLD, chronological_split


EXPERIMENT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts"
MODEL_DIR = EXPERIMENT_DIR / "models"
DATA_PATH = PROJECT_ROOT / "artifacts" / "modeled_dataset.csv"
RANDOM_STATE = 42
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)
EXPECTED_COUNTS = {
    "train": {"rows": 1604, "recoveries": 240},
    "validation": {"rows": 484, "recoveries": 76},
    "test": {"rows": 362, "recoveries": 65},
}
PROTECTED_PATHS = [
    PROJECT_ROOT / "app.py",
    PROJECT_ROOT / "train.py",
    PROJECT_ROOT / "validate_project.py",
    PROJECT_ROOT / "src" / "pipeline.py",
    PROJECT_ROOT / "artifacts" / "metrics.json",
    PROJECT_ROOT / "artifacts" / "model_summary.csv",
    PROJECT_ROOT / "artifacts" / "modeled_dataset.csv",
    PROJECT_ROOT / "artifacts" / "test_demo.csv",
    PROJECT_ROOT / "artifacts" / "recovery_transition_diagnostics.csv",
    PROJECT_ROOT / "models" / "random_forest_classifier.joblib",
    PROJECT_ROOT / "models" / "catboost_regressor.cbm",
    PROJECT_ROOT
    / "artifacts"
    / "traffic_context_experiments"
    / "selected_model_comparison.csv",
    PROJECT_ROOT
    / "optimization_experiment"
    / "artifacts"
    / "experiment_results.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256(path)
        for path in PROTECTED_PATHS
    }


def load_recovery_splits() -> dict[str, pd.DataFrame]:
    modeled = pd.read_csv(DATA_PATH, parse_dates=["timestamp", "future_timestamp"])
    split_data = chronological_split(modeled)
    raw_splits = {
        "train": split_data.train,
        "validation": split_data.valid,
        "test": split_data.test,
    }
    recovery_splits: dict[str, pd.DataFrame] = {}
    for name, part in raw_splits.items():
        saturated = part.loc[
            part["occupancy_ratio"].astype(float) >= SATURATION_THRESHOLD
        ].copy()
        saturated["recovery"] = (
            saturated["future_occupancy_ratio"].astype(float) < SATURATION_THRESHOLD
        ).astype(int)
        expected = EXPECTED_COUNTS[name]
        if len(saturated) != expected["rows"]:
            raise AssertionError(
                f"{name} saturated count changed: {len(saturated)} != {expected['rows']}"
            )
        if int(saturated["recovery"].sum()) != expected["recoveries"]:
            raise AssertionError(
                f"{name} recovery count changed: "
                f"{int(saturated['recovery'].sum())} != {expected['recoveries']}"
            )
        recovery_splits[name] = saturated
    return recovery_splits


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame[FEATURES].copy()
    features["parking_id"] = features["parking_id"].astype(str)
    return features


def sklearn_preprocessor() -> ColumnTransformer:
    numeric_features = [feature for feature in FEATURES if feature != "parking_id"]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "parking_id",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["parking_id"],
            ),
        ]
    )


def build_logistic(class_weight: str | None) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", sklearn_preprocessor()),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight=class_weight,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_random_forest(class_weight: str | None) -> Pipeline:
    numeric_features = [feature for feature in FEATURES if feature != "parking_id"]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "parking_id",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["parking_id"],
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight=class_weight,
                    n_jobs=1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def probability_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int | dict[str, int]]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "rows": int(len(y_true)),
        "positive_rows": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def threshold_search(
    candidate_id: str,
    family: str,
    config: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray = THRESHOLDS,
) -> tuple[float, dict[str, object], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        metrics = probability_metrics(y_true, probabilities, float(threshold))
        confusion = metrics.pop("confusion_matrix")
        rows.append(
            {
                "candidate_id": candidate_id,
                "family": family,
                "config": config,
                **metrics,
                **confusion,
            }
        )
    table = pd.DataFrame(rows)
    table["precision_recall_balance"] = table[["precision", "recall"]].min(axis=1)
    best_index = table.sort_values(
        ["f1", "precision_recall_balance", "pr_auc", "brier", "threshold"],
        ascending=[False, False, False, True, True],
    ).index[0]
    best_threshold = float(table.loc[best_index, "threshold"])
    best_metrics = probability_metrics(y_true, probabilities, best_threshold)
    for index, row in enumerate(rows):
        row["is_selected_threshold"] = bool(index == best_index)
    return best_threshold, best_metrics, rows


def candidate_specs(positive_weight: float) -> list[dict[str, object]]:
    return [
        {
            "candidate_id": "logistic_default",
            "family": "logistic_regression",
            "config": "class_weight=None",
            "kind": "sklearn",
            "model": build_logistic(None),
        },
        {
            "candidate_id": "logistic_balanced",
            "family": "logistic_regression",
            "config": "class_weight=balanced",
            "kind": "sklearn",
            "model": build_logistic("balanced"),
        },
        {
            "candidate_id": "random_forest_default",
            "family": "random_forest",
            "config": "500 trees; min_samples_leaf=2; class_weight=None",
            "kind": "sklearn",
            "model": build_random_forest(None),
        },
        {
            "candidate_id": "random_forest_balanced",
            "family": "random_forest",
            "config": "500 trees; min_samples_leaf=2; class_weight=balanced_subsample",
            "kind": "sklearn",
            "model": build_random_forest("balanced_subsample"),
        },
        {
            "candidate_id": "catboost_default",
            "family": "catboost",
            "config": "depth=7; learning_rate=0.05; class_weights=None",
            "kind": "catboost",
            "model": CatBoostClassifier(
                iterations=700,
                depth=7,
                learning_rate=0.05,
                loss_function="Logloss",
                eval_metric="Logloss",
                random_seed=RANDOM_STATE,
                verbose=False,
                allow_writing_files=False,
            ),
        },
        {
            "candidate_id": "catboost_balanced",
            "family": "catboost",
            "config": f"depth=7; learning_rate=0.05; class_weights=[1,{positive_weight:.6f}]",
            "kind": "catboost",
            "model": CatBoostClassifier(
                iterations=700,
                depth=7,
                learning_rate=0.05,
                loss_function="Logloss",
                eval_metric="Logloss",
                class_weights=[1.0, positive_weight],
                random_seed=RANDOM_STATE,
                verbose=False,
                allow_writing_files=False,
            ),
        },
    ]


def predict_probabilities(
    model: object,
    kind: str,
    features: pd.DataFrame,
) -> np.ndarray:
    if kind == "catboost":
        pool = Pool(features, cat_features=["parking_id"])
        return np.asarray(model.predict_proba(pool)[:, 1], dtype=float)
    return np.asarray(model.predict_proba(features)[:, 1], dtype=float)


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    protected_before = protected_hashes()

    splits = load_recovery_splits()
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]
    X_train = feature_frame(train)
    X_validation = feature_frame(validation)
    X_test = feature_frame(test)
    y_train = train["recovery"].astype(int).to_numpy()
    y_validation = validation["recovery"].astype(int).to_numpy()
    y_test = test["recovery"].astype(int).to_numpy()

    positive_weight = float((len(y_train) - y_train.sum()) / y_train.sum())
    candidates: dict[str, dict[str, object]] = {}
    threshold_rows: list[dict[str, object]] = []

    baseline_probabilities = np.zeros(len(y_validation), dtype=float)
    baseline_metrics = probability_metrics(y_validation, baseline_probabilities, 0.50)
    baseline_confusion = baseline_metrics["confusion_matrix"]
    threshold_rows.append(
        {
            "candidate_id": "baseline_remains_saturated",
            "family": "baseline",
            "config": "always predict no recovery",
            **{key: value for key, value in baseline_metrics.items() if key != "confusion_matrix"},
            **baseline_confusion,
            "is_selected_threshold": True,
        }
    )
    candidates["baseline_remains_saturated"] = {
        "family": "baseline",
        "config": "always predict no recovery",
        "kind": "baseline",
        "fit_seconds": 0.0,
        "threshold": 0.50,
        "validation": baseline_metrics,
        "model": None,
    }

    for spec in candidate_specs(positive_weight):
        candidate_id = str(spec["candidate_id"])
        family = str(spec["family"])
        config = str(spec["config"])
        kind = str(spec["kind"])
        model = spec["model"]
        started = time.perf_counter()
        if kind == "catboost":
            model.fit(
                Pool(X_train, y_train, cat_features=["parking_id"]),
                eval_set=Pool(
                    X_validation,
                    y_validation,
                    cat_features=["parking_id"],
                ),
                early_stopping_rounds=80,
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        validation_probabilities = predict_probabilities(model, kind, X_validation)
        selected_threshold, validation_metrics, search_rows = threshold_search(
            candidate_id,
            family,
            config,
            y_validation,
            validation_probabilities,
        )
        threshold_rows.extend(search_rows)
        candidates[candidate_id] = {
            "family": family,
            "config": config,
            "kind": kind,
            "fit_seconds": fit_seconds,
            "threshold": selected_threshold,
            "validation": validation_metrics,
            "model": model,
        }

    # Freeze model selection using validation results only. No test probabilities
    # or labels have been accessed above this point.
    selection_rows = []
    for candidate_id, candidate in candidates.items():
        validation_metrics = candidate["validation"]
        precision = float(validation_metrics["precision"])
        recall = float(validation_metrics["recall"])
        f1 = float(validation_metrics["f1"])
        pr_auc = float(validation_metrics["pr_auc"])
        selection_rows.append(
            {
                "candidate_id": candidate_id,
                "selection_score": (f1 + pr_auc) / 2.0,
                "f1": f1,
                "pr_auc": pr_auc,
                "precision_recall_balance": min(precision, recall),
                "brier": float(validation_metrics["brier"]),
            }
        )
    selected_candidate = str(
        pd.DataFrame(selection_rows)
        .sort_values(
            [
                "selection_score",
                "f1",
                "pr_auc",
                "precision_recall_balance",
                "brier",
                "candidate_id",
            ],
            ascending=[False, False, False, False, True, True],
        )
        .iloc[0]["candidate_id"]
    )
    selection_frozen_before_test = True

    # The untouched test set is evaluated only after the selected candidate and
    # threshold have been frozen above. Test results never affect selection.
    for candidate_id, candidate in candidates.items():
        if candidate["kind"] == "baseline":
            test_probabilities = np.zeros(len(y_test), dtype=float)
        else:
            test_probabilities = predict_probabilities(
                candidate["model"], str(candidate["kind"]), X_test
            )
        candidate["test_probabilities"] = test_probabilities
        candidate["test"] = probability_metrics(
            y_test,
            test_probabilities,
            float(candidate["threshold"]),
        )

    selected = candidates[selected_candidate]
    selected_threshold = float(selected["threshold"])
    selected_probabilities = np.asarray(selected["test_probabilities"], dtype=float)
    selected_predictions = (selected_probabilities >= selected_threshold).astype(int)
    prediction_export = test[
        [
            "parking_id",
            "timestamp",
            "capacity",
            "occupancy",
            "occupancy_ratio",
            "future_timestamp",
            "future_occupancy",
            "future_occupancy_ratio",
            "target_offset_minutes",
        ]
    ].copy()
    prediction_export["recovery"] = y_test
    prediction_export["recovery_probability"] = selected_probabilities
    prediction_export["predicted_recovery"] = selected_predictions
    prediction_export.to_csv(
        ARTIFACT_DIR / "selected_model_test_predictions.csv", index=False
    )

    selected_model_file: str | None = None
    if selected["kind"] == "catboost":
        selected_model_file = f"models/{selected_candidate}.cbm"
        selected["model"].save_model(EXPERIMENT_DIR / selected_model_file)
    elif selected["kind"] == "sklearn":
        selected_model_file = f"models/{selected_candidate}.joblib"
        joblib.dump(selected["model"], EXPERIMENT_DIR / selected_model_file)

    comparison_rows = []
    for candidate_id, candidate in candidates.items():
        row: dict[str, object] = {
            "candidate_id": candidate_id,
            "family": candidate["family"],
            "config": candidate["config"],
            "fit_seconds": candidate["fit_seconds"],
            "threshold": candidate["threshold"],
            "selected": candidate_id == selected_candidate,
        }
        for split_name in ["validation", "test"]:
            split_metrics = candidate[split_name]
            for metric_name in [
                "rows",
                "positive_rows",
                "prevalence",
                "precision",
                "recall",
                "f1",
                "pr_auc",
                "roc_auc",
                "brier",
            ]:
                row[f"{split_name}_{metric_name}"] = split_metrics[metric_name]
            for cell, value in split_metrics["confusion_matrix"].items():
                row[f"{split_name}_{cell}"] = value
        comparison_rows.append(row)
    pd.DataFrame(comparison_rows).sort_values(
        ["selected", "validation_f1", "validation_pr_auc"],
        ascending=[False, False, False],
    ).to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)

    pd.DataFrame(threshold_rows).to_csv(
        ARTIFACT_DIR / "threshold_search.csv", index=False
    )

    protected_after = protected_hashes()
    if protected_before != protected_after:
        changed = [
            path
            for path in protected_before
            if protected_before[path] != protected_after[path]
        ]
        raise AssertionError(f"Protected deployed files changed: {changed}")

    serializable_candidates = {}
    for candidate_id, candidate in candidates.items():
        serializable_candidates[candidate_id] = {
            key: value
            for key, value in candidate.items()
            if key not in {"model", "test_probabilities"}
        }
    metrics = {
        "objective": (
            "Predict recovery below 90% approximately 30 minutes later among "
            "currently saturated rows only."
        ),
        "features": FEATURES,
        "leakage_controls": {
            "eligible_rows": "occupancy_ratio >= 0.90",
            "target": "future_occupancy_ratio < 0.90",
            "split": "existing chronological train/validation/test split",
            "selection_data": "validation only",
            "test_role": "evaluated only after candidate and threshold were frozen",
            "telematics_used": False,
        },
        "class_prevalence": {
            name: {
                "rows": int(len(part)),
                "recovery_rows": int(part["recovery"].sum()),
                "prevalence": float(part["recovery"].mean()),
            }
            for name, part in splits.items()
        },
        "train_positive_class_weight": positive_weight,
        "selection": {
            "rule": (
                "maximize the mean of validation F1 and validation PR-AUC; then "
                "F1, PR-AUC, minimum of precision/recall, lower Brier score"
            ),
            "selected_candidate": selected_candidate,
            "selected_family": selected["family"],
            "selected_config": selected["config"],
            "selected_threshold": selected_threshold,
            "selected_model_file": selected_model_file,
            "frozen_before_test_evaluation": selection_frozen_before_test,
        },
        "candidates": serializable_candidates,
        "selected_validation": selected["validation"],
        "selected_test": selected["test"],
        "protected_files_sha256": protected_before,
    }
    with (ARTIFACT_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps({
        "selected_candidate": selected_candidate,
        "selected_threshold": selected_threshold,
        "validation": selected["validation"],
        "test": selected["test"],
        "protected_files_unchanged": True,
    }, indent=2))


if __name__ == "__main__":
    main()
