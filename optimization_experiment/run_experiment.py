from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import (
    FEATURES,
    SATURATION_THRESHOLD,
    add_lag_features,
    build_random_forest_classifier,
    build_future_target,
    choose_threshold_for_f1,
    chronological_split,
    classification_metrics,
    early_warning_candidates,
    load_and_clean_csv,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts"
MODEL_DIR = EXPERIMENT_DIR / "models"
DATA_PATH = ROOT / "data" / "dataset.csv"
DEPLOYED_RF_PATH = ROOT / "models" / "random_forest_classifier.joblib"
DEPLOYED_METRICS_PATH = ROOT / "artifacts" / "metrics.json"
RANDOM_STATE = 42

DISTANCE_CHANGE_FEATURES = [
    "distance_to_90",
    "occupancy_change_30",
    "occupancy_change_60",
    "occupancy_change_90",
]
TREND_FEATURES = [
    "rolling_slope_60",
    "rolling_slope_90",
    "occupancy_acceleration",
    "rolling_std_60",
    "rolling_std_90",
    "estimated_minutes_to_90",
]
HISTORY_FEATURES = [
    "garage_slot_hist_mean",
    "garage_slot_hist_median",
    "garage_slot_hist_std",
    "garage_slot_hist_p90",
    "garage_slot_hist_count",
]
FEATURE_SETS = {
    "existing": list(FEATURES),
    "distance_changes": [*FEATURES, *DISTANCE_CHANGE_FEATURES],
    "dynamics": [*FEATURES, *DISTANCE_CHANGE_FEATURES, *TREND_FEATURES],
    "dynamics_history": [
        *FEATURES,
        *DISTANCE_CHANGE_FEATURES,
        *TREND_FEATURES,
        *HISTORY_FEATURES,
    ],
}
CLASSIFICATION_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier",
)


@dataclass
class CandidateSpec:
    candidate_id: str
    family: str
    feature_set: str
    params: dict[str, Any]
    is_new: bool = True


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, allow_nan=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rowwise_slope(frame: pd.DataFrame, value_columns: list[str], minute_columns: list[str]) -> np.ndarray:
    values = frame[value_columns].to_numpy(dtype=float)
    minutes = frame[minute_columns].to_numpy(dtype=float)
    output = np.full(len(frame), np.nan, dtype=float)
    for index, (row_values, row_minutes) in enumerate(zip(values, minutes)):
        valid = np.isfinite(row_values) & np.isfinite(row_minutes)
        if valid.sum() < 2:
            continue
        x = row_minutes[valid]
        y = row_values[valid]
        centered = x - x.mean()
        denominator = np.square(centered).sum()
        if denominator > 0:
            output[index] = float((centered * (y - y.mean())).sum() / denominator)
    return output


def rowwise_std(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].to_numpy(dtype=float)
    counts = np.isfinite(values).sum(axis=1)
    output = np.nanstd(values, axis=1, ddof=0)
    output[counts < 2] = np.nan
    return output


def add_candidate_dynamics(raw: pd.DataFrame) -> pd.DataFrame:
    out = add_lag_features(raw).copy()
    grouped = out.groupby("parking_id", group_keys=False)

    out["lag90_ratio"] = grouped["occupancy_ratio"].shift(3)
    out["lag30_time"] = grouped["timestamp"].shift(1)
    out["lag60_time"] = grouped["timestamp"].shift(2)
    out["lag90_time"] = grouped["timestamp"].shift(3)
    out["minutes_from_lag30"] = (
        out["timestamp"] - out["lag30_time"]
    ).dt.total_seconds() / 60.0
    out["minutes_from_lag60"] = (
        out["timestamp"] - out["lag60_time"]
    ).dt.total_seconds() / 60.0
    out["minutes_from_lag90"] = (
        out["timestamp"] - out["lag90_time"]
    ).dt.total_seconds() / 60.0

    out["lag90_ratio"] = out["lag90_ratio"].where(
        out["minutes_from_lag90"].between(75, 105)
    )
    out["distance_to_90"] = SATURATION_THRESHOLD - out["occupancy_ratio"]
    out["occupancy_change_30"] = out["occupancy_ratio"] - out["lag30_ratio"]
    out["occupancy_change_60"] = out["occupancy_ratio"] - out["lag60_ratio"]
    out["occupancy_change_90"] = out["occupancy_ratio"] - out["lag90_ratio"]

    out["minute_current"] = 0.0
    out["minute_lag30"] = -out["minutes_from_lag30"]
    out["minute_lag60"] = -out["minutes_from_lag60"]
    out["minute_lag90"] = -out["minutes_from_lag90"]
    out["rolling_slope_60"] = rowwise_slope(
        out,
        ["lag60_ratio", "lag30_ratio", "occupancy_ratio"],
        ["minute_lag60", "minute_lag30", "minute_current"],
    )
    out["rolling_slope_90"] = rowwise_slope(
        out,
        ["lag90_ratio", "lag60_ratio", "lag30_ratio", "occupancy_ratio"],
        ["minute_lag90", "minute_lag60", "minute_lag30", "minute_current"],
    )
    out["rolling_std_60"] = rowwise_std(
        out, ["lag60_ratio", "lag30_ratio", "occupancy_ratio"]
    )
    out["rolling_std_90"] = rowwise_std(
        out, ["lag90_ratio", "lag60_ratio", "lag30_ratio", "occupancy_ratio"]
    )

    recent_rate = out["occupancy_change_30"] / out["minutes_from_lag30"]
    prior_interval = out["minutes_from_lag60"] - out["minutes_from_lag30"]
    prior_rate = (out["lag30_ratio"] - out["lag60_ratio"]) / prior_interval
    valid_acceleration = (
        out["minutes_from_lag30"].between(20, 40)
        & prior_interval.between(20, 40)
    )
    out["occupancy_acceleration"] = (recent_rate - prior_rate).where(valid_acceleration)

    positive_growth = out["rolling_slope_60"].where(out["rolling_slope_60"] > 1e-6)
    minutes_to_90 = out["distance_to_90"] / positive_growth
    out["estimated_minutes_to_90"] = minutes_to_90.clip(lower=0, upper=999).fillna(999.0)

    minute_of_day = out["timestamp"].dt.hour * 60 + out["timestamp"].dt.minute
    out["weekday_time_slot"] = out["weekday"] * 48 + (minute_of_day // 30).astype(int)
    helper_columns = [
        "lag30_time",
        "lag60_time",
        "lag90_time",
        "minute_current",
        "minute_lag30",
        "minute_lag60",
        "minute_lag90",
    ]
    return out.drop(columns=helper_columns)


def add_past_only_training_history(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values("timestamp").copy()
    keys = ["parking_id", "weekday_time_slot"]
    grouped = out.groupby(keys, sort=False)["occupancy_ratio"]
    out["garage_slot_hist_count"] = out.groupby(keys, sort=False).cumcount().astype(float)
    out["garage_slot_hist_mean"] = grouped.transform(
        lambda series: series.shift(1).expanding(min_periods=1).mean()
    )
    out["garage_slot_hist_median"] = grouped.transform(
        lambda series: series.shift(1).expanding(min_periods=1).median()
    )
    out["garage_slot_hist_std"] = grouped.transform(
        lambda series: series.shift(1).expanding(min_periods=2).std(ddof=0)
    )
    out["garage_slot_hist_p90"] = grouped.transform(
        lambda series: series.shift(1).expanding(min_periods=1).quantile(0.90)
    )
    return out


def history_lookup(training_frame: pd.DataFrame) -> pd.DataFrame:
    return (
        training_frame.groupby(["parking_id", "weekday_time_slot"], observed=True)[
            "occupancy_ratio"
        ]
        .agg(
            garage_slot_hist_mean="mean",
            garage_slot_hist_median="median",
            garage_slot_hist_std=lambda values: values.std(ddof=0),
            garage_slot_hist_p90=lambda values: values.quantile(0.90),
            garage_slot_hist_count="count",
        )
        .reset_index()
    )


def apply_training_history(eval_frame: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    out = eval_frame.drop(columns=HISTORY_FEATURES, errors="ignore").copy()
    return out.merge(lookup, on=["parking_id", "weekday_time_slot"], how="left")


def enrich_fold(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Historical context uses every earlier parking observation. Saturated rows are
    # removed only after their occupancy has contributed to past-only aggregates.
    train_with_history = add_past_only_training_history(train)
    lookup = history_lookup(train)
    validation_with_history = apply_training_history(validation, lookup)
    return (
        early_warning_candidates(train_with_history),
        early_warning_candidates(validation_with_history),
    )


def chronological_rolling_folds(modeled: pd.DataFrame) -> list[dict[str, Any]]:
    unique_times = np.array(sorted(modeled["timestamp"].dropna().unique()))
    boundaries = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.85)]
    folds = []
    for number, (train_fraction, valid_fraction) in enumerate(boundaries, start=1):
        train_index = max(1, int(len(unique_times) * train_fraction) - 1)
        valid_index = max(train_index + 1, int(len(unique_times) * valid_fraction) - 1)
        train_end = pd.Timestamp(unique_times[train_index])
        valid_end = pd.Timestamp(unique_times[min(valid_index, len(unique_times) - 1)])
        train = modeled.loc[modeled["future_timestamp"] <= train_end].copy()
        validation = modeled.loc[
            (modeled["timestamp"] > train_end)
            & (modeled["future_timestamp"] <= valid_end)
        ].copy()
        train, validation = enrich_fold(train, validation)
        if train["will_become_saturated_soon"].nunique() < 2:
            raise ValueError(f"Rolling fold {number} training data has only one class.")
        if validation["will_become_saturated_soon"].nunique() < 2:
            raise ValueError(f"Rolling fold {number} validation data has only one class.")
        folds.append({
            "fold": number,
            "train_end": train_end,
            "validation_end": valid_end,
            "train": train,
            "validation": validation,
        })
    return folds


def feature_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    matrix = frame[feature_columns].copy()
    matrix["parking_id"] = matrix["parking_id"].astype(str)
    return matrix


def build_rf(feature_columns: list[str], params: dict[str, Any]):
    model = build_random_forest_classifier(
        random_state=RANDOM_STATE, feature_columns=feature_columns
    )
    model.set_params(
        model__n_estimators=params.get("n_estimators", 400),
        model__min_samples_leaf=params.get("min_samples_leaf", 2),
        model__max_depth=params.get("max_depth"),
        model__max_features=params.get("max_features", "sqrt"),
        model__class_weight=params.get("class_weight"),
        model__n_jobs=1,
    )
    return model


def build_catboost(params: dict[str, Any]) -> CatBoostClassifier:
    settings: dict[str, Any] = {
        "iterations": 700,
        "depth": params.get("depth", 7),
        "learning_rate": params.get("learning_rate", 0.05),
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "random_seed": RANDOM_STATE,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": 1,
    }
    if params.get("class_weights") is not None:
        settings["class_weights"] = params["class_weights"]
    if params.get("auto_class_weights") is not None:
        settings["auto_class_weights"] = params["auto_class_weights"]
    return CatBoostClassifier(**settings)


def fit_predict(
    spec: CandidateSpec,
    feature_columns: list[str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[Any, np.ndarray, float]:
    x_train = feature_matrix(train, feature_columns)
    x_validation = feature_matrix(validation, feature_columns)
    y_train = train["will_become_saturated_soon"].astype(int)
    y_validation = validation["will_become_saturated_soon"].astype(int)
    start = time.perf_counter()

    if spec.family == "random_forest":
        model = build_rf(feature_columns, spec.params)
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_validation)[:, 1]
    elif spec.family == "catboost":
        model = build_catboost(spec.params)
        train_pool = Pool(x_train, y_train, cat_features=["parking_id"])
        validation_pool = Pool(x_validation, y_validation, cat_features=["parking_id"])
        model.fit(train_pool, eval_set=validation_pool, early_stopping_rounds=80)
        probabilities = model.predict_proba(validation_pool)[:, 1]
    elif spec.family == "balanced_random_forest":
        from imblearn.ensemble import BalancedRandomForestClassifier

        model = build_rf(feature_columns, spec.params)
        model.steps[-1] = (
            "model",
            BalancedRandomForestClassifier(
                n_estimators=spec.params.get("n_estimators", 400),
                min_samples_leaf=spec.params.get("min_samples_leaf", 2),
                max_depth=spec.params.get("max_depth"),
                sampling_strategy="all",
                replacement=True,
                n_jobs=1,
                random_state=RANDOM_STATE,
            ),
        )
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_validation)[:, 1]
    elif spec.family == "xgboost":
        from xgboost import XGBClassifier

        model = build_rf(feature_columns, spec.params)
        model.steps[-1] = (
            "model",
            XGBClassifier(
                n_estimators=spec.params.get("n_estimators", 500),
                max_depth=spec.params.get("max_depth", 6),
                learning_rate=spec.params.get("learning_rate", 0.05),
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=spec.params["scale_pos_weight"],
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        )
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_validation)[:, 1]
    else:
        raise ValueError(f"Unsupported family: {spec.family}")

    return model, probabilities, time.perf_counter() - start


def evaluate_candidate(
    spec: CandidateSpec,
    folds: list[dict[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    feature_columns = FEATURE_SETS[spec.feature_set]
    fold_rows = []
    threshold_rows = []
    for fold in folds:
        _, probabilities, fit_seconds = fit_predict(
            spec,
            feature_columns,
            fold["train"],
            fold["validation"],
        )
        target = fold["validation"]["will_become_saturated_soon"].astype(int)
        threshold, table = choose_threshold_for_f1(target, probabilities)
        metrics = classification_metrics(target, probabilities, threshold)
        fold_rows.append({
            "candidate_id": spec.candidate_id,
            "family": spec.family,
            "feature_set": spec.feature_set,
            "fold": fold["fold"],
            "train_end": fold["train_end"],
            "validation_end": fold["validation_end"],
            "train_rows": len(fold["train"]),
            "validation_rows": len(fold["validation"]),
            "validation_positives": int(target.sum()),
            "fit_seconds": fit_seconds,
            **metrics,
        })
        table = table.copy()
        table.insert(0, "candidate_id", spec.candidate_id)
        table.insert(1, "fold", fold["fold"])
        threshold_rows.append(table)

    fold_frame = pd.DataFrame(fold_rows)
    summary = {
        "candidate_id": spec.candidate_id,
        "family": spec.family,
        "feature_set": spec.feature_set,
        "is_new": spec.is_new,
        "params": json.dumps(json_ready(spec.params), sort_keys=True),
        "mean_precision": float(fold_frame["precision"].mean()),
        "mean_recall": float(fold_frame["recall"].mean()),
        "mean_f1": float(fold_frame["f1"].mean()),
        "mean_pr_auc": float(fold_frame["pr_auc"].mean()),
        "mean_roc_auc": float(fold_frame["roc_auc"].mean()),
        "mean_brier": float(fold_frame["brier"].mean()),
        "mean_fit_seconds": float(fold_frame["fit_seconds"].mean()),
    }
    summary["selection_score"] = (summary["mean_f1"] + summary["mean_pr_auc"]) / 2.0
    return summary, fold_frame, pd.concat(threshold_rows, ignore_index=True)


def selection_key(row: dict[str, Any] | pd.Series) -> tuple[float, float, float, float, float]:
    return (
        float(row["selection_score"]),
        float(row["mean_pr_auc"]),
        float(row["mean_f1"]),
        float(row["mean_precision"]),
        float(row["mean_recall"]),
    )


def candidate_specs(selected_feature_set: str, availability: dict[str, bool], imbalance_ratio: float) -> list[CandidateSpec]:
    specs = [
        CandidateSpec(
            candidate_id="existing_random_forest",
            family="random_forest",
            feature_set="existing",
            params={
                "n_estimators": 400,
                "min_samples_leaf": 2,
                "max_depth": None,
                "max_features": "sqrt",
                "class_weight": None,
            },
            is_new=False,
        ),
        CandidateSpec(
            candidate_id="rf_default_selected_features",
            family="random_forest",
            feature_set=selected_feature_set,
            params={
                "n_estimators": 400,
                "min_samples_leaf": 2,
                "max_depth": None,
                "max_features": "sqrt",
                "class_weight": None,
            },
        ),
    ]

    weights = {
        "none": None,
        "balanced": "balanced",
        "balanced_subsample": "balanced_subsample",
        "positive_2": {0: 1.0, 1: 2.0},
        "positive_4": {0: 1.0, 1: 4.0},
    }
    for weight_name, class_weight in weights.items():
        for min_leaf in (1, 2, 4):
            for max_depth in (None, 12):
                specs.append(CandidateSpec(
                    candidate_id=f"rf_{weight_name}_leaf{min_leaf}_depth{max_depth or 'none'}",
                    family="random_forest",
                    feature_set=selected_feature_set,
                    params={
                        "n_estimators": 500,
                        "min_samples_leaf": min_leaf,
                        "max_depth": max_depth,
                        "max_features": "sqrt",
                        "class_weight": class_weight,
                    },
                ))

    catboost_weights = [
        ("none", {"class_weights": None}),
        ("positive_2", {"class_weights": [1.0, 2.0]}),
        ("positive_4", {"class_weights": [1.0, 4.0]}),
        ("positive_8", {"class_weights": [1.0, 8.0]}),
        ("auto_balanced", {"auto_class_weights": "Balanced"}),
    ]
    for name, params in catboost_weights:
        specs.append(CandidateSpec(
            candidate_id=f"catboost_{name}",
            family="catboost",
            feature_set=selected_feature_set,
            params={"depth": 7, "learning_rate": 0.05, **params},
        ))

    if availability["imbalanced_learn"]:
        for min_leaf in (1, 2, 4):
            specs.append(CandidateSpec(
                candidate_id=f"balanced_random_forest_leaf{min_leaf}",
                family="balanced_random_forest",
                feature_set=selected_feature_set,
                params={
                    "n_estimators": 500,
                    "min_samples_leaf": min_leaf,
                    "max_depth": None,
                },
            ))

    if availability["xgboost"]:
        xgb_weights = sorted({1.0, math.sqrt(imbalance_ratio), imbalance_ratio / 2.0, imbalance_ratio})
        for index, scale_weight in enumerate(xgb_weights, start=1):
            specs.append(CandidateSpec(
                candidate_id=f"xgboost_weight{index}",
                family="xgboost",
                feature_set=selected_feature_set,
                params={
                    "n_estimators": 500,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "scale_pos_weight": float(scale_weight),
                },
            ))
    return specs


def fit_final_candidate(
    spec: CandidateSpec,
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[Any, np.ndarray, float, dict[str, float]]:
    model, probabilities, _ = fit_predict(
        spec, FEATURE_SETS[spec.feature_set], train, validation
    )
    target = validation["will_become_saturated_soon"].astype(int)
    threshold, _ = choose_threshold_for_f1(target, probabilities)
    metrics = classification_metrics(target, probabilities, threshold)
    return model, probabilities, threshold, metrics


def predict_model(model: Any, family: str, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    matrix = feature_matrix(frame, feature_columns)
    if family == "catboost":
        return model.predict_proba(Pool(matrix, cat_features=["parking_id"]))[:, 1]
    return model.predict_proba(matrix)[:, 1]


def save_model(model: Any, spec: CandidateSpec) -> str:
    if spec.family == "catboost":
        path = MODEL_DIR / f"{spec.candidate_id}.cbm"
        model.save_model(path)
    else:
        path = MODEL_DIR / f"{spec.candidate_id}.joblib"
        joblib.dump(model, path)
    return str(path.relative_to(EXPERIMENT_DIR))


def paired_cluster_bootstrap(
    frame: pd.DataFrame,
    baseline_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    baseline_threshold: float,
    candidate_threshold: float,
    iterations: int = 1000,
) -> dict[str, dict[str, float]]:
    y = frame["will_become_saturated_soon"].astype(int).to_numpy()
    clusters = frame["timestamp"].dt.date.astype(str).to_numpy()
    unique_clusters = np.unique(clusters)
    rng = np.random.default_rng(RANDOM_STATE)
    deltas = {name: [] for name in ("precision", "recall", "f1", "pr_auc")}

    for _ in range(iterations):
        sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        sampled_indices = np.concatenate([
            np.flatnonzero(clusters == cluster) for cluster in sampled_clusters
        ])
        sampled_y = y[sampled_indices]
        if sampled_y.min() == sampled_y.max():
            continue
        base_probability = baseline_probabilities[sampled_indices]
        new_probability = candidate_probabilities[sampled_indices]
        base_prediction = (base_probability >= baseline_threshold).astype(int)
        new_prediction = (new_probability >= candidate_threshold).astype(int)
        deltas["precision"].append(
            precision_score(sampled_y, new_prediction, zero_division=0)
            - precision_score(sampled_y, base_prediction, zero_division=0)
        )
        deltas["recall"].append(
            recall_score(sampled_y, new_prediction, zero_division=0)
            - recall_score(sampled_y, base_prediction, zero_division=0)
        )
        deltas["f1"].append(
            f1_score(sampled_y, new_prediction, zero_division=0)
            - f1_score(sampled_y, base_prediction, zero_division=0)
        )
        deltas["pr_auc"].append(
            average_precision_score(sampled_y, new_probability)
            - average_precision_score(sampled_y, base_probability)
        )

    return {
        metric: {
            "mean_delta": float(np.mean(values)),
            "ci_2_5": float(np.quantile(values, 0.025)),
            "ci_97_5": float(np.quantile(values, 0.975)),
            "bootstrap_samples": len(values),
            "cluster_unit": "calendar_day",
        }
        for metric, values in deltas.items()
    }


def investigate_target_feasibility(raw: pd.DataFrame, modeled: pd.DataFrame) -> dict[str, Any]:
    ordered = raw.sort_values(["parking_id", "timestamp"]).copy()
    intervals = (
        ordered.groupby("parking_id")["timestamp"].diff().dt.total_seconds().div(60).dropna()
    )
    non_overnight = intervals.loc[intervals <= 120]

    eligible = early_warning_candidates(modeled)[
        ["parking_id", "timestamp", "will_become_saturated_soon"]
    ].copy()
    counts = np.zeros(len(eligible), dtype=int)
    any_saturated = np.zeros(len(eligible), dtype=bool)
    by_garage = {
        str(parking_id): group.sort_values("timestamp")
        for parking_id, group in ordered.groupby("parking_id")
    }
    for row_number, row in enumerate(eligible.itertuples(index=False)):
        garage = by_garage[str(row.parking_id)]
        timestamps = garage["timestamp"].to_numpy(dtype="datetime64[ns]")
        start = np.searchsorted(timestamps, np.datetime64(row.timestamp), side="right")
        end = np.searchsorted(
            timestamps,
            np.datetime64(row.timestamp + pd.Timedelta(minutes=30)),
            side="right",
        )
        observations = garage.iloc[start:end]
        counts[row_number] = len(observations)
        if len(observations):
            any_saturated[row_number] = bool(
                (observations["occupancy"] / observations["capacity"] >= SATURATION_THRESHOLD).any()
            )

    observable = counts > 0
    current_target = eligible["will_become_saturated_soon"].astype(bool).to_numpy()
    agreement = current_target[observable] == any_saturated[observable]
    result = {
        "raw_rows": len(raw),
        "garages": int(raw["parking_id"].nunique()),
        "interval_minutes": {
            "median_all": float(intervals.median()),
            "median_non_overnight": float(non_overnight.median()),
            "p10_non_overnight": float(non_overnight.quantile(0.10)),
            "p90_non_overnight": float(non_overnight.quantile(0.90)),
            "fraction_at_most_15": float((non_overnight <= 15).mean()),
            "fraction_at_most_30": float((non_overnight <= 30).mean()),
            "fraction_between_20_and_40": float(non_overnight.between(20, 40).mean()),
        },
        "eligible_modeled_rows": len(eligible),
        "fraction_with_observation_within_30": float(observable.mean()),
        "fraction_with_two_or_more_observations_within_30": float((counts >= 2).mean()),
        "observable_rows": int(observable.sum()),
        "current_vs_any_within_30_agreement_on_observable_rows": float(agreement.mean()),
        "current_positive_alt_negative": int((current_target[observable] & ~any_saturated[observable]).sum()),
        "current_negative_alt_positive": int((~current_target[observable] & any_saturated[observable]).sum()),
    }
    supports = (
        result["fraction_with_observation_within_30"] >= 0.95
        and result["fraction_with_two_or_more_observations_within_30"] >= 0.50
    )
    result["supports_crosses_within_30_target"] = supports
    result["conclusion"] = (
        "Sampling supports observing within-window crossings."
        if supports
        else "Sampling is too sparse to reliably observe every crossing within 30 minutes; "
        "the alternative target would mostly describe the next observation, not the crossing time."
    )
    return result


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    protected_files = [
        DEPLOYED_RF_PATH,
        ROOT / "models" / "catboost_regressor.cbm",
        ROOT / "app.py",
        DEPLOYED_METRICS_PATH,
        ROOT / "artifacts" / "model_summary.csv",
    ]
    protected_before = {str(path.relative_to(ROOT)): sha256(path) for path in protected_files}

    availability = {
        "imbalanced_learn": importlib.util.find_spec("imblearn") is not None,
        "xgboost": importlib.util.find_spec("xgboost") is not None,
    }
    write_json(ARTIFACT_DIR / "library_availability.json", availability)

    raw = load_and_clean_csv(DATA_PATH)
    dynamics = add_candidate_dynamics(raw)
    modeled = build_future_target(dynamics)
    target_feasibility = investigate_target_feasibility(raw, modeled)
    write_json(ARTIFACT_DIR / "target_feasibility.json", target_feasibility)

    folds = chronological_rolling_folds(modeled)
    fold_summary = [{
        "fold": fold["fold"],
        "train_end": fold["train_end"],
        "validation_end": fold["validation_end"],
        "train_rows": len(fold["train"]),
        "validation_rows": len(fold["validation"]),
        "train_positives": int(fold["train"]["will_become_saturated_soon"].sum()),
        "validation_positives": int(fold["validation"]["will_become_saturated_soon"].sum()),
    } for fold in folds]
    pd.DataFrame(fold_summary).to_csv(ARTIFACT_DIR / "rolling_fold_summary.csv", index=False)

    print("Feature selection with expanding chronological validation folds", flush=True)
    feature_summaries = []
    feature_fold_rows = []
    feature_threshold_rows = []
    for feature_set in FEATURE_SETS:
        spec = CandidateSpec(
            candidate_id=f"feature_probe_{feature_set}",
            family="random_forest",
            feature_set=feature_set,
            params={
                "n_estimators": 400,
                "min_samples_leaf": 2,
                "max_depth": None,
                "max_features": "sqrt",
                "class_weight": None,
            },
            is_new=feature_set != "existing",
        )
        summary, fold_rows, threshold_rows = evaluate_candidate(spec, folds)
        feature_summaries.append(summary)
        feature_fold_rows.append(fold_rows)
        feature_threshold_rows.append(threshold_rows)
        print(
            f"  {feature_set}: PR-AUC={summary['mean_pr_auc']:.4f}, "
            f"F1={summary['mean_f1']:.4f}",
            flush=True,
        )

    feature_selection = pd.DataFrame(feature_summaries).sort_values(
        ["selection_score", "mean_pr_auc", "mean_f1"], ascending=False
    )
    feature_selection.to_csv(ARTIFACT_DIR / "feature_selection.csv", index=False)
    pd.concat(feature_fold_rows, ignore_index=True).to_csv(
        ARTIFACT_DIR / "feature_selection_fold_metrics.csv", index=False
    )
    pd.concat(feature_threshold_rows, ignore_index=True).to_csv(
        ARTIFACT_DIR / "feature_selection_threshold_search.csv", index=False
    )
    selected_feature_set = str(feature_selection.iloc[0]["feature_set"])

    final_split = chronological_split(modeled)
    original_train, original_validation = enrich_fold(final_split.train, final_split.valid)
    original_test = apply_training_history(
        early_warning_candidates(final_split.test),
        history_lookup(final_split.train),
    )

    negative_rows = int((original_train["will_become_saturated_soon"] == 0).sum())
    positive_rows = int((original_train["will_become_saturated_soon"] == 1).sum())
    imbalance_ratio = negative_rows / positive_rows
    specs = candidate_specs(selected_feature_set, availability, imbalance_ratio)

    print(
        f"Model tuning on feature set '{selected_feature_set}' with {len(specs)} candidates",
        flush=True,
    )
    model_summaries = []
    model_fold_rows = []
    model_threshold_rows = []
    for index, spec in enumerate(specs, start=1):
        summary, fold_rows, threshold_rows = evaluate_candidate(spec, folds)
        model_summaries.append(summary)
        model_fold_rows.append(fold_rows)
        model_threshold_rows.append(threshold_rows)
        print(
            f"  [{index}/{len(specs)}] {spec.candidate_id}: "
            f"PR-AUC={summary['mean_pr_auc']:.4f}, F1={summary['mean_f1']:.4f}",
            flush=True,
        )

    model_comparison = pd.DataFrame(model_summaries).sort_values(
        ["selection_score", "mean_pr_auc", "mean_f1", "mean_precision", "mean_recall"],
        ascending=False,
    )
    model_comparison.to_csv(ARTIFACT_DIR / "rolling_model_comparison.csv", index=False)
    pd.concat(model_fold_rows, ignore_index=True).to_csv(
        ARTIFACT_DIR / "rolling_model_fold_metrics.csv", index=False
    )
    pd.concat(model_threshold_rows, ignore_index=True).to_csv(
        ARTIFACT_DIR / "rolling_threshold_search.csv", index=False
    )

    selected_id = str(model_comparison.iloc[0]["candidate_id"])
    selected_spec = next(spec for spec in specs if spec.candidate_id == selected_id)
    best_new_row = model_comparison.loc[model_comparison["is_new"].astype(bool)].iloc[0]
    baseline_rolling_row = model_comparison.loc[
        model_comparison["candidate_id"] == "existing_random_forest"
    ].iloc[0]

    winning_model, winning_validation_prob, winning_threshold, winning_validation_metrics = (
        fit_final_candidate(selected_spec, original_train, original_validation)
    )
    winning_model_file = save_model(winning_model, selected_spec)

    deployed_model = joblib.load(DEPLOYED_RF_PATH)
    deployed_model.named_steps["model"].n_jobs = 1
    baseline_validation_prob = deployed_model.predict_proba(
        feature_matrix(original_validation, FEATURE_SETS["existing"])
    )[:, 1]
    with DEPLOYED_METRICS_PATH.open("r", encoding="utf-8") as handle:
        deployed_metrics = json.load(handle)
    baseline_threshold = float(deployed_metrics["deployed_classifier"]["threshold"])
    baseline_validation_metrics = classification_metrics(
        original_validation["will_become_saturated_soon"].astype(int),
        baseline_validation_prob,
        baseline_threshold,
    )
    for metric_name in CLASSIFICATION_METRICS:
        stored_value = float(deployed_metrics["deployed_classifier"]["validation"][metric_name])
        if not np.isclose(baseline_validation_metrics[metric_name], stored_value, rtol=1e-9, atol=1e-12):
            raise AssertionError(
                f"Baseline validation {metric_name} differs from deployed metrics: "
                f"{baseline_validation_metrics[metric_name]} vs {stored_value}"
            )

    validation_bootstrap = paired_cluster_bootstrap(
        original_validation,
        baseline_validation_prob,
        winning_validation_prob,
        baseline_threshold,
        winning_threshold,
    )

    frozen_selection = {
        "frozen_before_test_evaluation": True,
        "selection_data": "three expanding chronological validation folds only",
        "selection_rule": (
            "maximize mean of rolling PR-AUC and threshold-optimized F1; then PR-AUC, "
            "F1, precision and recall"
        ),
        "selected_candidate": selected_spec.candidate_id,
        "family": selected_spec.family,
        "feature_set": selected_spec.feature_set,
        "features": FEATURE_SETS[selected_spec.feature_set],
        "params": selected_spec.params,
        "threshold": winning_threshold,
        "model_file": winning_model_file,
        "rolling_validation": model_comparison.iloc[0].to_dict(),
        "final_validation": winning_validation_metrics,
        "baseline_final_validation": baseline_validation_metrics,
        "validation_bootstrap_deltas": validation_bootstrap,
    }
    write_json(ARTIFACT_DIR / "frozen_selection.json", frozen_selection)

    # Test evaluation begins only after the winning configuration and threshold are frozen above.
    baseline_test_prob = deployed_model.predict_proba(
        feature_matrix(original_test, FEATURE_SETS["existing"])
    )[:, 1]
    winning_test_prob = predict_model(
        winning_model,
        selected_spec.family,
        original_test,
        FEATURE_SETS[selected_spec.feature_set],
    )
    baseline_test_metrics = classification_metrics(
        original_test["will_become_saturated_soon"].astype(int),
        baseline_test_prob,
        baseline_threshold,
    )
    winning_test_metrics = classification_metrics(
        original_test["will_become_saturated_soon"].astype(int),
        winning_test_prob,
        winning_threshold,
    )
    test_bootstrap = paired_cluster_bootstrap(
        original_test,
        baseline_test_prob,
        winning_test_prob,
        baseline_threshold,
        winning_threshold,
    )

    validation_deltas = {
        metric: winning_validation_metrics[metric] - baseline_validation_metrics[metric]
        for metric in ("precision", "recall", "f1", "pr_auc")
    }
    test_deltas = {
        metric: winning_test_metrics[metric] - baseline_test_metrics[metric]
        for metric in ("precision", "recall", "f1", "pr_auc")
    }
    operationally_meaningful = (
        validation_deltas["f1"] >= 0.02
        and validation_deltas["pr_auc"] >= 0.02
        and test_deltas["f1"] >= 0.02
        and test_deltas["pr_auc"] >= 0.02
        and validation_deltas["precision"] > 0
        and validation_deltas["recall"] > 0
        and test_deltas["precision"] > 0
        and test_deltas["recall"] > 0
    )
    statistically_supported = (
        validation_bootstrap["f1"]["ci_2_5"] > 0
        and validation_bootstrap["pr_auc"]["ci_2_5"] > 0
        and test_bootstrap["f1"]["ci_2_5"] > 0
        and test_bootstrap["pr_auc"]["ci_2_5"] > 0
    )
    recommend_replacement = operationally_meaningful and statistically_supported

    predictions = original_test[[
        "parking_id",
        "timestamp",
        "occupancy_ratio",
        "future_occupancy_ratio",
        "will_become_saturated_soon",
    ]].copy()
    predictions["baseline_probability"] = baseline_test_prob
    predictions["baseline_threshold"] = baseline_threshold
    predictions["winning_candidate"] = selected_spec.candidate_id
    predictions["winning_probability"] = winning_test_prob
    predictions["winning_threshold"] = winning_threshold
    predictions.to_csv(ARTIFACT_DIR / "untouched_test_predictions.csv", index=False)

    feature_manifest = {
        "feature_sets": FEATURE_SETS,
        "selected_feature_set": selected_feature_set,
        "history_method": (
            "Training rows use groupwise expanding statistics shifted by one observation. "
            "Validation and test rows use aggregates fitted only on the training period."
        ),
        "estimated_minutes_to_90": (
            "distance_to_90 divided by positive 60-minute rolling slope; zero, negative or "
            "missing growth is encoded as 999 minutes and positive estimates are capped at 999."
        ),
    }
    write_json(ARTIFACT_DIR / "feature_manifest.json", feature_manifest)

    results = {
        "baseline_validation": baseline_validation_metrics,
        "baseline_rolling_validation": baseline_rolling_row.to_dict(),
        "best_new_rolling_validation": best_new_row.to_dict(),
        "winner": frozen_selection,
        "untouched_test": {
            "baseline": baseline_test_metrics,
            "winner": winning_test_metrics,
            "deltas": test_deltas,
            "bootstrap_deltas": test_bootstrap,
        },
        "validation_deltas": validation_deltas,
        "operational_materiality_rule": (
            "F1 and PR-AUC improve by at least 0.02 on validation and test, with precision "
            "and recall both improving on validation and test."
        ),
        "operationally_meaningful": operationally_meaningful,
        "statistically_supported": statistically_supported,
        "recommend_replacement": recommend_replacement,
        "target_feasibility": target_feasibility,
        "availability": availability,
    }
    write_json(ARTIFACT_DIR / "experiment_results.json", results)

    protected_after = {str(path.relative_to(ROOT)): sha256(path) for path in protected_files}
    preservation = {
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "all_unchanged": protected_before == protected_after,
    }
    write_json(ARTIFACT_DIR / "deployment_preservation_check.json", preservation)
    if not preservation["all_unchanged"]:
        raise AssertionError("A protected deployment file changed during the experiment.")

    print("\nFrozen winner and untouched-test evaluation complete", flush=True)
    print(json.dumps(json_ready({
        "selected_candidate": selected_spec.candidate_id,
        "selected_feature_set": selected_spec.feature_set,
        "threshold": winning_threshold,
        "baseline_validation": baseline_validation_metrics,
        "winner_validation": winning_validation_metrics,
        "baseline_test": baseline_test_metrics,
        "winner_test": winning_test_metrics,
        "recommend_replacement": recommend_replacement,
        "protected_deployment_unchanged": preservation["all_unchanged"],
    }), indent=2), flush=True)


if __name__ == "__main__":
    main()
