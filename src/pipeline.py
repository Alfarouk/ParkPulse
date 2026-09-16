from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

SATURATION_THRESHOLD = 0.90
FORECAST_MINUTES = 30
TARGET_TOLERANCE_MINUTES = 10

RAW_RENAME = {
    "SystemCodeNumber": "parking_id",
    "Capacity": "capacity",
    "Occupancy": "occupancy",
    "LastUpdated": "timestamp",
}

FEATURES = [
    "parking_id",
    "capacity",
    "occupancy",
    "occupancy_ratio",
    "lag30_ratio",
    "lag60_ratio",
    "delta30_ratio",
    "delta60_ratio",
    "hour_sin",
    "hour_cos",
    "weekday",
    "is_weekend",
]

NUMERIC_FEATURES = [f for f in FEATURES if f != "parking_id"]
CATEGORICAL_FEATURES = ["parking_id"]


@dataclass
class SplitData:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def load_and_clean_csv(path: str | Path) -> pd.DataFrame:
    """Load UCI Parking Birmingham CSV and apply conservative sanity cleaning."""
    df = pd.read_csv(path)
    missing_cols = [c for c in RAW_RENAME if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected raw columns: {missing_cols}")

    df = df.rename(columns=RAW_RENAME)[list(RAW_RENAME.values())].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")
    df["occupancy"] = pd.to_numeric(df["occupancy"], errors="coerce")
    df["parking_id"] = df["parking_id"].astype("string")

    df = df.dropna(subset=["parking_id", "capacity", "occupancy", "timestamp"])
    df = df[(df["capacity"] > 0) & (df["occupancy"] >= 0)]
    df = df[df["occupancy"] <= df["capacity"]]
    df = df.drop_duplicates(subset=["parking_id", "timestamp"], keep="last")
    df = df.sort_values(["parking_id", "timestamp"]).reset_index(drop=True)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create only past-looking features; masks lags that cross large time gaps/day boundaries."""
    out = df.copy()
    out["occupancy_ratio"] = out["occupancy"] / out["capacity"]

    g = out.groupby("parking_id", group_keys=False)
    out["prev1_ratio"] = g["occupancy_ratio"].shift(1)
    out["prev2_ratio"] = g["occupancy_ratio"].shift(2)
    out["prev1_time"] = g["timestamp"].shift(1)
    out["prev2_time"] = g["timestamp"].shift(2)

    dt1 = (out["timestamp"] - out["prev1_time"]).dt.total_seconds() / 60.0
    dt2 = (out["timestamp"] - out["prev2_time"]).dt.total_seconds() / 60.0

    out["lag30_ratio"] = out["prev1_ratio"].where(dt1.between(20, 40))
    out["lag60_ratio"] = out["prev2_ratio"].where(dt2.between(45, 75))
    out["delta30_ratio"] = out["occupancy_ratio"] - out["lag30_ratio"]
    out["delta60_ratio"] = out["occupancy_ratio"] - out["lag60_ratio"]

    minute_of_day = out["timestamp"].dt.hour * 60 + out["timestamp"].dt.minute
    angle = 2 * np.pi * minute_of_day / (24 * 60)
    out["hour_sin"] = np.sin(angle)
    out["hour_cos"] = np.cos(angle)
    out["weekday"] = out["timestamp"].dt.dayofweek.astype(int)
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)

    return out.drop(columns=["prev1_ratio", "prev2_ratio", "prev1_time", "prev2_time"])


def build_future_target(
    df: pd.DataFrame,
    minutes: int = FORECAST_MINUTES,
    tolerance_minutes: int = TARGET_TOLERANCE_MINUTES,
) -> pd.DataFrame:
    """
    Match each observation to the nearest observation around t + minutes within tolerance.
    This is safer than exact timestamp equality when sensor timestamps contain seconds/jitter.
    """
    pieces = []
    tolerance = pd.Timedelta(minutes=tolerance_minutes)

    for parking_id, g in df.groupby("parking_id", sort=False):
        g = g.sort_values("timestamp").copy()
        left = g.copy()
        left["desired_future_time"] = left["timestamp"] + pd.Timedelta(minutes=minutes)

        right = g[["timestamp", "occupancy", "capacity"]].rename(
            columns={
                "timestamp": "future_timestamp",
                "occupancy": "future_occupancy",
                "capacity": "future_capacity",
            }
        )

        merged = pd.merge_asof(
            left.sort_values("desired_future_time"),
            right.sort_values("future_timestamp"),
            left_on="desired_future_time",
            right_on="future_timestamp",
            direction="nearest",
            tolerance=tolerance,
        )
        merged["parking_id"] = parking_id
        pieces.append(merged)

    out = pd.concat(pieces, ignore_index=True)
    out = out.dropna(subset=["future_occupancy", "future_capacity", "future_timestamp"])

    # Guard against accidentally matching a non-future observation.
    out = out[out["future_timestamp"] > out["timestamp"]]

    out["future_occupancy_ratio"] = out["future_occupancy"] / out["future_capacity"]
    out["currently_saturated"] = out["occupancy_ratio"] >= SATURATION_THRESHOLD
    out["future_saturated"] = (out["future_occupancy_ratio"] >= SATURATION_THRESHOLD).astype(int)
    out["will_become_saturated_soon"] = (
        (~out["currently_saturated"]) & (out["future_occupancy_ratio"] >= SATURATION_THRESHOLD)
    ).astype(int)
    out["target_offset_minutes"] = (
        out["future_timestamp"] - out["timestamp"]
    ).dt.total_seconds() / 60.0
    return out.sort_values("timestamp").reset_index(drop=True)


def chronological_split(df: pd.DataFrame, train_frac: float = 0.70, valid_frac: float = 0.15) -> SplitData:
    """Split by time, never randomly. Future labels are kept inside their own split windows."""
    ordered = df.sort_values("timestamp").copy()
    unique_times = np.array(sorted(ordered["timestamp"].dropna().unique()))
    if len(unique_times) < 10:
        raise ValueError("Not enough unique timestamps for a chronological split.")

    train_idx = max(1, int(len(unique_times) * train_frac) - 1)
    valid_idx = max(train_idx + 1, int(len(unique_times) * (train_frac + valid_frac)) - 1)
    train_end = pd.Timestamp(unique_times[train_idx])
    valid_end = pd.Timestamp(unique_times[min(valid_idx, len(unique_times) - 1)])

    train = ordered[ordered["future_timestamp"] <= train_end].copy()
    valid = ordered[(ordered["timestamp"] > train_end) & (ordered["future_timestamp"] <= valid_end)].copy()
    test = ordered[ordered["timestamp"] > valid_end].copy()

    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError("One chronological split is empty; inspect timestamp coverage.")
    return SplitData(train=train, valid=valid, test=test)


def early_warning_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows that are not saturated at prediction time."""
    return df.loc[df["occupancy_ratio"] < SATURATION_THRESHOLD].copy()


def sklearn_preprocessor(feature_columns: Sequence[str] = FEATURES) -> ColumnTransformer:
    numeric_features = [feature for feature in feature_columns if feature != "parking_id"]
    categorical_features = [feature for feature in feature_columns if feature == "parking_id"]
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_features),
            ("cat", categorical, categorical_features),
        ]
    )


def build_logistic_classifier(
    random_state: int = 42,
    feature_columns: Sequence[str] = FEATURES,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", sklearn_preprocessor(feature_columns)),
            ("model", LogisticRegression(max_iter=3000, random_state=random_state)),
        ]
    )


def build_random_forest_classifier(
    random_state: int = 42,
    feature_columns: Sequence[str] = FEATURES,
) -> Pipeline:
    numeric_features = [feature for feature in feature_columns if feature != "parking_id"]
    categorical_features = [feature for feature in feature_columns if feature == "parking_id"]
    prep = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("prep", prep),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    min_samples_leaf=2,
                    class_weight=None,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_catboost_classifier(random_state: int = 42) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=700,
        depth=7,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
    )


def build_random_forest_regressor(random_state: int = 42) -> Pipeline:
    prep = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("prep", prep),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_catboost_regressor(random_state: int = 42) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=700,
        depth=7,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
    )


def classification_metrics(y_true, probabilities, threshold: float = 0.50) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    pred = (probabilities >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) == 2
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, probabilities)) if both_classes else float("nan"),
        "brier": float(brier_score_loss(y_true, probabilities)),
    }


def regression_metrics(y_true, pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "r2": float(r2_score(y_true, pred)),
    }


def choose_threshold_for_recall(
    y_true,
    probabilities,
    minimum_precision: float = 0.50,
) -> Tuple[float, pd.DataFrame]:
    """Pick threshold on validation data only: maximize recall while keeping precision acceptable."""
    y_arr = np.asarray(y_true).astype(int)
    if len(np.unique(y_arr)) < 2:
        # A threshold cannot be meaningfully tuned when validation contains only one class.
        return 0.50, pd.DataFrame([classification_metrics(y_arr, probabilities, 0.50)])

    rows = []
    for threshold in np.arange(0.10, 0.91, 0.01):
        m = classification_metrics(y_true, probabilities, float(threshold))
        rows.append(m)
    table = pd.DataFrame(rows)
    feasible = table[table["precision"] >= minimum_precision]
    if len(feasible):
        best = feasible.sort_values(["recall", "f1"], ascending=False).iloc[0]
    else:
        best = table.sort_values("f1", ascending=False).iloc[0]
    return float(best["threshold"]), table


def choose_threshold_for_f1(
    y_true,
    probabilities,
) -> Tuple[float, pd.DataFrame]:
    """Pick the maximum-F1 threshold using validation labels and probabilities only."""
    y_arr = np.asarray(y_true).astype(int)
    if len(np.unique(y_arr)) < 2:
        # A threshold cannot be meaningfully tuned when validation contains only one class.
        return 0.50, pd.DataFrame([classification_metrics(y_arr, probabilities, 0.50)])

    rows = [
        classification_metrics(y_arr, probabilities, float(threshold))
        for threshold in np.arange(0.10, 0.91, 0.01)
    ]
    table = pd.DataFrame(rows)
    best = table.loc[table["f1"].idxmax()]
    return float(best["threshold"]), table


def prepare_dataset(csv_path: str | Path) -> pd.DataFrame:
    raw = load_and_clean_csv(csv_path)
    with_features = add_lag_features(raw)
    modeled = build_future_target(with_features)
    return modeled
