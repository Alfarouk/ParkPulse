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
    build_catboost_regressor,
    build_logistic_classifier,
    build_random_forest_classifier,
    build_random_forest_regressor,
    choose_threshold_for_f1,
    chronological_split,
    classification_metrics,
    early_warning_candidates,
    prepare_dataset,
    regression_metrics,
)

DATA_PATH = Path("data/dataset.csv")
MODEL_DIR = Path("models")
ARTIFACT_DIR = Path("artifacts")
MODEL_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)


def timed_fit(name, fit_fn):
    start = time.perf_counter()
    result = fit_fn()
    seconds = time.perf_counter() - start
    print(f"{name} fit time: {seconds:.3f}s")
    return result, seconds


def main():
    df = prepare_dataset(DATA_PATH)
    df.to_csv(ARTIFACT_DIR / "modeled_dataset.csv", index=False)

    splits = chronological_split(df)
    splits.test.to_csv(ARTIFACT_DIR / "test_demo.csv", index=False)
    classifier_splits = {
        "train": early_warning_candidates(splits.train),
        "valid": early_warning_candidates(splits.valid),
        "test": early_warning_candidates(splits.test),
    }
    print("All rows:", {"train": len(splits.train), "valid": len(splits.valid), "test": len(splits.test)})
    print("Classifier rows (currently below 90%):", {
        name: len(part) for name, part in classifier_splits.items()
    })
    print("Early-warning positive rate:", {
        name: float(part.will_become_saturated_soon.mean())
        for name, part in classifier_splits.items()
    })

    X_train = classifier_splits["train"][FEATURES].copy()
    y_train = classifier_splits["train"]["will_become_saturated_soon"].astype(int)
    X_valid = classifier_splits["valid"][FEATURES].copy()
    y_valid = classifier_splits["valid"]["will_become_saturated_soon"].astype(int)
    X_test = classifier_splits["test"][FEATURES].copy()
    y_test = classifier_splits["test"]["will_become_saturated_soon"].astype(int)

    # CatBoost expects categorical values to be strings and missing numeric values are fine.
    for part in (X_train, X_valid, X_test):
        part["parking_id"] = part["parking_id"].astype(str)

    results = {}

    # Persistence baselines: future state ~= current state.
    baseline_prob = X_test["occupancy_ratio"].to_numpy()
    results["persistence_classifier"] = {
        "fit_seconds": 0.0,
        "threshold": 0.90,
        "test": classification_metrics(y_test, baseline_prob, 0.90),
    }

    logistic, seconds = timed_fit("Logistic", lambda: build_logistic_classifier().fit(X_train, y_train))
    val_p = logistic.predict_proba(X_valid)[:, 1]
    threshold, _ = choose_threshold_for_f1(y_valid, val_p)
    logistic_test_p = logistic.predict_proba(X_test)[:, 1]
    results["logistic"] = {
        "fit_seconds": seconds,
        "threshold": threshold,
        "validation": classification_metrics(y_valid, val_p, threshold),
        "test": classification_metrics(y_test, logistic_test_p, threshold),
    }
    joblib.dump(logistic, MODEL_DIR / "logistic_classifier.joblib")

    rf, seconds = timed_fit("RandomForest", lambda: build_random_forest_classifier().fit(X_train, y_train))
    val_p = rf.predict_proba(X_valid)[:, 1]
    threshold, _ = choose_threshold_for_f1(y_valid, val_p)
    random_forest_test_p = rf.predict_proba(X_test)[:, 1]
    results["random_forest"] = {
        "fit_seconds": seconds,
        "threshold": threshold,
        "validation": classification_metrics(y_valid, val_p, threshold),
        "test": classification_metrics(y_test, random_forest_test_p, threshold),
    }
    joblib.dump(rf, MODEL_DIR / "random_forest_classifier.joblib")

    cat = build_catboost_classifier()
    train_pool = Pool(X_train, y_train, cat_features=["parking_id"])
    valid_pool = Pool(X_valid, y_valid, cat_features=["parking_id"])
    test_pool = Pool(X_test, y_test, cat_features=["parking_id"])
    _, seconds = timed_fit(
        "CatBoost",
        lambda: cat.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=80),
    )
    val_p = cat.predict_proba(valid_pool)[:, 1]
    threshold, threshold_table = choose_threshold_for_f1(y_valid, val_p)
    catboost_test_p = cat.predict_proba(test_pool)[:, 1]
    results["catboost"] = {
        "fit_seconds": seconds,
        "threshold": threshold,
        "best_iteration": int(cat.get_best_iteration()),
        "validation": classification_metrics(y_valid, val_p, threshold),
        "test": classification_metrics(y_test, catboost_test_p, threshold),
    }
    threshold_table.to_csv(ARTIFACT_DIR / "catboost_threshold_search.csv", index=False)
    cat.save_model(MODEL_DIR / "catboost_classifier.cbm")

    classifier_probabilities = {
        "logistic": logistic_test_p,
        "random_forest": random_forest_test_p,
        "catboost": catboost_test_p,
    }
    classifier_model_files = {
        "logistic": "logistic_classifier.joblib",
        "random_forest": "random_forest_classifier.joblib",
        "catboost": "catboost_classifier.cbm",
    }
    selected_classifier = max(
        classifier_probabilities,
        key=lambda name: (
            results[name]["validation"]["f1"],
            results[name]["validation"]["pr_auc"],
            -results[name]["validation"]["brier"],
        ),
    )
    results["deployed_classifier"] = {
        "model": selected_classifier,
        "model_file": classifier_model_files[selected_classifier],
        "experiment": "parking_only",
        "feature_set": "FEATURES",
        "fit_seconds": results[selected_classifier]["fit_seconds"],
        "threshold": results[selected_classifier]["threshold"],
        "selection_rule": (
            "highest validation F1; validation PR-AUC then lower validation Brier as tie-breakers"
        ),
        "validation": results[selected_classifier]["validation"],
        "test": results[selected_classifier]["test"],
    }

    # Regression is a secondary output for predicted occupancy count.
    yr_train = splits.train["future_occupancy"].astype(float)
    yr_valid = splits.valid["future_occupancy"].astype(float)
    yr_test = splits.test["future_occupancy"].astype(float)
    X_train_reg = splits.train[FEATURES].copy()
    X_valid_reg = splits.valid[FEATURES].copy()
    X_test_reg = splits.test[FEATURES].copy()
    for part in (X_train_reg, X_valid_reg, X_test_reg):
        part["parking_id"] = part["parking_id"].astype(str)
    results["persistence_regressor"] = {
        "fit_seconds": 0.0,
        "test": regression_metrics(yr_test, splits.test["occupancy"].astype(float)),
    }

    rf_reg, seconds = timed_fit("RF regressor", lambda: build_random_forest_regressor().fit(X_train_reg, yr_train))
    results["rf_regressor"] = {
        "fit_seconds": seconds,
        "validation": regression_metrics(yr_valid, rf_reg.predict(X_valid_reg)),
        "test": regression_metrics(yr_test, rf_reg.predict(X_test_reg)),
    }
    joblib.dump(rf_reg, MODEL_DIR / "random_forest_regressor.joblib")

    cat_reg = build_catboost_regressor()
    _, seconds = timed_fit(
        "CatBoost regressor",
        lambda: cat_reg.fit(
            Pool(X_train_reg, yr_train, cat_features=["parking_id"]),
            eval_set=Pool(X_valid_reg, yr_valid, cat_features=["parking_id"]),
            early_stopping_rounds=80,
        ),
    )
    reg_test_pred = cat_reg.predict(Pool(X_test_reg, cat_features=["parking_id"]))
    reg_valid_pred = cat_reg.predict(Pool(X_valid_reg, cat_features=["parking_id"]))
    results["catboost_regressor"] = {
        "fit_seconds": seconds,
        "best_iteration": int(cat_reg.get_best_iteration()),
        "validation": regression_metrics(yr_valid, reg_valid_pred),
        "test": regression_metrics(yr_test, reg_test_pred),
    }
    cat_reg.save_model(MODEL_DIR / "catboost_regressor.cbm")

    test_predictions = splits.test[[
        "parking_id", "timestamp", "capacity", "occupancy",
        "occupancy_ratio", "currently_saturated", "future_timestamp",
        "future_occupancy", "future_occupancy_ratio", "future_saturated",
        "will_become_saturated_soon", "target_offset_minutes"
    ]].copy()
    test_predictions["saturation_probability"] = pd.NA
    test_predictions.loc[classifier_splits["test"].index, "saturation_probability"] = (
        classifier_probabilities[selected_classifier]
    )
    test_predictions["saturation_probability"] = pd.to_numeric(
        test_predictions["saturation_probability"], errors="coerce"
    )
    test_predictions["predicted_future_occupancy"] = reg_test_pred
    test_predictions.to_csv(ARTIFACT_DIR / "deployed_test_predictions.csv", index=False)

    # Preserve the CatBoost classifier export as an experiment artifact even when
    # validation selects a different classifier for deployment.
    catboost_predictions = test_predictions.copy()
    catboost_predictions["saturation_probability"] = pd.NA
    catboost_predictions.loc[classifier_splits["test"].index, "saturation_probability"] = (
        catboost_test_p
    )
    catboost_predictions["saturation_probability"] = pd.to_numeric(
        catboost_predictions["saturation_probability"], errors="coerce"
    )
    catboost_predictions.to_csv(
        ARTIFACT_DIR / "catboost_test_predictions.csv", index=False
    )

    with open(ARTIFACT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary_rows = []
    for name, info in results.items():
        row = {"model": name, "fit_seconds": info["fit_seconds"]}
        if "test" in info:
            row.update({f"test_{k}": v for k, v in info["test"].items()})
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(ARTIFACT_DIR / "model_summary.csv", index=False)

    print(json.dumps(results, indent=2))
    print("\nPrimary decision rule: choose model and alert threshold using validation data only; use the test set only for final evaluation.")


if __name__ == "__main__":
    main()
