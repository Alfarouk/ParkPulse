from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor, Pool

from src.pipeline import FEATURES, SATURATION_THRESHOLD

ARTIFACT_DIR = Path("artifacts")
MODEL_DIR = Path("models")
TRAFFIC_COMPARISON_PATH = (
    ARTIFACT_DIR / "traffic_context_experiments" / "selected_model_comparison.csv"
)
OPTIMIZATION_RESULTS_PATH = (
    Path("optimization_experiment") / "artifacts" / "experiment_results.json"
)

st.set_page_config(page_title="ParkPulse", layout="wide")
st.title("ParkPulse — 30-Minute Parking Saturation Early Warning")
st.caption(
    "Historical Birmingham test-period demo: predict saturation risk about 30 minutes "
    "ahead and rank car parks by risk."
)

if not (ARTIFACT_DIR / "test_demo.csv").exists():
    st.error("Run `python train.py` first.")
    st.stop()

with open(ARTIFACT_DIR / "metrics.json", "r", encoding="utf-8") as f:
    metrics = json.load(f)
deployment = metrics["deployed_classifier"]
early_warning_threshold = float(deployment["threshold"])

# The following comparison artifacts are presentation-only context. They do not affect
# model loading, feature generation, predictions, or the deployed threshold.
traffic_comparison = pd.read_csv(TRAFFIC_COMPARISON_PATH)
with open(OPTIMIZATION_RESULTS_PATH, "r", encoding="utf-8") as f:
    optimization_results = json.load(f)

clf = joblib.load(MODEL_DIR / deployment["model_file"])
# Prediction is small enough to run serially and avoids spawning worker pools in
# constrained Streamlit hosting environments. This does not alter the trained model.
clf.named_steps["model"].n_jobs = 1
reg = CatBoostRegressor()
reg.load_model(MODEL_DIR / "catboost_regressor.cbm")

df = pd.read_csv(
    ARTIFACT_DIR / "test_demo.csv",
    parse_dates=["timestamp", "future_timestamp"],
)
df["slot"] = df["timestamp"].dt.round("30min")

# Use the latest observation for each car park within each displayed half-hour slot.
historical_rows = (
    df.sort_values(["slot", "parking_id", "timestamp"])
    .drop_duplicates(subset=["slot", "parking_id"], keep="last")
    .copy()
)
historical_rows["currently_saturated"] = (
    historical_rows["occupancy_ratio"] >= SATURATION_THRESHOLD
)
historical_rows["early_warning_probability"] = np.nan

# Timestamp filtering uses only current/past-looking model features and current occupancy.
eligible_history = ~historical_rows["currently_saturated"]
if eligible_history.any():
    history_X = historical_rows.loc[eligible_history, FEATURES].copy()
    history_X["parking_id"] = history_X["parking_id"].astype(str)
    historical_rows.loc[eligible_history, "early_warning_probability"] = (
        clf.predict_proba(history_X)[:, 1]
    )

historical_rows["status"] = np.select(
    [
        historical_rows["currently_saturated"],
        historical_rows["early_warning_probability"] >= early_warning_threshold,
    ],
    ["ALREADY SATURATED", "EARLY WARNING"],
    default="NORMAL",
)

all_slots = sorted(historical_rows["slot"].dropna().unique())
warning_slots = sorted(
    historical_rows.loc[historical_rows["status"] == "EARLY WARNING", "slot"].unique()
)
show_warning_slots_only = st.checkbox("Show timestamps with early warnings only")
slots = warning_slots if show_warning_slots_only else all_slots
if not slots:
    st.warning("No historical timestamps contain an early warning.")
    st.stop()

selected_slot = st.selectbox(
    "Choose historical observation window",
    slots,
    index=max(0, len(slots) - 2),
)
st.caption(
    "Raw sensor observations are irregular. ParkPulse rounds timestamps to 30-minute "
    "windows for navigation only; the exact historical timestamp is shown below."
)
slot_df = historical_rows[historical_rows["slot"] == selected_slot].copy()

X = slot_df[FEATURES].copy()
X["parking_id"] = X["parking_id"].astype(str)
reg_pool = Pool(X, cat_features=["parking_id"])
slot_df["raw_predicted_occupancy"] = reg.predict(reg_pool)
slot_df["predicted_occupancy"] = np.clip(
    slot_df["raw_predicted_occupancy"],
    0,
    slot_df["capacity"],
)
slot_df["predicted_fill_pct"] = slot_df["predicted_occupancy"] / slot_df["capacity"] * 100
slot_df["current_fill_pct"] = slot_df["occupancy"] / slot_df["capacity"] * 100

status_order = {
    "ALREADY SATURATED": 0,
    "EARLY WARNING": 1,
    "NORMAL": 2,
}
ranking = (
    slot_df.assign(_status_order=slot_df["status"].map(status_order))
    .sort_values(["_status_order", "early_warning_probability"], ascending=[True, False])
    .drop(columns="_status_order")
    .reset_index(drop=True)
)
ranking.index = ranking.index + 1

status_counts = ranking["status"].value_counts()
count_columns = st.columns(3)
count_columns[0].metric(
    "🔴 Already saturated",
    int(status_counts.get("ALREADY SATURATED", 0)),
)
count_columns[1].metric(
    "🟠 Early warnings",
    int(status_counts.get("EARLY WARNING", 0)),
)
count_columns[2].metric(
    "🟢 Normal",
    int(status_counts.get("NORMAL", 0)),
)


def style_status(value):
    styles = {
        "ALREADY SATURATED": "background-color: #fee2e2; color: #991b1b; font-weight: 700",
        "EARLY WARNING": "background-color: #fef3c7; color: #92400e; font-weight: 700",
        "NORMAL": "background-color: #dcfce7; color: #166534; font-weight: 700",
    }
    return styles.get(value, "")


st.subheader("Citywide risk ranking")
st.dataframe(
    ranking[[
        "parking_id", "current_fill_pct", "predicted_fill_pct",
        "early_warning_probability", "status"
    ]].rename(columns={
        "parking_id": "Car park",
        "current_fill_pct": "Current fill %",
        "predicted_fill_pct": "Predicted fill %",
        "early_warning_probability": "30-min early-warning probability",
        "status": "Status",
    }).style.format({
        "Current fill %": "{:.1f}%",
        "Predicted fill %": "{:.1f}%",
        "30-min early-warning probability": "{:.1%}",
    }, na_rep="Not applicable").map(style_status, subset=["Status"]),
    width="stretch",
)

selected_garage = st.selectbox("Inspect one car park", ranking["parking_id"].tolist())
row = ranking[ranking["parking_id"] == selected_garage].iloc[0]

st.subheader("Selected historical record")
if row["status"] == "ALREADY SATURATED":
    st.error("🔴 ALREADY SATURATED")
elif row["status"] == "EARLY WARNING":
    st.warning("🟠 EARLY WARNING")
else:
    st.success("🟢 NORMAL")

detail_1, detail_2, detail_3 = st.columns(3)
detail_1.metric("Car park", str(row["parking_id"]))
detail_2.metric("Historical timestamp", row["timestamp"].strftime("%d %b %Y, %H:%M:%S"))
detail_3.metric("Capacity", f"{int(round(row['capacity'])):,} cars")

prediction_1, prediction_2, prediction_3, prediction_4 = st.columns(4)
prediction_1.metric("Current occupied cars", f"{int(round(row['occupancy'])):,}")
prediction_2.metric("Current fill", f"{row['current_fill_pct']:.1f}%")
prediction_3.metric("Predicted cars (~30 min)", f"{int(round(row['predicted_occupancy'])):,}")
prediction_4.metric("Predicted fill (~30 min)", f"{row['predicted_fill_pct']:.1f}%")

decision_1, decision_2 = st.columns(2)
if row["currently_saturated"]:
    decision_1.metric("Random Forest early-warning probability", "Not applicable")
    decision_2.metric("Decision", "ALREADY SATURATED")
    st.write(
        f"**Interpretation:** {selected_garage} is already at least 90% occupied, "
        "so it is not evaluated as an early-warning candidate."
    )
else:
    decision_1.metric(
        "Random Forest early-warning probability",
        f"{row['early_warning_probability']:.1%}",
    )
    decision_2.metric("Decision", row["status"])
    st.write(
        f"**Interpretation:** {selected_garage} currently has capacity available and has a "
        f"{row['early_warning_probability']:.1%} probability of becoming at least 90% full "
        "about 30 minutes later."
    )

st.markdown("#### Historical actual outcome (~30 minutes later)")
actual_1, actual_2, actual_3, actual_4 = st.columns(4)
actual_1.metric("Actual occupied cars", f"{int(round(row['future_occupancy'])):,}")
actual_2.metric("Actual fill", f"{row['future_occupancy_ratio']:.1%}")
actual_3.metric("Actual forecast offset", f"{row['target_offset_minutes']:.1f} min")
actual_4.metric(
    "Actual observation timestamp",
    row["future_timestamp"].strftime("%d %b %Y, %H:%M:%S"),
)

if row["currently_saturated"]:
    st.info("Classifier outcome: NOT APPLICABLE — garage was already saturated.")
else:
    issued_warning = row["status"] == "EARLY WARNING"
    became_saturated = row["future_occupancy_ratio"] >= SATURATION_THRESHOLD
    if issued_warning and became_saturated:
        st.success("Classifier outcome: TRUE POSITIVE — warning issued and the garage became saturated.")
    elif issued_warning and not became_saturated:
        st.warning("Classifier outcome: FALSE POSITIVE — warning issued, but the garage stayed below 90%.")
    elif not issued_warning and became_saturated:
        st.error("Classifier outcome: FALSE NEGATIVE — no warning was issued, but the garage became saturated.")
    else:
        st.success("Classifier outcome: TRUE NEGATIVE — no warning was issued and the garage stayed below 90%.")

with st.expander("How ParkPulse makes predictions"):
    st.write(
        "For garages below 90%, the parking-only early-warning Random Forest estimates the "
        "probability of becoming saturated about 30 minutes later and compares it with the "
        f"validation-selected {early_warning_threshold:.2f} threshold. Garages already at least "
        "90% full are marked ALREADY SATURATED and are outside the classifier's target problem."
    )
    st.write(
        "The CatBoost regressor independently estimates the number of occupied spaces about 30 "
        "minutes later. Its point estimate does not determine the early-warning status."
    )

with st.expander("Model Performance"):
    classifier_test = deployment["test"]
    regressor_test = metrics["catboost_regressor"]["test"]
    eligible_test = df.loc[df["occupancy_ratio"] < SATURATION_THRESHOLD]
    positive_test = int(eligible_test["will_become_saturated_soon"].astype(int).sum())
    st.caption(
        "Performance on the untouched chronological test period. "
        f"{positive_test:,} positive early-warning events among {len(eligible_test):,} "
        f"eligible rows ({positive_test / len(eligible_test):.2%})."
    )
    st.markdown("**Random Forest early-warning classifier**")
    clf_metrics = st.columns(4)
    clf_metrics[0].metric("Precision", f"{classifier_test['precision']:.2%}")
    clf_metrics[1].metric("Recall", f"{classifier_test['recall']:.2%}")
    clf_metrics[2].metric("F1", f"{classifier_test['f1']:.2%}")
    clf_metrics[3].metric("PR-AUC", f"{classifier_test['pr_auc']:.2%}")
    clf_metrics_2 = st.columns(3)
    clf_metrics_2[0].metric("ROC-AUC", f"{classifier_test['roc_auc']:.2%}")
    clf_metrics_2[1].metric("Brier score", f"{classifier_test['brier']:.4f}")
    clf_metrics_2[2].metric("Alert threshold", f"{early_warning_threshold:.2f}")

    st.markdown("**CatBoost 30-minute occupancy regressor**")
    reg_metrics = st.columns(3)
    reg_metrics[0].metric("MAE", f"{regressor_test['mae']:.2f} cars")
    reg_metrics[1].metric("RMSE", f"{regressor_test['rmse']:.2f} cars")
    reg_metrics[2].metric("R²", f"{regressor_test['r2']:.4f}")
    st.caption(
        "Regression metrics use raw CatBoost outputs. Dashboard occupancy values are clipped "
        "only for display to the physical range from 0 to the car-park capacity."
    )

with st.expander("Second Dataset Experiment"):
    experiment_labels = {
        "parking_only": "Parking-only Random Forest",
        "parking_plus_traffic": "Parking + telematics Random Forest",
    }
    traffic_display = traffic_comparison.assign(
        Experiment=traffic_comparison["experiment"].map(experiment_labels)
    )[["Experiment", "test_precision", "test_recall", "test_f1", "test_pr_auc"]].rename(
        columns={
            "test_precision": "Precision",
            "test_recall": "Recall",
            "test_f1": "F1",
            "test_pr_auc": "PR-AUC",
        }
    )
    st.caption("Untouched chronological test-period comparison.")
    st.dataframe(
        traffic_display.style.format({
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1": "{:.2%}",
            "PR-AUC": "{:.2%}",
        }),
        hide_index=True,
        width="stretch",
    )
    st.write(
        "Adding typical citywide telematics context increased recall, but reduced precision, "
        "F1, and PR-AUC. It remains a controlled experimental comparison; the stronger "
        "parking-only Random Forest was retained for deployment."
    )

with st.expander("Optimization Experiment"):
    untouched_test = optimization_results["untouched_test"]
    optimization_display = pd.DataFrame([
        {
            "Model": "Deployed parking-only Random Forest",
            "Precision": untouched_test["baseline"]["precision"],
            "Recall": untouched_test["baseline"]["recall"],
            "F1": untouched_test["baseline"]["f1"],
            "PR-AUC": untouched_test["baseline"]["pr_auc"],
        },
        {
            "Model": "Optimized CatBoost candidate",
            "Precision": untouched_test["winner"]["precision"],
            "Recall": untouched_test["winner"]["recall"],
            "F1": untouched_test["winner"]["f1"],
            "PR-AUC": untouched_test["winner"]["pr_auc"],
        },
    ])
    st.caption("Untouched chronological test-period comparison after freezing the candidate.")
    st.dataframe(
        optimization_display.style.format({
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1": "{:.2%}",
            "PR-AUC": "{:.2%}",
        }),
        hide_index=True,
        width="stretch",
    )
    st.write(
        "The optimized CatBoost candidate improved development and rolling-validation results, "
        "but that advantage did not hold on the untouched test period. The deployed Random Forest "
        "was therefore retained."
    )

st.caption(
    "Early-warning classification uses the parking-only Random Forest with its "
    f"validation-selected {early_warning_threshold:.2f} threshold. The 30-minute occupancy "
    "forecast uses CatBoost. The telematics-enhanced classifier remains experimental because "
    "it reduced F1 and PR-AUC. "
    "This is a historical simulation/backtest using held-out observations, not live Birmingham data."
)
