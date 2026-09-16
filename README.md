# ParkPulse

ParkPulse is a **historical machine-learning backtest for Birmingham parking saturation**. It is designed as an early-warning system: for a car park that is **currently below 90% occupancy**, ParkPulse estimates whether it will reach **at least 90% occupancy approximately 30 minutes later**.

This repository is **not connected to live Birmingham parking or traffic feeds** and should not be interpreted as a current operational service.

## Core question

> Which currently-unsaturated car parks are likely to cross 90% occupancy approximately 30 minutes later?

ParkPulse uses two deployed models for different purposes:

- **Random Forest classifier:** produces the early-warning risk for car parks currently below 90% occupancy.
- **CatBoost regressor:** estimates the number of occupied spaces approximately 30 minutes later for all car parks.

Car parks that are already at least 90% occupied are outside the classifier's target problem and are shown as `ALREADY SATURATED` with early-warning probability `Not applicable`.

## Data and target construction

The main dataset is the UCI **Parking Birmingham** dataset. The raw file contains 35,717 observations from 30 car parks covering 4 October 2016 to 19 December 2016.

The main pipeline:

1. removes missing/invalid rows, negative occupancies, rows where occupancy exceeds capacity, and duplicate car-park/timestamp records;
2. creates only past-looking lag, change, calendar, and occupancy features;
3. matches each current observation to the nearest observation around `t + 30 minutes`, with a tolerance of ±10 minutes;
4. therefore evaluates targets at an actual offset of **20–40 minutes** rather than pretending every sensor reading occurs exactly 30 minutes later.

After cleaning, feature generation, and future-target matching, the modeled dataset contains **32,281 rows**.

### Early-warning target

A row is classifier-eligible only when:

```text
current occupancy ratio < 0.90
```

The positive class is:

```text
current occupancy ratio < 0.90
AND
future occupancy ratio >= 0.90
```

Rows already at or above 90% are excluded from classifier training and evaluation rather than being converted into negative examples.

## Features

The deployed parking-only models use 12 features available at prediction time:

```text
parking_id
capacity
occupancy
occupancy_ratio
lag30_ratio
lag60_ratio
delta30_ratio
delta60_ratio
hour_sin
hour_cos
weekday
is_weekend
```

## Leakage-safe evaluation

The data is split **chronologically**, never randomly. Future labels are also kept inside their corresponding split boundaries so an observation in one period cannot use a target from a later split.

| Split | All modeled rows | Classifier-eligible rows |
|---|---:|---:|
| Train | 23,197 | 21,593 |
| Validation | 4,160 | 3,676 |
| Untouched test | 4,873 | 4,511 |

The untouched classifier test set contains **70 positive early-warning events among 4,511 eligible rows**, a prevalence of about **1.55%**. Because the positive class is rare, the project emphasizes **precision, recall, F1, and PR-AUC** rather than headline accuracy.

## Final deployed system

### Early-warning classifier

- Model: **parking-only Random Forest**
- Model file: `models/random_forest_classifier.joblib`
- Alert threshold: **0.21**
- Threshold selection: maximum F1 on the chronological validation set

Decision routing:

```python
if current_fill >= 0.90:
    status = "ALREADY SATURATED"
elif rf_probability >= 0.21:
    status = "EARLY WARNING"
else:
    status = "NORMAL"
```

### Occupancy regressor

- Model: **CatBoost Regressor**
- Model file: `models/catboost_regressor.cbm`
- Output: estimated occupied-car count approximately 30 minutes later

The classifier and regressor are deliberately separate. The **classifier controls the saturation warning**; the CatBoost point estimate does not.

## Final held-out performance

### Random Forest early-warning classifier

| Split | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 65.71% | 77.53% | 71.13% | 78.97% | 99.30% | 0.0110 | 0.21 |
| Untouched test | 66.67% | 68.57% | 67.61% | 77.41% | 99.28% | 0.0072 | 0.21 |

### CatBoost occupancy regressor

| Split | MAE | RMSE | R² |
|---|---:|---:|---:|
| Validation | 17.14 cars | 28.32 cars | 0.9985 |
| Untouched test | 18.78 cars | 34.66 cars | 0.9978 |

`artifacts/metrics.json` stores the full-precision deployment metrics.

### Why the classifier is the warning model

The regression model is strong overall, but a single point estimate is not the same as a threshold-crossing probability. Among the **70** untouched-test rows that were below 90% and actually reached at least 90% approximately 30 minutes later, the raw CatBoost point forecast remained below 90% in **29 cases (41.43%)**.

For that reason, ParkPulse uses the Random Forest classifier for the early-warning decision and treats the CatBoost forecast as a supporting continuous occupancy estimate.

The Streamlit dashboard clips displayed regression predictions to the physical range `[0, capacity]`. Reported regression metrics are calculated from the **raw** CatBoost predictions, before display clipping.

## Supporting experiments

The extra experiments are retained as evidence of model-development decisions, not as additional deployed systems.

### Birmingham telematics context

A second University of Birmingham telematics dataset was aggregated into typical citywide day/hour traffic context and added to the parking features. On the untouched test period:

| Model | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| Parking-only Random Forest | 66.67% | 68.57% | 67.61% | 77.41% |
| Parking + telematics Random Forest | 56.99% | 75.71% | 65.03% | 75.51% |

Traffic context increased recall but reduced precision, F1, and PR-AUC, so the **parking-only Random Forest was retained**.

See `artifacts/traffic_context_experiments/` and `compare_traffic_context.py`.

### Performance optimization

A separate optimization study tested richer dynamics/history features and tuned model candidates using rolling chronological validation. The best development candidate was a CatBoost classifier, but the improvement did not generalize to the untouched test period:

| Model | Test Precision | Test Recall | Test F1 | Test PR-AUC |
|---|---:|---:|---:|---:|
| Deployed Random Forest | 66.67% | 68.57% | 67.61% | 77.41% |
| Optimized CatBoost candidate | 62.67% | 67.14% | 64.83% | 76.71% |

The simpler deployed Random Forest was therefore retained.

See `optimization_experiment/`.

## Repository layout

```text
app.py                         Streamlit historical backtest demo
train.py                       Parking-only training/evaluation pipeline
validate_project.py            Final deployment validator
compare_traffic_context.py     Second-dataset experiment runner
validate_traffic_context.py    Traffic mapping/results validator
src/                           Shared feature and traffic-context code
models/                        Two deployed model binaries
artifacts/                     Final metrics, predictions, and experiment evidence
data/                          Parking dataset and telematics download instructions
optimization_experiment/       Supporting model-optimization research
```

An older recovery-forecasting study is retained in `recovery_experiment/` as **archived exploratory work outside the final ParkPulse scope**. It is not loaded by the dashboard and is not part of the deployed decision logic.

## Setup and run

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Training is optional because the two deployed model binaries and validated artifacts are included. Re-running `train.py` may create additional non-deployed comparison binaries locally; `.gitignore` keeps those reproducible outputs out of the repository.

## Validation

Core deployment:

```powershell
python validate_project.py
```

Supporting optimization experiment:

```powershell
python optimization_experiment/validate_experiment.py
```

Traffic-context experiment, when the local `data/Year_2016.csv` file is available:

```powershell
python validate_traffic_context.py
```

## Limitations

- The data is historical and comes from 2016; this project does **not** claim that the same model represents Birmingham parking conditions today.
- The dashboard is a held-out historical simulation/backtest, not a live service.
- Parking IDs do not include travel-time or geospatial-routing information, so ParkPulse ranks lower-risk car parks rather than recommending a destination based on journey time.
- A production version would need live parking/traffic feeds, monitoring for distribution or concept drift, and periodic retraining/revalidation.

## Data attribution

See `data/README.md` for the UCI Parking Birmingham citation, licensing information, and the University of Birmingham telematics source.
