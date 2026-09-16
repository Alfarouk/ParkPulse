# ParkPulse

ParkPulse is an academic historical/backtest prototype for Birmingham parking occupancy. It is not connected to live Birmingham data. Given a historical observation, it estimates occupied spaces about 30 minutes later and issues a true early warning only when a garage is currently below 90% occupancy but is at risk of reaching at least 90%.

## Final deployed system

- **Early-warning classifier:** parking-only Random Forest, loaded from `models/random_forest_classifier.joblib`.
- **Alert threshold:** `0.21`, selected by maximum F1 on the chronological validation set.
- **Eligibility:** current occupancy ratio below `0.90` only.
- **Statuses:** `NORMAL`, `EARLY WARNING`, or `ALREADY SATURATED`.
- **Occupancy regressor:** CatBoost, loaded from `models/catboost_regressor.cbm`, applied to all rows.

For classifier-eligible rows, the positive target is future occupancy ratio at least `0.90` approximately 30 minutes later. Rows already at least 90% occupied are excluded from classifier training, validation, and testing rather than converted into negative examples. The Streamlit dashboard marks them `ALREADY SATURATED` and displays the early-warning probability as not applicable.

## Leakage-safe methodology

Parking observations are ordered chronologically and split into train, validation, and untouched test periods. Future labels are kept within their split boundaries. Lag and change features use only information available at prediction time. Model, feature, hyperparameter, and threshold choices use training/validation data only; the test period is reserved for final evaluation.

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

`artifacts/metrics.json` is the source of truth for full-precision values. `artifacts/model_summary.csv` provides the deployment/experiment inventory without requiring non-deployed model binaries.

## Retained experiments

### Birmingham telematics second dataset

`compare_traffic_context.py`, `validate_traffic_context.py`, `src/traffic_context.py`, and `artifacts/traffic_context_experiments/` preserve the day/hour mapping, traffic aggregates, validation evidence, threshold searches, and model comparisons. On untouched test data, parking-only Random Forest achieved F1 67.61% and PR-AUC 77.41%; parking plus telematics achieved F1 65.03% and PR-AUC 75.51%. The telematics features therefore remain experimental and are not loaded by the app.

The large `data/Year_2016.csv` file is intentionally ignored by Git. See `data/README.md` for its University of Birmingham source and placement instructions.

### Performance optimization

`optimization_experiment/` preserves the runner, validator, frozen validation-only selection, rolling comparisons, feature manifests, target-feasibility analysis, and untouched-test predictions. The optimized candidate improved development/rolling validation but did not improve the untouched test period, so it did not replace the deployed Random Forest.

### Recovery forecasting

`recovery_experiment/` preserves the runner, validator, metrics, model comparison, threshold search, and selected-model test predictions. It studied whether already-saturated garages would fall below 90% about 30 minutes later. The validation-selected Random Forest achieved test precision 69.86%, recall 78.46%, F1 73.91%, and PR-AUC 82.12% at threshold 0.32. Recovery forecasting remains experimental and is not part of deployed decision routing.

Non-deployed experiment binaries are omitted from the submission because the retained runners reproduce them and the compact metrics/predictions preserve the evidence.

## Repository layout

```text
app.py                         Streamlit historical demo
train.py                       Parking-only training pipeline
validate_project.py            Final deployment validator
compare_traffic_context.py     Telematics experiment runner
validate_traffic_context.py    Telematics mapping/results validator
src/                           Shared feature and traffic-context code
models/                        Two deployed model binaries only
artifacts/                     Final metrics, predictions, and experiment evidence
data/                          Parking data and local telematics instructions
optimization_experiment/       Isolated optimization research
recovery_experiment/           Isolated recovery research
```

## Setup and run

From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The included `data/dataset.csv` is the original Parking Birmingham dataset. Training is optional because the two deployed model binaries and validated artifacts are included.

## Validation

```powershell
python validate_project.py
python validate_traffic_context.py
python optimization_experiment/validate_experiment.py
python recovery_experiment/validate_experiment.py
```

The traffic validator requires the locally downloaded `data/Year_2016.csv`. Experimental model binaries are not required by the validators.
