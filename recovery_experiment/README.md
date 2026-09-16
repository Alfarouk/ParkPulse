# ParkPulse Recovery Classifier Experiment

This experiment predicts whether a currently saturated garage will recover below 90% occupancy approximately 30 minutes later. It was developed and validated in isolation and remains experimental. The deployed Streamlit dashboard does not load it or use recovery predictions in operational status routing.

## Leakage controls

- Eligibility: current `occupancy_ratio >= 0.90` only.
- Positive target: `future_occupancy_ratio < 0.90`.
- Features: the existing 12 parking-only ParkPulse features available at prediction time.
- Split: the existing chronological train/validation/untouched-test split.
- Model configuration, model selection, and threshold selection use validation data only.
- The selected candidate and threshold are frozen before test evaluation.
- No telematics or future-derived feature is used.

Class prevalence is 240/1,604 (14.96%) in train, 76/484 (15.70%) in validation, and 65/362 (17.96%) in test.

## Validation results

Thresholds below are each candidate's validation-F1-maximizing threshold. The final candidate is selected by the mean of validation F1 and PR-AUC, followed by F1, PR-AUC, precision/recall balance, and Brier score.

| Candidate | Threshold | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Remains-saturated baseline | 0.50 | 0.00% | 0.00% | 0.00% | 15.70% | 50.00% | 0.1570 | 408 / 0 / 76 / 0 |
| Logistic default | 0.31 | 67.50% | 71.05% | 69.23% | 75.08% | 93.07% | 0.0705 | 382 / 26 / 22 / 54 |
| Logistic balanced | 0.67 | 68.67% | 75.00% | 71.70% | 74.33% | 92.75% | 0.1003 | 382 / 26 / 19 / 57 |
| **Random Forest default — selected** | **0.32** | **72.15%** | **75.00%** | **73.55%** | **79.57%** | **93.89%** | **0.0644** | **386 / 22 / 19 / 57** |
| Random Forest balanced | 0.36 | 66.67% | 78.95% | 72.29% | 79.38% | 94.11% | 0.0645 | 378 / 30 / 16 / 60 |
| CatBoost default | 0.33 | 72.97% | 71.05% | 72.00% | 79.55% | 94.53% | 0.0655 | 388 / 20 / 22 / 54 |
| CatBoost balanced | 0.63 | 67.47% | 73.68% | 70.44% | 79.56% | 94.16% | 0.0802 | 381 / 27 / 20 / 56 |

## Untouched test results

These results were computed only after the default Random Forest and threshold 0.32 were frozen from validation. Results for other candidates are reported for transparency and did not affect selection.

| Candidate | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Remains-saturated baseline | 0.00% | 0.00% | 0.00% | 17.96% | 50.00% | 0.1796 | 297 / 0 / 65 / 0 |
| Logistic default | 67.47% | 86.15% | 75.68% | 82.69% | 94.52% | 0.0648 | 270 / 27 / 9 / 56 |
| Logistic balanced | 66.67% | 86.15% | 75.17% | 81.64% | 94.04% | 0.1119 | 269 / 28 / 9 / 56 |
| **Random Forest default — selected** | **69.86%** | **78.46%** | **73.91%** | **82.12%** | **93.59%** | **0.0683** | **275 / 22 / 14 / 51** |
| Random Forest balanced | 64.71% | 84.62% | 73.33% | 82.69% | 93.82% | 0.0689 | 267 / 30 / 10 / 55 |
| CatBoost default | 75.71% | 81.54% | 78.52% | 83.96% | 94.81% | 0.0634 | 280 / 17 / 12 / 53 |
| CatBoost balanced | 70.13% | 83.08% | 76.06% | 84.63% | 94.92% | 0.0865 | 274 / 23 / 11 / 54 |

The selected Random Forest detected 51 of 65 test recoveries and produced 22 false recovery alerts. Although CatBoost happened to score higher on test, it was not selected because its validation-only selection score was lower; choosing it from test results would violate the experiment protocol.

## Deployment decision

The selected model substantially improves on the remains-saturated baseline, but its approximately 30% false-alert share among predicted recoveries was not considered reliable enough for the deployed operational decision logic. The dashboard therefore continues to label all currently saturated garages `ALREADY SATURATED`. Recovery metrics remain available as research evidence only.

The selected experimental model binary is omitted from the final repository because `run_experiment.py` reproduces it and the metrics, threshold search, comparison table, and test predictions are retained. The validator checks those saved results without requiring the binary.

## Reproduce and validate

From the ParkPulse project root:

```powershell
.\.venv\Scripts\python.exe recovery_experiment\run_experiment.py
.\.venv\Scripts\python.exe recovery_experiment\validate_experiment.py
```

All generated files remain inside `recovery_experiment/`.
