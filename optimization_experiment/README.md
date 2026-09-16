# ParkPulse performance-optimization experiment

This directory is isolated from the deployed ParkPulse system. The experiment keeps the existing true early-warning target and chronological test period unchanged, selects features, hyperparameters, model and threshold using three expanding chronological validation folds, freezes the winning configuration, and only then evaluates the untouched test period.

## Result

The rolling-validation winner was CatBoost with a 4× positive-class weight, the `dynamics_history` feature set and a final validation-selected threshold of `0.45`. It improved rolling validation, but the gain did not generalize to the untouched test period. The deployed parking-only Random Forest should remain in place.

| Model | Split | Precision | Recall | F1 | PR-AUC |
|---|---|---:|---:|---:|---:|
| Deployed Random Forest | Validation | 0.6571 | 0.7753 | 0.7113 | 0.7897 |
| Experimental CatBoost | Validation | 0.6800 | 0.7640 | 0.7196 | 0.8000 |
| Deployed Random Forest | Test | 0.6667 | 0.6857 | 0.6761 | 0.7741 |
| Experimental CatBoost | Test | 0.6267 | 0.6714 | 0.6483 | 0.7671 |

The experimental model reduced test precision, recall, F1 and PR-AUC. Bootstrap intervals also do not establish a statistically reliable F1 or PR-AUC improvement.

## Added feature groups

- Distance to 90% and occupancy changes over 30, 60 and 90 minutes.
- Rolling occupancy slopes over 60 and 90 minutes.
- Occupancy acceleration and rolling standard deviation.
- Estimated minutes to 90%, using only positive recent growth and a safe 999-minute value for zero, negative or missing growth.
- Garage-specific same-weekday/time mean, median, standard deviation, 90th percentile and observation count. Training rows use shifted expanding history; validation and test use training-period aggregates only.

## Candidate models

- Existing Random Forest.
- Random Forest class-weight, depth and leaf-size variants.
- CatBoost with no class weights, 2×, 4× and 8× positive weights, and automatic balancing.
- Balanced Random Forest was skipped because `imbalanced-learn` is not installed.
- XGBoost was skipped because `xgboost` is not installed.

## Alternative target assessment

The median non-overnight sampling interval is 30 minutes, only about 50.5% of eligible rows have an observation within the next 30 minutes, and none has two observations in that window. The data therefore cannot reliably distinguish a transient crossing from occupancy at the next observation. The target was not changed.

Run `python optimization_experiment/run_experiment.py` to reproduce the experiment and `python optimization_experiment/validate_experiment.py` to validate saved results and deployment preservation.

The experimental winning-model binary is omitted from the final repository. The runner reproduces it, while the frozen selection, rolling comparisons, metrics, and untouched-test predictions retain the academic evidence and are sufficient for validation.
