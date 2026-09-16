# Birmingham telematics comparison

This directory contains compact evidence for the experimental comparison between the deployed parking-only feature set and parking features augmented with typical University of Birmingham 2016 telematics context.

The experiment maps parking timestamps into five day groups and seven hour windows, then joins global traffic-speed and acceleration aggregates calculated with missing values ignored. It does not treat the telematics data as a live timestamp-level feed.

The validation-selected parking-only Random Forest remained the deployed classifier. On untouched test data it achieved F1 67.61% and PR-AUC 77.41%, compared with F1 65.03% and PR-AUC 75.51% for the parking-plus-telematics Random Forest. The added context increased recall but reduced precision, F1, and PR-AUC.

Retained files include mapping validation, the 35-context lookup, all-model comparison metrics, validation threshold searches, selected-model comparisons, and test predictions. Model binaries are omitted because `compare_traffic_context.py` reproduces them and `validate_traffic_context.py` validates the retained evidence.
