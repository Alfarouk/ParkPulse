# Data files

## Parking Birmingham

`dataset.csv` is the small Parking Birmingham dataset used by the main ParkPulse pipeline and is included in the repository.

## University of Birmingham telematics data

The second-dataset experiment uses `Year_2016.csv`, the University of Birmingham 2016 telematics dataset containing aggregated speed and acceleration characteristics for Birmingham road segments and day/hour groups.

Download it from the University of Birmingham eData record:

- Dataset record and DOI: https://doi.org/10.25500/edata.bham.00001375
- Direct 2016 CSV: https://edata.bham.ac.uk/1375/2/Year_2016.csv

Place the downloaded file at:

```text
data/Year_2016.csv
```

The file is used only by `compare_traffic_context.py` and `validate_traffic_context.py`. It is intentionally excluded from Git because of its size; keeping it locally allows the experiment and validator to be reproduced.
