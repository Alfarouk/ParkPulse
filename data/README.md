# Data files

## Parking Birmingham

`dataset.csv` is the main historical parking dataset used by ParkPulse and is included in this repository.

Source:

- **Dataset:** Parking Birmingham
- **Creator:** Daniel Stolfi
- **Repository:** UCI Machine Learning Repository
- **DOI:** https://doi.org/10.24432/C51K5Z
- **UCI page:** https://archive.ics.uci.edu/dataset/482/parking+birmingham
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Raw size:** 35,717 observations, 4 variables
- **Historical coverage:** 2016-10-04 through 2016-12-19

Citation:

> Stolfi, D. (2017). Parking Birmingham [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51K5Z.

The UCI record states that the dataset is licensed under CC BY 4.0. The copy at `data/dataset.csv` is redistributed under those dataset terms with attribution to the original source.

## University of Birmingham telematics data

The optional second-dataset experiment uses `Year_2016.csv` from the University of Birmingham eData record **Telematics data to analyse trades-off between air quality improvement and decarbonization strategies**.

Source:

- **Creators:** Omid Ghaffarpasand and Francis Pope
- **Publisher:** University of Birmingham
- **Dataset record / DOI:** https://doi.org/10.25500/edata.bham.00001375
- **Direct 2016 CSV:** https://edata.bham.ac.uk/1375/2/Year_2016.csv
- **Official file checksum:** `f861534b6f11bbb8dd17c3732223eb01`

Place the downloaded file at:

```text
data/Year_2016.csv
```

`Year_2016.csv` is intentionally excluded from Git because of its size. It is used only by `compare_traffic_context.py` and `validate_traffic_context.py`.

ParkPulse does **not** treat this file as a live timestamp-level traffic feed. The experiment aggregates historical telematics values into typical citywide day/hour traffic context and tests whether that additional context improves early-warning classification.
