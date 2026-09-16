from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DAY_GROUPS = ("D0", "D1", "D2.3.4", "D5", "D6")
HOUR_BUCKETS = ("H0.6", "H7.8", "H9.11", "H12.13", "H14.15", "H16.18", "H19.22")

DAY_GROUP_BY_WEEKDAY = {
    6: "D0",       # Sunday
    0: "D1",       # Monday
    1: "D2.3.4",   # Tuesday
    2: "D2.3.4",   # Wednesday
    3: "D2.3.4",   # Thursday
    4: "D5",       # Friday
    5: "D6",       # Saturday
}

HOUR_BUCKET_BY_HOUR = {
    **{hour: "H0.6" for hour in range(0, 7)},
    **{hour: "H7.8" for hour in range(7, 9)},
    **{hour: "H9.11" for hour in range(9, 12)},
    **{hour: "H12.13" for hour in range(12, 14)},
    **{hour: "H14.15" for hour in range(14, 16)},
    **{hour: "H16.18" for hour in range(16, 19)},
    **{hour: "H19.22" for hour in range(19, 23)},
}

# Speeds in the telematics source are metres per second; 10 m/s is 36 km/h.
SLOW_ROAD_SPEED_THRESHOLD = 10.0

TRAFFIC_FEATURES = [
    "traffic_speed_mean",
    "traffic_speed_median",
    "traffic_speed_std",
    "traffic_acceleration_mean",
    "traffic_slow_road_fraction",
    "traffic_speed_major",
    "traffic_speed_minor",
    "traffic_speed_residential",
]

MAPPING_COLUMNS = [
    "traffic_day_group",
    "traffic_hour_bucket",
    "traffic_context_key",
    "traffic_speed_source_column",
    "traffic_acceleration_source_column",
]


def _expected_hour_bucket(hours: pd.Series) -> pd.Series:
    conditions = [
        hours.between(0, 6),
        hours.between(7, 8),
        hours.between(9, 11),
        hours.between(12, 13),
        hours.between(14, 15),
        hours.between(16, 18),
        hours.between(19, 22),
    ]
    return pd.Series(
        np.select(conditions, HOUR_BUCKETS, default=None),
        index=hours.index,
        dtype="string",
    )


def add_traffic_mapping_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map each timestamp to the matching day group and telematics hour window."""
    out = df.copy()
    timestamps = pd.to_datetime(out["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Traffic mapping requires valid parking timestamps.")

    out["traffic_day_group"] = timestamps.dt.dayofweek.map(DAY_GROUP_BY_WEEKDAY).astype("string")
    out["traffic_hour_bucket"] = timestamps.dt.hour.map(HOUR_BUCKET_BY_HOUR).astype("string")
    unmapped = out["traffic_day_group"].isna() | out["traffic_hour_bucket"].isna()
    if unmapped.any():
        bad = out.loc[unmapped, ["timestamp"]].head().to_dict("records")
        raise ValueError(
            "Parking timestamps outside the supported telematics day/hour groups: "
            f"{bad}"
        )

    out["traffic_context_key"] = out["traffic_day_group"] + out["traffic_hour_bucket"]
    out["traffic_speed_source_column"] = "v_" + out["traffic_context_key"]
    out["traffic_acceleration_source_column"] = "a_" + out["traffic_context_key"]
    return out


def validate_timestamp_mapping(mapped: pd.DataFrame) -> dict[str, int]:
    """Independently verify every mapped parking timestamp against the stated rules."""
    timestamps = pd.to_datetime(mapped["timestamp"], errors="coerce")
    expected_days = timestamps.dt.day_name().map({
        "Sunday": "D0",
        "Monday": "D1",
        "Tuesday": "D2.3.4",
        "Wednesday": "D2.3.4",
        "Thursday": "D2.3.4",
        "Friday": "D5",
        "Saturday": "D6",
    }).astype("string")
    expected_hours = _expected_hour_bucket(timestamps.dt.hour)
    expected_keys = expected_days + expected_hours

    checks = {
        "day_group_mismatches": int((mapped["traffic_day_group"] != expected_days).sum()),
        "hour_bucket_mismatches": int((mapped["traffic_hour_bucket"] != expected_hours).sum()),
        "context_key_mismatches": int((mapped["traffic_context_key"] != expected_keys).sum()),
        "speed_column_mismatches": int(
            (mapped["traffic_speed_source_column"] != "v_" + expected_keys).sum()
        ),
        "acceleration_column_mismatches": int(
            (mapped["traffic_acceleration_source_column"] != "a_" + expected_keys).sum()
        ),
    }
    if any(checks.values()):
        raise AssertionError(f"Traffic timestamp mapping failed: {checks}")
    return {"validated_rows": len(mapped), **checks}


def validate_mapping_boundaries() -> dict[str, int]:
    """Check all weekday rules and every supported hour boundary from 00 through 22."""
    dates = pd.date_range("2016-01-04", periods=7, freq="D")  # Monday through Sunday.
    timestamps = [
        date + pd.Timedelta(hours=hour)
        for date in dates
        for hour in range(23)
    ]
    mapped = add_traffic_mapping_columns(pd.DataFrame({"timestamp": timestamps}))
    result = validate_timestamp_mapping(mapped)
    result["validated_weekdays"] = int(mapped["timestamp"].dt.dayofweek.nunique())
    result["validated_hours"] = int(mapped["timestamp"].dt.hour.nunique())
    result["validated_context_keys"] = int(mapped["traffic_context_key"].nunique())
    return result


def expected_source_columns() -> list[str]:
    keys = [f"{day_group}{hour_bucket}" for day_group in DAY_GROUPS for hour_bucket in HOUR_BUCKETS]
    return ["Road Type", *[f"v_{key}" for key in keys], *[f"a_{key}" for key in keys]]


def build_traffic_context_lookup(csv_path: str | Path) -> pd.DataFrame:
    """Aggregate typical citywide traffic conditions for each day/hour context."""
    csv_path = Path(csv_path)
    required = expected_source_columns()
    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"Telematics CSV is missing required columns: {missing}")

    telematics = pd.read_csv(csv_path, usecols=required, low_memory=False)
    numeric_columns = [column for column in required if column != "Road Type"]
    telematics[numeric_columns] = telematics[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    road_type = telematics["Road Type"].astype("string").str.strip().str.lower()

    rows = []
    for day_group in DAY_GROUPS:
        for hour_bucket in HOUR_BUCKETS:
            key = f"{day_group}{hour_bucket}"
            speed_column = f"v_{key}"
            acceleration_column = f"a_{key}"
            speed = telematics[speed_column]
            acceleration = telematics[acceleration_column]
            valid_speed = speed.dropna()
            valid_acceleration = acceleration.dropna()

            rows.append({
                "traffic_day_group": day_group,
                "traffic_hour_bucket": hour_bucket,
                "traffic_context_key": key,
                "traffic_speed_source_column": speed_column,
                "traffic_acceleration_source_column": acceleration_column,
                "traffic_speed_mean": float(valid_speed.mean()),
                "traffic_speed_median": float(valid_speed.median()),
                "traffic_speed_std": float(valid_speed.std(ddof=1)),
                "traffic_acceleration_mean": float(valid_acceleration.mean()),
                "traffic_slow_road_fraction": float(
                    (valid_speed < SLOW_ROAD_SPEED_THRESHOLD).mean()
                ),
                "traffic_speed_major": float(speed[road_type == "major"].mean(skipna=True)),
                "traffic_speed_minor": float(speed[road_type == "minor"].mean(skipna=True)),
                "traffic_speed_residential": float(
                    speed[road_type == "residential"].mean(skipna=True)
                ),
                "traffic_speed_valid_segments": int(valid_speed.size),
                "traffic_speed_missing_segments": int(speed.isna().sum()),
                "traffic_acceleration_valid_segments": int(valid_acceleration.size),
                "traffic_acceleration_missing_segments": int(acceleration.isna().sum()),
                "traffic_slow_speed_threshold_mps": SLOW_ROAD_SPEED_THRESHOLD,
            })

    lookup = pd.DataFrame(rows)
    if lookup["traffic_context_key"].duplicated().any() or len(lookup) != 35:
        raise AssertionError("Traffic lookup must contain exactly 35 unique day/hour contexts.")
    return lookup


def attach_traffic_context(df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """Attach aggregate context by key without filling between unrelated road segments."""
    out = add_traffic_mapping_columns(df)
    lookup_by_key = lookup.set_index("traffic_context_key")
    missing_keys = sorted(set(out["traffic_context_key"]) - set(lookup_by_key.index))
    if missing_keys:
        raise ValueError(f"No telematics aggregates exist for keys: {missing_keys}")

    for feature in TRAFFIC_FEATURES:
        out[feature] = out["traffic_context_key"].map(lookup_by_key[feature])
    validate_timestamp_mapping(out)
    return out


def validate_lookup_source_columns(
    lookup: pd.DataFrame,
    available_columns: Iterable[str],
) -> None:
    available = set(available_columns)
    referenced = set(lookup["traffic_speed_source_column"]) | set(
        lookup["traffic_acceleration_source_column"]
    )
    missing = sorted(referenced - available)
    if missing:
        raise AssertionError(f"Lookup references missing telematics columns: {missing}")
