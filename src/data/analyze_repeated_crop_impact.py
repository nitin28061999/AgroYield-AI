
"""Identify crops most affected by repeated observations."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "crop-wise-area-production-yield.csv"
REPORT_DIR = ROOT / "reports"


def analyze_repeated_crop_impact():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[df["season"] != "Total"].copy()

    keys = [
        "state_name",
        "district_name",
        "crop_name",
        "crop_code",
        "season",
    ]

    repeated_groups = (
        df.groupby(
            keys + ["area", "production"],
            dropna=False,
        )
        .size()
        .reset_index(name="occurrences")
    )

    repeated_groups = repeated_groups[
        repeated_groups["occurrences"] >= 3
    ]

    repeated_keys = repeated_groups[keys + ["area", "production"]]

    df = df.merge(
        repeated_keys.assign(is_repeated=True),
        on=keys + ["area", "production"],
        how="left",
    )

    df["is_repeated"] = df["is_repeated"].fillna(False)

    crop_summary = (
    df.groupby(
        ["crop_name", "crop_code"],
        dropna=False,
    )
    .agg(
        total_records=("crop_name", "size"),
        repeated_records=("is_repeated", "sum"),
    )
    .reset_index()
)

    crop_summary["repeated_percentage"] = (
        crop_summary["repeated_records"]
        / crop_summary["total_records"]
        * 100
    )

    crop_summary = crop_summary.sort_values(
        "repeated_records",
        ascending=False,
    )

    print("\nTOP 25 CROPS BY REPEATED RECORD COUNT")
    print("=" * 75)
    print(crop_summary.head(25).to_string(index=False))

    print("\nTOP 25 CROPS BY REPEATED PERCENTAGE")
    print("=" * 75)
    print(
        crop_summary.sort_values(
            "repeated_percentage",
            ascending=False,
        ).head(25).to_string(index=False)
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    crop_summary.to_csv(
        REPORT_DIR / "repeated_observations_by_crop.csv",
        index=False,
    )

    print("\nReport saved:")
    print("reports/repeated_observations_by_crop.csv")


if __name__ == "__main__":
    analyze_repeated_crop_impact()
