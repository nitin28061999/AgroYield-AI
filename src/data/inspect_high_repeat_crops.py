
"""Inspect repeated observations in high-priority crops."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "crop-wise-area-production-yield.csv"
REPORT_DIR = ROOT / "reports"

PRIORITY_CROPS = [
    "Cardamom",
    "Castor Seed",
    "Tobacco",
    "Mesta",
    "Niger Seed",
    "Linseed",
]


def inspect_high_repeat_crops():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[df["season"] != "Total"].copy()
    df = df[df["crop_name"].isin(PRIORITY_CROPS)].copy()

    keys = [
        "state_name",
        "district_name",
        "crop_name",
        "crop_code",
        "season",
    ]

    repeated = (
        df.groupby(
            keys + ["area", "production"],
            dropna=False,
        )
        .size()
        .reset_index(name="occurrences")
    )

    repeated = repeated[repeated["occurrences"] >= 3].copy()

    repeated["production_missing"] = (
        repeated["production"].isna()
    )

    repeated["simple_area_value"] = repeated["area"].isin(
        [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    )

    repeated["simple_production_value"] = (
        repeated["production"].isin(
            [1, 5, 10, 20, 50, 100, 200, 500, 1000]
        )
    )

    print("\nREPEATED GROUPS BY PRIORITY CROP")
    print("=" * 75)

    summary = (
        repeated.groupby(
            ["crop_name", "crop_code"],
            dropna=False,
        )
        .agg(
            repeated_groups=("occurrences", "size"),
            records_in_groups=("occurrences", "sum"),
            missing_production_groups=("production_missing", "sum"),
            simple_area_groups=("simple_area_value", "sum"),
            simple_production_groups=(
                "simple_production_value", "sum"
            ),
        )
        .reset_index()
    )

    print(summary.to_string(index=False))

    for crop in PRIORITY_CROPS:
        crop_data = repeated[repeated["crop_name"] == crop]

        print(f"\n\nTOP REPEATED GROUPS: {crop.upper()}")
        print("-" * 75)

        print(
            crop_data.sort_values(
                "occurrences",
                ascending=False,
            ).head(10).to_string(index=False)
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    repeated.to_csv(
        REPORT_DIR / "high_priority_repeated_groups.csv",
        index=False,
    )

    summary.to_csv(
        REPORT_DIR / "high_priority_repeated_summary.csv",
        index=False,
    )

    print("\nReports saved:")
    print("reports/high_priority_repeated_groups.csv")
    print("reports/high_priority_repeated_summary.csv")


if __name__ == "__main__":
    inspect_high_repeat_crops()
