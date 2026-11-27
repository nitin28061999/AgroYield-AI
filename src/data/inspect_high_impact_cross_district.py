
"""Inspect cross-district repetitions involving five or more districts."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "reports" / "cross_district_repeated_values.csv"
OUTPUT_PATH = ROOT / "reports" / "high_impact_cross_district.csv"


def inspect_high_impact_groups():
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    high_impact = df[
        df["district_count"] >= 5
    ].copy()

    high_impact = high_impact.sort_values(
        ["crop_name", "state_name", "district_count", "year"],
        ascending=[True, True, False, True],
    )

    high_impact.to_csv(OUTPUT_PATH, index=False)

    print("\nHIGH-IMPACT CROSS-DISTRICT GROUPS")
    print("=" * 100)
    print("Total groups:", len(high_impact))

    print("\nCROP AND STATE SUMMARY")
    print("=" * 100)

    summary = (
        high_impact.groupby(
            ["crop_name", "crop_code", "state_name"],
            dropna=False,
        )
        .agg(
            repeated_groups=("record_count", "size"),
            max_districts=("district_count", "max"),
            distinct_years=("year", "nunique"),
            missing_production_groups=(
                "production",
                lambda x: x.isna().sum(),
            ),
        )
        .reset_index()
        .sort_values(
            ["repeated_groups", "max_districts"],
            ascending=False,
        )
    )

    print(summary.to_string(index=False))

    print("\nDETAILED HIGH-IMPACT RECORDS")
    print("=" * 100)

    columns = [
        "year",
        "state_name",
        "crop_name",
        "crop_code",
        "season",
        "area",
        "production",
        "district_count",
        "districts",
    ]

    print(high_impact[columns].to_string(index=False))

    print(f"\nReport saved: {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    inspect_high_impact_groups()
