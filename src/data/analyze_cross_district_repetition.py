
"""Detect repeated crop values across different districts."""

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


def analyze_cross_district_repetition():
    df = pd.read_csv(DATA_PATH, low_memory=False)

    df = df[
        (df["season"] != "Total")
        & df["crop_name"].isin(PRIORITY_CROPS)
    ].copy()

    keys = [
        "year",
        "state_name",
        "crop_name",
        "crop_code",
        "season",
        "area",
        "production",
    ]

    result = (
        df.groupby(keys, dropna=False)
        .agg(
            district_count=("district_name", "nunique"),
            districts=(
                "district_name",
                lambda values: " | ".join(
                    sorted(set(values.dropna().astype(str)))
                ),
            ),
            record_count=("district_name", "size"),
        )
        .reset_index()
    )

    result = result[
        result["district_count"] >= 2
    ].copy()

    result = result.sort_values(
        ["district_count", "record_count"],
        ascending=False,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORT_DIR / "cross_district_repeated_values.csv"
    result.to_csv(output_path, index=False)

    print("\nCROSS-DISTRICT REPEATED VALUE SUMMARY")
    print("=" * 75)
    print(f"Groups with identical values across districts: {len(result)}")
    print(
        "Distinct crop-year combinations:",
        result.groupby(
            ["crop_name", "crop_code", "year"],
            dropna=False,
        ).ngroups,
    )

    print("\nTOP 25 CROSS-DISTRICT MATCHES")
    print("=" * 75)

    print(
        result[
            [
                "year",
                "state_name",
                "crop_name",
                "crop_code",
                "season",
                "area",
                "production",
                "district_count",
                "record_count",
                "districts",
            ]
        ].head(25).to_string(index=False)
    )

    print(f"\nReport saved: {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    analyze_cross_district_repetition()
