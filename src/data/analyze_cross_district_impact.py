
"""Measure the impact of cross-district repeated values."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "reports" / "cross_district_repeated_values.csv"
OUTPUT_PATH = ROOT / "reports" / "cross_district_impact_by_crop.csv"


def analyze_cross_district_impact():
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    # Summarize each crop and crop code separately.
    summary = (
        df.groupby(
            ["crop_name", "crop_code"],
            dropna=False,
        )
        .agg(
            repeated_groups=("record_count", "size"),
            total_records_in_groups=("record_count", "sum"),
            max_districts=("district_count", "max"),
            groups_with_2_districts=(
                "district_count",
                lambda x: (x == 2).sum(),
            ),
            groups_with_3_to_4_districts=(
                "district_count",
                lambda x: x.between(3, 4).sum(),
            ),
            groups_with_5plus_districts=(
                "district_count",
                lambda x: (x >= 5).sum(),
            ),
            missing_production_groups=(
                "production",
                lambda x: x.isna().sum(),
            ),
            zero_production_groups=(
                "production",
                lambda x: (x == 0).sum(),
            ),
        )
        .reset_index()
        .sort_values(
            ["repeated_groups", "max_districts"],
            ascending=False,
        )
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)

    print("\nCROSS-DISTRICT IMPACT BY CROP")
    print("=" * 90)
    print(summary.to_string(index=False))

    print("\nOVERALL SUMMARY")
    print("=" * 90)
    print("Total repeated groups:", len(df))
    print(
        "Groups involving 5+ districts:",
        (df["district_count"] >= 5).sum(),
    )
    print(
        "Groups involving 3+ districts:",
        (df["district_count"] >= 3).sum(),
    )
    print(
        "Groups with missing production:",
        df["production"].isna().sum(),
    )
    print(
        "Groups with zero production:",
        (df["production"] == 0).sum(),
    )

    print(f"\nReport saved: {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    analyze_cross_district_impact()
