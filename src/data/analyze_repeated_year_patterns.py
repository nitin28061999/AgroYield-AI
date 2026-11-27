
"""Analyze the year distribution of repeated crop observations."""

import itertools
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


def max_consecutive_run(years):
    start_years = sorted(
        {
            int(str(year).split("-")[0])
            for year in years
        }
    )

    if not start_years:
        return 0

    longest = 1
    current = 1

    for previous, current_year in itertools.pairwise(start_years):
        if current_year == previous + 1:
            current += 1
        else:
            current = 1

        longest = max(longest, current)

    return longest


def analyze_repeated_year_patterns():
    df = pd.read_csv(DATA_PATH, low_memory=False)

    df = df[
        (df["season"] != "Total")
        & df["crop_name"].isin(PRIORITY_CROPS)
    ].copy()

    keys = [
        "state_name",
        "district_name",
        "crop_name",
        "crop_code",
        "season",
        "area",
        "production",
    ]

    patterns = (
        df.groupby(keys, dropna=False)
        .agg(
            occurrences=("year", "size"),
            unique_years=("year", "nunique"),
            years=(
                "year",
                lambda values: " | ".join(
                    sorted(
                        set(values.astype(str)),
                        key=lambda value: int(value.split("-")[0]),
                    )
                ),
            ),
            max_rows_in_one_year=(
                "year",
                lambda values: values.value_counts().max(),
            ),
        )
        .reset_index()
    )

    patterns = patterns[
        patterns["occurrences"] >= 3
    ].copy()

    patterns["max_consecutive_year_run"] = patterns[
        "years"
    ].apply(
        lambda value: max_consecutive_run(value.split(" | "))
    )

    patterns["production_missing"] = (
        patterns["production"].isna()
    )

    patterns["same_year_extra_records"] = (
        patterns["occurrences"] - patterns["unique_years"]
    )

    patterns = patterns.sort_values(
        ["unique_years", "occurrences"],
        ascending=False,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        REPORT_DIR / "high_priority_repeated_temporal_patterns.csv"
    )

    patterns.to_csv(output_path, index=False)

    print("\nREPEATED OBSERVATION TEMPORAL SUMMARY")
    print("=" * 75)

    print(
        patterns.groupby(
            ["crop_name", "crop_code"],
            dropna=False,
        )
        .agg(
            repeated_groups=("occurrences", "size"),
            max_unique_years=("unique_years", "max"),
            max_consecutive_run=(
                "max_consecutive_year_run",
                "max",
            ),
            groups_with_same_year_extras=(
                "same_year_extra_records",
                lambda values: (values > 0).sum(),
            ),
        )
        .reset_index()
        .to_string(index=False)
    )

    print("\nTOP 15 LONGEST REPEATED YEAR PATTERNS")
    print("=" * 75)

    columns = [
        "crop_name",
        "crop_code",
        "state_name",
        "district_name",
        "season",
        "area",
        "production",
        "occurrences",
        "unique_years",
        "max_consecutive_year_run",
        "same_year_extra_records",
        "years",
    ]

    print(patterns[columns].head(15).to_string(index=False))

    print(f"\nReport saved: {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    analyze_repeated_year_patterns()
