
"""Analyze the year-wise impact of repeated observations."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "crop-wise-area-production-yield.csv"
REPORT_DIR = ROOT / "reports"


def analyze_repeated_years():
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Exclude aggregate Total season records.
    df = df[df["season"] != "Total"].copy()

    keys = [
    "state_name",
    "district_name",
    "crop_name",
    "crop_code",
    "season",
]

    # Count how many times each area-production combination
    # occurs for the same district/crop/season across years.
    repeated = (
        df.groupby(
            keys + ["area", "production"],
            dropna=False,
        )
        .size()
        .reset_index(name="occurrences")
    )

    repeated = repeated[repeated["occurrences"] >= 3]

    # Identify records belonging to repeated groups.
    repeated_keys = repeated[keys + ["area", "production"]]

    df = df.merge(
        repeated_keys.assign(is_repeated=True),
        on=keys + ["area", "production"],
        how="left",
    )

    df["is_repeated"] = df["is_repeated"].fillna(False)

    # Year-wise summary.
    year_summary = (
        df.groupby("year")
        .agg(
            total_records=("year", "size"),
            repeated_records=("is_repeated", "sum"),
        )
        .reset_index()
    )

    year_summary["repeated_percentage"] = (
        year_summary["repeated_records"]
        / year_summary["total_records"]
        * 100
    )

    print("\nYEAR-WISE REPEATED OBSERVATION SUMMARY")
    print("=" * 75)
    print(year_summary.to_string(index=False))

    print("\nOVERALL SUMMARY")
    print("=" * 75)
    print(f"Total records: {len(df):,}")
    print(f"Repeated records: {df['is_repeated'].sum():,}")
    print(
        "Repeated percentage: "
        f"{df['is_repeated'].mean() * 100:.2f}%"
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    year_summary.to_csv(
        REPORT_DIR / "repeated_observations_by_year.csv",
        index=False,
    )

    print("\nReport saved:")
    print("reports/repeated_observations_by_year.csv")


if __name__ == "__main__":
    analyze_repeated_years()
