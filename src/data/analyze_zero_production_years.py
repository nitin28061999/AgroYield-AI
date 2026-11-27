
"""Analyze the temporal distribution of zero-production records."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "data" / "raw" / "crop-wise-area-production-yield.csv"

YEAR_REPORT = ROOT / "reports" / "zero_production_by_year.csv"
CROP_YEAR_REPORT = ROOT / "reports" / "zero_production_by_crop_year.csv"


def analyze_zero_production_years():  # sourcery skip: extract-duplicate-method
    df = pd.read_csv(RAW_PATH, low_memory=False)

    df = df[df["season"] != "Total"].copy()
    zero = df[df["production"] == 0].copy()

    print("\nZERO PRODUCTION: YEAR ANALYSIS")
    print("=" * 75)

    year_summary = (
        zero.groupby("year")
        .agg(
            zero_records=("id", "size"),
            crops=("crop_code", "nunique"),
            states=("state_name", "nunique"),
            districts=("district_name", "nunique"),
        )
        .reset_index()
        .sort_values("year")
    )

    print(year_summary.to_string(index=False))
    year_summary.to_csv(YEAR_REPORT, index=False)

    print("\nTOP CROP-YEAR COMBINATIONS")
    print("=" * 75)

    crop_year = (
        zero.groupby(
            ["crop_name", "crop_code", "year"],
            dropna=False,
        )
        .agg(
            zero_records=("id", "size"),
            states=("state_name", "nunique"),
            districts=("district_name", "nunique"),
            total_area=("area", "sum"),
            median_area=("area", "median"),
        )
        .reset_index()
        .sort_values("zero_records", ascending=False)
    )

    print(crop_year.head(30).to_string(index=False))
    crop_year.to_csv(CROP_YEAR_REPORT, index=False)

    print("\nAREA DISTRIBUTION")
    print("=" * 75)
    print(zero["area"].describe().to_string())

    print(f"\nYear report saved: {YEAR_REPORT.relative_to(ROOT)}")
    print(f"Crop-year report saved: {CROP_YEAR_REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    analyze_zero_production_years()
