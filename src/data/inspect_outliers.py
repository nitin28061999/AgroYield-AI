
"""Inspect unusually high crop-yield observations."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "data" / "raw"
    / "crop-wise-area-production-yield.csv"
)

REPORT_PATH = (
    ROOT / "reports"
    / "yield_outlier_investigation.csv"
)


def inspect_outliers():
    df = pd.read_csv(DATA_PATH, low_memory=False)

    df["calculated_yield"] = (
        df["production"] / df["area"]
    )

    # Exclude records without observed production.
    valid = df.dropna(subset=["production", "yield"]).copy()

    print("=" * 70)
    print("AGROYIELD AI - YIELD OUTLIER INVESTIGATION")
    print("=" * 70)

    print("\n1. TOP 30 HIGHEST YIELD RECORDS")

    top_records = valid.nlargest(30, "yield")

    columns = [
        "year",
        "state_name",
        "district_name",
        "crop_name",
        "season",
        "area",
        "production",
        "yield",
        "calculated_yield",
    ]

    print(top_records[columns].to_string(index=False))

    print("\n2. CROP-WISE YIELD SUMMARY")

    crop_summary = (
        valid.groupby("crop_name")["yield"]
        .agg(
            records="count",
            minimum="min",
            median="median",
            mean="mean",
            maximum="max",
        )
        .sort_values("maximum", ascending=False)
    ) # pyright: ignore[reportCallIssue]

    print(crop_summary.head(20).to_string())

    print("\n3. HIGH-YIELD RECORDS BY CROP")

    for crop in ["Coconut", "Cashewnut", "Onion", "Sugarcane", "Maize"]:
        crop_data = valid[valid["crop_name"] == crop]

        if not crop_data.empty:
            print(f"\n{crop}")
            print(f"Records: {len(crop_data):,}")
            print(f"Median yield: {crop_data['yield'].median():.3f}")
            print(f"99th percentile: {crop_data['yield'].quantile(0.99):.3f}")
            print(f"Maximum yield: {crop_data['yield'].max():.3f}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    top_records[columns].to_csv(
        REPORT_PATH,
        index=False,
    )

    print(f"\nOutlier investigation saved to: {REPORT_PATH}")
    print("\nINVESTIGATION COMPLETED")


if __name__ == "__main__":
    inspect_outliers()
