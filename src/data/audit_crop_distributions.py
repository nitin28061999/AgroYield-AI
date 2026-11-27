
"""Audit crop-wise yield distributions and extreme observations."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "data" / "raw"
    / "crop-wise-area-production-yield.csv"
)

REPORT_DIR = ROOT / "reports"


def audit_crop_distributions():
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Remove aggregate Total rows for this distribution audit.
    # Keep the original raw CSV unchanged.
    df = df[df["season"] != "Total"].copy()

    df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    df["production"] = pd.to_numeric(
        df["production"], errors="coerce"
    )

    df = df.dropna(subset=["yield"])

    print("=" * 75)
    print("AGROYIELD AI - CROP-WISE YIELD DISTRIBUTION AUDIT")
    print("=" * 75)

    # 1. Crop-wise distribution statistics.
    summary = (
        df.groupby("crop_name")["yield"]
        .agg(
            records="count",
            minimum="min",
            p50="median",
            p90=lambda x: x.quantile(0.90),
            p95=lambda x: x.quantile(0.95),
            p99=lambda x: x.quantile(0.99),
            maximum="max",
        )
        .sort_values("maximum", ascending=False)
    ) # pyright: ignore[reportCallIssue]

    summary = summary.round(3)

    print("\n1. TOP 25 CROPS BY MAXIMUM YIELD")
    print(summary.head(25).to_string())

    # 2. Inspect crops with unusually large maximum-to-median ratios.
    positive_median = summary["p50"] > 0

    summary["max_to_median_ratio"] = (
        summary["maximum"] / summary["p50"].where(positive_median)
    )

    print("\n2. CROPS WITH HIGHEST MAXIMUM / MEDIAN RATIOS")
    print(
        summary.sort_values(
            "max_to_median_ratio", ascending=False
        ).head(20).round(2).to_string()
    )

    # 3. Inspect detailed records for selected crops.
    selected_crops = [
        "Coconut",
        "Cashewnut",
        "Onion",
        "Sugarcane",
        "Maize",
    ]

    columns = [
        "year",
        "state_name",
        "district_name",
        "crop_name",
        "season",
        "area",
        "production",
        "yield",
        "production_unit",
        "yield_unit",
    ]

    print("\n3. TOP 5 YIELD RECORDS FOR SELECTED CROPS")

    for crop in selected_crops:
        crop_df = df[df["crop_name"] == crop]

        if crop_df.empty:
            continue

        print(f"\n--- {crop} ---")

        print(
            crop_df.nlargest(5, "yield")[columns]
            .to_string(index=False)
        )

    # 4. Save audit reports.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        REPORT_DIR / "crop_yield_distribution_summary.csv"
    )

    print("\nReports saved:")
    print("reports/crop_yield_distribution_summary.csv")

    print("\nAUDIT COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    audit_crop_distributions()
