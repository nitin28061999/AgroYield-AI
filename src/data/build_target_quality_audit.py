
"""Classify crop records by yield-target quality."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "data" / "raw" / "crop-wise-area-production-yield.csv"
OUTPUT_PATH = ROOT / "reports" / "target_quality_audit.csv"


def build_audit():
    df = pd.read_csv(RAW_PATH, low_memory=False)

    df = df[df["season"] != "Total"].copy()

    df["calculated_yield"] = df["production"] / df["area"]

    conditions = [
        df["production"].isna(),
        (df["production"] == 0),
        (df["production"] > 0) & (df["yield"] == 0),
        (df["production"] > 0) & (df["yield"] > 0),
    ]

    labels = [
        "missing_production",
        "zero_production",
        "positive_production_rounded_zero_yield",
        "positive_production_positive_yield",
    ]

    df["target_quality"] = pd.Series(
        pd.NA, index=df.index, dtype="object"
    )

    for condition, label in zip(conditions, labels):
        df.loc[condition, "target_quality"] = label

    summary = (
        df.groupby("target_quality", dropna=False)
        .agg(
            records=("id", "size"),
            crops=("crop_code", "nunique"),
            states=("state_name", "nunique"),
        )
        .reset_index()
    )

    print("\nTARGET QUALITY AUDIT")
    print("=" * 75)
    print(summary.to_string(index=False))

    print("\nPRECISION-LIMITED RECORDS")
    print("=" * 75)

    precision = df[
        df["target_quality"]
        == "positive_production_rounded_zero_yield"
    ]

    print("Records:", len(precision))
    print(
        "Maximum calculated yield:",
        precision["calculated_yield"].max(),
    )
    print(
        "Minimum calculated yield:",
        precision["calculated_yield"].min(),
    )

    summary.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSummary saved: {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    build_audit()
