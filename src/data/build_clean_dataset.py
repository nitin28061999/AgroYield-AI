
"""Build a target-quality-controlled dataset without modifying raw data."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "data" / "raw" / "crop-wise-area-production-yield.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "crop_yield_quality_controlled.csv"
SUMMARY_PATH = ROOT / "reports" / "final_target_quality_summary.csv"


def main():  # sourcery skip: extract-duplicate-method
    df = pd.read_csv(RAW_PATH, low_memory=False)

    # Exclude overlapping aggregate season rows.
    df = df[df["season"] != "Total"].copy()

    # Calculate the target independently for auditing.
    df["calculated_yield"] = df["production"] / df["area"]

    df["target_quality"] = "review_required"

    missing = df["production"].isna()
    zero = df["production"].eq(0)
    positive = df["production"].gt(0)

    df.loc[missing, "target_quality"] = "missing_production"
    df.loc[zero, "target_quality"] = "zero_production"
    df.loc[
        positive & (df["yield"] > 0),
        "target_quality",
    ] = "valid_positive_yield"

    df.loc[
        positive & (df["yield"] == 0),
        "target_quality",
    ] = "positive_production_rounded_zero_yield"

    # Flag suspicious 2002-04 crop identities discovered in the audit.
    anomaly_crops = [
        (726, "Ash Gourd"),
        (725, "Bitter Gourd"),
        (723, "Bottle Gourd"),
        (721, "Cauliflower"),
        (722, "Cucumber"),
        (788, "Lab-Lab"),
        (605, "Other Citrus Fruit"),
        (631, "Pome Granet"),
        (731, "Ribed Guard"),
        (733, "Snak Guard"),
        (624, "Water Melon"),
        (704, "Yam"),
    ]

    anomaly_mask = (
        df["year"].isin(["2002-2003", "2003-2004"])
        & df.set_index(["crop_code", "crop_name"]).index.isin(anomaly_crops)
        & df["production"].eq(0)
    )

    df["historical_zero_production_anomaly"] = anomaly_mask

    # Initial primary training eligibility.
    df["eligible_for_primary_training"] = (
        df["target_quality"].eq("valid_positive_yield")
        & ~df["historical_zero_production_anomaly"]
    )

    summary = (
        df.groupby("target_quality", dropna=False)
        .agg(
            records=("id", "size"),
            crops=("crop_code", "nunique"),
            states=("state_name", "nunique"),
        )
        .reset_index()
    )

    print("\nFINAL TARGET QUALITY SUMMARY")
    print("=" * 75)
    print(summary.to_string(index=False))

    print("\nHISTORICAL ANOMALY FLAGS")
    print("=" * 75)
    print(
        "Flagged records:",
        int(df["historical_zero_production_anomaly"].sum()),
    )

    print("\nPRIMARY TRAINING CANDIDATES")
    print("=" * 75)
    print(
        "Eligible records:",
        int(df["eligible_for_primary_training"].sum()),
    )

    df.to_csv(OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print(f"\nDataset saved: {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Summary saved: {SUMMARY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
