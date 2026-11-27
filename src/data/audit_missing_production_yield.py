
"""Audit missing production records and their corresponding yield values."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "data" / "raw" / "crop-wise-area-production-yield.csv"

REPORT_PATH = ROOT / "reports" / "missing_production_yield_audit.csv"
SAMPLE_PATH = ROOT / "reports" / "missing_production_yield_samples.csv"


def audit_missing_production():
    df = pd.read_csv(RAW_PATH, low_memory=False)

    # Exclude overlapping aggregate season records.
    df = df[df["season"] != "Total"].copy()

    missing = df[df["production"].isna()].copy()

    print("\nMISSING PRODUCTION / YIELD AUDIT")
    print("=" * 75)
    print("Records excluding Total season:", len(df))
    print("Missing production records:", len(missing))
    print(
        "Missing production + zero yield:",
        (missing["yield"] == 0).sum(),
    )
    print(
        "Missing production + positive yield:",
        (missing["yield"] > 0).sum(),
    )
    print(
        "Missing production + missing yield:",
        missing["yield"].isna().sum(),
    )

    print("\nYIELD DISTRIBUTION FOR MISSING PRODUCTION")
    print("=" * 75)
    print(missing["yield"].describe().to_string())

    print("\nTOP CROPS WITH MISSING PRODUCTION")
    print("=" * 75)

    crop_summary = (
        missing.groupby(["crop_name", "crop_code"], dropna=False)
        .agg(
            missing_records=("id", "size"),
            zero_yield=("yield", lambda s: (s == 0).sum()),
            positive_yield=("yield", lambda s: (s > 0).sum()),
        )
        .sort_values("missing_records", ascending=False)
        .reset_index()
    )

    print(crop_summary.head(20).to_string(index=False))
    crop_summary.to_csv(REPORT_PATH, index=False)

    sample_columns = [
        "id",
        "year",
        "state_name",
        "district_name",
        "crop_name",
        "crop_code",
        "season",
        "area",
        "production",
        "yield",
    ]

    samples = missing[sample_columns].head(100)
    samples.to_csv(SAMPLE_PATH, index=False)

    print("\nSample records:")
    print(samples.head(20).to_string(index=False))

    print(f"\nSummary saved: {REPORT_PATH.relative_to(ROOT)}")
    print(f"Samples saved: {SAMPLE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    audit_missing_production()
