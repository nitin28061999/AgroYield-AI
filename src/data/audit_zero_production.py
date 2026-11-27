
"""Investigate records with zero reported production."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "data" / "raw" / "crop-wise-area-production-yield.csv"
OUTPUT_PATH = ROOT / "reports" / "zero_production_audit.csv"


def audit_zero_production():
    df = pd.read_csv(RAW_PATH, low_memory=False)

    df = df[df["season"] != "Total"].copy()

    zero = df[df["production"] == 0].copy()

    print("\nZERO PRODUCTION AUDIT")
    print("=" * 75)
    print("Zero-production records:", len(zero))
    print("Distinct crops:", zero["crop_code"].nunique())
    print("Distinct states:", zero["state_name"].nunique())
    print("Yield values:")
    print(zero["yield"].value_counts(dropna=False).to_string())

    print("\nTOP CROPS")
    print("=" * 75)

    crop_summary = (
        zero.groupby(["crop_name", "crop_code"], dropna=False)
        .agg(
            records=("id", "size"),
            districts=("district_name", "nunique"),
            states=("state_name", "nunique"),
            years=("year", "nunique"),
        )
        .sort_values("records", ascending=False)
        .reset_index()
    )

    print(crop_summary.head(25).to_string(index=False))

    print("\nTOP STATES")
    print("=" * 75)

    state_summary = (
        zero.groupby("state_name")
        .size()
        .sort_values(ascending=False)
    )

    print(state_summary.head(15).to_string())

    zero.to_csv(OUTPUT_PATH, index=False)

    print(f"\nFull audit saved: {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    audit_zero_production()
