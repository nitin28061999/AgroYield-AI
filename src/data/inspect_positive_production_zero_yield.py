
"""Inspect records with positive production but zero reported yield."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "data" / "raw" / "crop-wise-area-production-yield.csv"
OUTPUT_PATH = ROOT / "reports" / "positive_production_zero_yield.csv"


def inspect_records():
    df = pd.read_csv(RAW_PATH, low_memory=False)

    df = df[df["season"] != "Total"].copy()

    flagged = df[
        (df["production"] > 0) &
        (df["yield"] == 0)
    ].copy()

    flagged["calculated_yield"] = (
        flagged["production"] / flagged["area"]
    )

    columns = [
        "id", "year", "state_name", "district_name",
        "crop_name", "crop_code", "season",
        "area", "production", "yield", "calculated_yield",
    ]

    flagged = flagged[columns].sort_values(
        ["crop_name", "year", "state_name", "district_name"]
    )

    print("\nPOSITIVE PRODUCTION WITH ZERO YIELD")
    print("=" * 85)
    print("Flagged records:", len(flagged))
    print("Distinct crops:", flagged["crop_name"].nunique())
    print("Distinct states:", flagged["state_name"].nunique())

    print("\nBY CROP")
    print("=" * 85)

    print(
        flagged.groupby(["crop_name", "crop_code"])
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\nFLAGGED RECORDS")
    print("=" * 85)
    print(flagged.to_string(index=False))

    flagged.to_csv(OUTPUT_PATH, index=False)
    print(f"\nReport saved: {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    inspect_records()
