
"""Inspect complete historical production patterns for suspicious crops."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "data" / "raw" / "crop-wise-area-production-yield.csv"
FLAG_PATH = ROOT / "reports" / "zero_production_transition_audit.csv"

OUTPUT_PATH = ROOT / "reports" / "zero_production_crop_history.csv"


def main():
    df = pd.read_csv(RAW_PATH, low_memory=False)
    df = df[df["season"] != "Total"].copy()
    df = df[df["production"].notna()].copy()

    flagged = pd.read_csv(FLAG_PATH)

    # Preserve both crop name and crop code.
    crop_keys = flagged[["crop_name", "crop_code"]].drop_duplicates()

    selected = df.merge(
        crop_keys,
        on=["crop_name", "crop_code"],
        how="inner",
    )

    history = (
        selected.assign(
            zero_production=selected["production"].eq(0),
            positive_production=selected["production"].gt(0),
        )
        .groupby(
            ["crop_name", "crop_code", "year"],
            dropna=False,
        )
        .agg(
            total_records=("id", "size"),
            zero_records=("zero_production", "sum"),
            positive_records=("positive_production", "sum"),
            districts=("district_name", "nunique"),
            states=("state_name", "nunique"),
            total_area=("area", "sum"),
        )
        .reset_index()
    )

    history["zero_rate_pct"] = (
        history["zero_records"] / history["total_records"] * 100
    ).round(2)

    years = sorted(
        df["year"].dropna().unique(),
        key=lambda value: int(str(value)[:4]),
    )

    pd.MultiIndex.from_product(
        [
            crop_keys["crop_name"].unique(),
            years,
        ],
        names=["crop_name", "year"],
    )

    # Keep crop_code in the final report; crop names alone are not unique.
    history = history.sort_values(
        ["crop_name", "crop_code", "year"]
    )

    print("\nCOMPLETE ZERO-PRODUCTION CROP HISTORY")
    print("=" * 90)
    print("Distinct crop identities:", len(crop_keys))
    print("Historical crop-year rows:", len(history))

    print("\n")
    print(history.to_string(index=False))

    history.to_csv(OUTPUT_PATH, index=False)

    print(f"\nReport saved: {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
