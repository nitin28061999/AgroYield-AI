
"""Profile and investigate data-quality issues in AgroYield AI."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "crop-wise-area-production-yield.csv"
)


def _print_value_counts(series, *, dropna=True, sort_index=False, limit=None):
    counts = series.value_counts(dropna=dropna)

    if sort_index:
        counts = counts.sort_index()

    if limit is not None:
        counts = counts.head(limit)

    print(counts.to_string())


def profile_dataset():  # sourcery skip: extract-duplicate-method
    """Investigate missing values, zero values and yield consistency."""

    print("=" * 70)
    print("AGROYIELD AI - DATA QUALITY INVESTIGATION")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Ensure numeric columns have numeric values.
    numeric_columns = ["area", "production", "yield"]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    # 1. Missing production investigation.
    print("\n1. MISSING PRODUCTION")

    missing_production = df[df["production"].isna()]

    print(f"Missing production records: {len(missing_production):,}")

    print("\nMissing production by season:")
    _print_value_counts(missing_production["season"], dropna=False)

    print("\nMissing production by year:")
    _print_value_counts(missing_production["year"], sort_index=True)

    print("\nMissing production by crop (top 15):")
    _print_value_counts(missing_production["crop_name"], limit=15)

    # 2. Zero-value consistency.
    print("\n2. ZERO-VALUE INVESTIGATION")

    zero_production = df["production"].eq(0)
    zero_yield = df["yield"].eq(0)

    print(f"Zero production: {zero_production.sum():,}")
    print(f"Zero yield: {zero_yield.sum():,}")

    print(
        "Zero production but positive yield:",
        (zero_production & df["yield"].gt(0)).sum(),
    )

    print(
        "Positive production but zero yield:",
        (df["production"].gt(0) & zero_yield).sum(),
    )

    print(
        "Missing production but zero yield:",
        (df["production"].isna() & zero_yield).sum(),
    )

    # 3. Verify the supplied yield calculation.
    print("\n3. YIELD CONSISTENCY")

    valid = (
        df["area"].gt(0)
        & df["production"].notna()
        & df["yield"].notna()
    )

    checked = df.loc[valid].copy()

    checked["calculated_yield"] = (
        checked["production"] / checked["area"]
    )

    checked["calculated_yield_rounded"] = (
        checked["calculated_yield"].round(3)
    )

    checked["yield_mismatch"] = (
        checked["calculated_yield_rounded"]
        != checked["yield"].round(3)
    )

    print(f"Records checked: {len(checked):,}")
    print(
        "Yield mismatches after rounding to 3 decimals:",
        checked["yield_mismatch"].sum(),
    )

    if checked["yield_mismatch"].any():
        print("\nSample yield mismatches:")
        print(
            checked.loc[
                checked["yield_mismatch"],
                [
                    "state_name",
                    "district_name",
                    "year",
                    "crop_name",
                    "season",
                    "area",
                    "production",
                    "yield",
                    "calculated_yield",
                ],
            ].head(10).to_string(index=False)
        )

    # 4. Investigate Total season overlap.
    print("\n4. TOTAL-SEASON OVERLAP")

    group_columns = [
        "year",
        "state_name",
        "district_name",
        "crop_name",
    ]

    season_counts = (
        df.groupby(group_columns, dropna=False)["season"]
        .transform("nunique")
    )

    total_rows = df["season"].eq("Total")

    overlapping_total = total_rows & season_counts.gt(1)

    print(f"Total-season records: {total_rows.sum():,}")
    print(
        "Total records sharing a crop/location/year "
        "with other seasons:",
        overlapping_total.sum(),
    )

    # 5. Missing crop codes.
    print("\n5. MISSING CROP CODES")

    missing_crop_codes = df[df["crop_code"].isna()]

    print(f"Missing crop codes: {len(missing_crop_codes):,}")

    if not missing_crop_codes.empty:
        print("\nAffected crop names:")
        _print_value_counts(missing_crop_codes["crop_name"], limit=20)

    # 6. Unit checks.
    print("\n6. UNIT CONSISTENCY")

    for column in [
        "area_unit",
        "production_unit",
        "yield_unit",
    ]:
        print(f"\n{column}:")
        _print_value_counts(df[column], dropna=False)

    print("\n7. FINAL SUMMARY")
    print(f"Total records: {len(df):,}")
    print(f"Unique crops: {df['crop_name'].nunique():,}")
    print(f"Unique districts: {df['district_name'].nunique():,}")
    print(f"Unique years: {df['year'].nunique():,}")

    print("\nDATA QUALITY INVESTIGATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    profile_dataset()
