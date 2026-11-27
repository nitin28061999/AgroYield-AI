
"""Validate the raw crop production dataset."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "crop-wise-area-production-yield.csv"
)

EXPECTED_COLUMNS = [
    "id",
    "year",
    "state_name",
    "state_code",
    "district_name",
    "district_code",
    "crop_name",
    "crop_code",
    "crop_type",
    "season",
    "area",
    "area_unit",
    "production",
    "production_unit",
    "yield",
    "yield_unit",
]


def validate_dataset():
    """Load the dataset and perform initial quality checks."""

    print("=" * 65)
    print("AGROYIELD AI - DATA VALIDATION")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)

    print(f"\nDataset path: {DATA_PATH}")
    print(f"Dataset shape: {df.shape}")

    print("\n--- COLUMN VALIDATION ---")

    print("Actual columns:")
    print(df.columns.tolist())

    missing_columns = [
        col for col in EXPECTED_COLUMNS
        if col not in df.columns
    ]

    extra_columns = [
        col for col in df.columns
        if col not in EXPECTED_COLUMNS
    ]

    print(f"Missing expected columns: {missing_columns}")
    print(f"Extra columns: {extra_columns}")

    print("\n--- DATA TYPES ---")
    print(df.dtypes)

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())

    print("\n--- DUPLICATE CHECK ---")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    print("\n--- NUMERIC VALIDATION ---")

    numeric_columns = ["area", "production", "yield"]

    for col in numeric_columns:
        if col in df.columns:
            values = pd.to_numeric(
                df[col], errors="coerce"
            )

            print(f"\n{col}:")
            print(f"  Invalid numeric values: {values.isna().sum() - df[col].isna().sum()}")
            print(f"  Negative values: {(values < 0).sum()}")
            print(f"  Zero values: {(values == 0).sum()}")
            print(f"  Minimum: {values.min()}")
            print(f"  Maximum: {values.max()}")

    print("\n--- CATEGORICAL SUMMARY ---")

    for col in [
        "state_name",
        "district_name",
        "crop_name",
        "crop_type",
        "season",
        "year",
    ]:
        if col in df.columns:
            print(
                f"{col}: {df[col].nunique()} unique values"
            )

    if "season" in df.columns:
        print("\nSeason distribution:")
        print(df["season"].value_counts(dropna=False))

    if "yield_unit" in df.columns:
        print("\nYield units:")
        print(df["yield_unit"].value_counts(dropna=False))

    print("\n--- DATA PREVIEW ---")
    print(df.head(5).to_string(index=False))

    print("\n--- VALIDATION COMPLETED ---")
    print("=" * 65)


if __name__ == "__main__":
    validate_dataset()
