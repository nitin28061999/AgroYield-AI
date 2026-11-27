
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = ROOT / "data" / "processed" / "splits_v1"
REPORT_DIR = ROOT / "reports"

COLLISION_CODES = ["1088", "1101", "188", "288", "507", "619", "688"]


def normalize_code(series):
    return (
        pd.to_numeric(series, errors="coerce")
        .astype("Int64")
        .astype("string")
    )


def main():
    train = pd.read_csv(SPLIT_DIR / "train.csv", low_memory=False)
    validation = pd.read_csv(
        SPLIT_DIR / "validation.csv",
        dtype={"id": "string"},
        low_memory=False,
    )
    predictions = pd.read_csv(
        REPORT_DIR / "catboost_v1_validation_predictions.csv",
        dtype={"id": "string"},
        low_memory=False,
    )

    train["id"] = train["id"].astype("string")
    validation["id"] = validation["id"].astype("string")
    predictions["id"] = predictions["id"].astype("string")

    for frame in [train, validation]:
        frame["crop_code"] = normalize_code(frame["crop_code"])

    validation = validation.merge(
        predictions[["id", "predicted_yield", "absolute_error"]],
        on="id",
        how="left",
        validate="one_to_one",
    )

    # Training distribution for each code/name identity.
    train_summary = (
        train[train["crop_code"].isin(COLLISION_CODES)]
        .groupby(["crop_code", "crop_name"], dropna=False)
        .agg(
            train_records=("yield", "size"),
            first_year=("start_year", "min"),
            last_year=("start_year", "max"),
            states=("state_code", "nunique"),
            districts=("district_code", "nunique"),
            median_yield=("yield", "median"),
            p95_yield=("yield", lambda x: x.quantile(0.95)),
            max_yield=("yield", "max"),
            median_area=("area", "median"),
        )
        .reset_index()
    )

    # Validation performance for each identity.
    validation_summary = (
        validation[validation["crop_code"].isin(COLLISION_CODES)]
        .groupby(["crop_code", "crop_name"], dropna=False)
        .agg(
            validation_records=("yield", "size"),
            validation_MAE=("absolute_error", "mean"),
            validation_max_error=("absolute_error", "max"),
            validation_median_yield=("yield", "median"),
            validation_max_yield=("yield", "max"),
        )
        .reset_index()
    )

    comparison = train_summary.merge(
        validation_summary,
        on=["crop_code", "crop_name"],
        how="outer",
    ).sort_values(["crop_code", "crop_name"])

    comparison.to_csv(
        REPORT_DIR / "crop_identity_collision_audit.csv",
        index=False,
    )

    # Inspect when different names under one code appeared.
    year_distribution = (
        train[train["crop_code"].isin(COLLISION_CODES)]
        .groupby(
            ["crop_code", "crop_name", "start_year"],
            dropna=False,
        )
        .size()
        .rename("records")
        .reset_index()
    )

    year_distribution.to_csv(
        REPORT_DIR / "crop_identity_collision_years.csv",
        index=False,
    )

    print("\nCROP CODE / NAME IDENTITY AUDIT")
    print("=" * 110)
    print(comparison.round(3).to_string(index=False))

    print("\nYEAR DISTRIBUTION BY CROP IDENTITY")
    print("=" * 110)

    for code in COLLISION_CODES:
        subset = year_distribution[
            year_distribution["crop_code"].eq(code)
        ]

        if not subset.empty:
            print(f"\nCrop code: {code}")
            print(
                subset.groupby("crop_name")
                .agg(
                    first_year=("start_year", "min"),
                    last_year=("start_year", "max"),
                    years_present=("start_year", "nunique"),
                    total_records=("records", "sum"),
                )
                .to_string()
            )

    print("\nReports saved:")
    print("  crop_identity_collision_audit.csv")
    print("  crop_identity_collision_years.csv")


if __name__ == "__main__":
    main()
