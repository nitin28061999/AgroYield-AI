
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = ROOT / "data" / "processed" / "splits_v1"
REPORT_DIR = ROOT / "reports"

PREDICTIONS_PATH = REPORT_DIR / "catboost_v1_validation_predictions.csv"

CROP_KEYS = ["crop_code", "crop_name"]


def normalize_code(series):
    return (
        pd.to_numeric(series, errors="coerce")
        .astype("Int64")
        .astype("string")
    )


def main():
    predictions = pd.read_csv(
        PREDICTIONS_PATH, dtype={"id": "string"}, low_memory=False
    )
    validation = pd.read_csv(
        SPLIT_DIR / "validation.csv",
        dtype={"id": "string"},
        low_memory=False,
    )
    train = pd.read_csv(
        SPLIT_DIR / "train.csv",
        low_memory=False,
    )

    for frame in [predictions, validation, train]:
        frame["crop_code"] = normalize_code(frame["crop_code"])

    # Attach location, area and season to validation predictions.
    details = predictions.merge(
        validation[
            [
                "id", "state_code", "district_code", "area",
                "season", "crop_type", "crop_identity",
                "district_identity",
            ]
        ],
        on="id",
        how="left",
        validate="one_to_one",
    )

    # Training history for each crop identity.
    train_history = (
        train.groupby(CROP_KEYS, dropna=False)
        .agg(
            train_records=("yield", "size"),
            train_yield_median=("yield", "median"),
            train_yield_p95=("yield", lambda x: x.quantile(0.95)),
            train_yield_max=("yield", "max"),
            train_area_median=("area", "median"),
            train_area_p95=("area", lambda x: x.quantile(0.95)),
            train_area_max=("area", "max"),
        )
        .reset_index()
    )

    details = details.merge(
        train_history,
        on=CROP_KEYS,
        how="left",
        validate="many_to_one",
    )

    details["prediction_to_train_median"] = (
        details["predicted_yield"]
        / details["train_yield_median"].replace(0, pd.NA)
    )

    details["area_to_train_median"] = (
        details["area"]
        / details["train_area_median"].replace(0, pd.NA)
    )

    # Inspect largest absolute errors.
    worst = details.sort_values(
        "absolute_error", ascending=False
    ).head(50)

    columns = [
        "id", "crop_code", "crop_name", "state_code",
        "district_code", "year", "season", "area",
        "actual_yield", "predicted_yield", "absolute_error",
        "statistical_outlier_flag", "train_records",
        "train_yield_median", "train_yield_p95",
        "train_yield_max", "train_area_median",
        "train_area_p95", "prediction_to_train_median",
        "area_to_train_median",
    ]

    worst[columns].to_csv(
        REPORT_DIR / "catboost_v1_error_record_investigation.csv",
        index=False,
    )

    print("\nTOP 30 VALIDATION ERRORS WITH TRAINING CONTEXT")
    print("=" * 140)
    print(worst[columns].head(30).to_string(index=False))

    # Training support for crops with high validation MAE.
    crop_validation = (
        details.groupby(CROP_KEYS, dropna=False)
        .agg(
            validation_records=("id", "size"),
            validation_MAE=("absolute_error", "mean"),
            validation_max_error=("absolute_error", "max"),
        )
        .reset_index()
    )

    crop_comparison = crop_validation.merge(
        train_history,
        on=CROP_KEYS,
        how="left",
        validate="one_to_one",
    )

    crop_comparison = crop_comparison.sort_values(
        "validation_MAE", ascending=False
    )

    crop_comparison.to_csv(
        REPORT_DIR / "catboost_v1_crop_train_validation_comparison.csv",
        index=False,
    )

    print("\nHIGH-ERROR CROPS — TRAINING SUPPORT")
    print("=" * 110)
    print(
        crop_comparison.head(20)
        .round(3)
        .to_string(index=False)
    )

    # Check crop-code/name collisions.
    identity_counts = (
        train.groupby("crop_code", dropna=False)["crop_name"]
        .nunique()
        .reset_index(name="distinct_crop_names")
    )

    collisions = identity_counts[
        identity_counts["distinct_crop_names"] > 1
    ]

    print("\nCROP CODES MAPPED TO MULTIPLE NAMES IN TRAINING DATA")
    print("=" * 75)

    if collisions.empty:
        print("No crop-code/name collisions found.")
    else:
        print(collisions.to_string(index=False))

        collision_details = train[
            train["crop_code"].isin(collisions["crop_code"])
        ].groupby(
            ["crop_code", "crop_name"], dropna=False
        ).agg(
            records=("yield", "size"),
            median_yield=("yield", "median"),
        ).reset_index()

        print("\nCollision details:")
        print(collision_details.to_string(index=False))

    print("\nReports saved:")
    print("  catboost_v1_error_record_investigation.csv")
    print("  catboost_v1_crop_train_validation_comparison.csv")


if __name__ == "__main__":
    main()
