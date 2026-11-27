
"""Create leakage-safe chronological datasets for model development."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT / "data" / "processed"
    / "crop_yield_modeling_candidates.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "splits"
REPORT_DIR = ROOT / "reports"

TRAIN_END = 2017
VALIDATION_END = 2019

CATEGORICAL_FEATURES = [
    "state_code",
    "district_code",
    "crop_code",
    "crop_name",
    "crop_type",
    "season",
]

NUMERIC_FEATURES = [
    "start_year",
    "area",
]

TARGET = "yield"


def main():  # sourcery skip: for-append-to-extend
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    # Extract the first year from labels such as 1997-1998.
    df["start_year"] = pd.to_numeric(
        df["year"].str.extract(r"^(\d{4})")[0],
        errors="coerce",
    )

    if df["start_year"].isna().any():
        raise ValueError("Some year values could not be parsed.")

    if df["area"].le(0).any():
        raise ValueError("Non-positive area found.")

    if df[TARGET].isna().any() or df[TARGET].le(0).any():
        raise ValueError("Invalid target values found.")

    # Keep categorical identifiers categorical, not numeric.
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string").fillna("MISSING")

    # Stable identities for coverage analysis.
    df["crop_identity"] = (
        df["crop_code"] + "|" + df["crop_name"]
    )

    df["district_identity"] = (
        df["state_code"] + "|" + df["district_code"]
    )

    # Chronological split.
    train = df[df["start_year"] <= TRAIN_END].copy()

    validation = df[
        df["start_year"].between(2018, VALIDATION_END)
    ].copy()

    test = df[
        df["start_year"].between(2020, 2022)
    ].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("One or more splits are empty.")

    # Explicit model feature columns.
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    # Keep target, features and audit metadata.
    output_columns = (
        ["id"]
        + feature_columns
        + [TARGET, "statistical_outlier_flag", "review_status",
           "crop_identity", "district_identity"]
    )

    train = train[output_columns]
    validation = validation[output_columns]
    test = test[output_columns]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test.csv", index=False)

    # Coverage report.
    train_crops = set(train["crop_identity"])
    train_districts = set(train["district_identity"])

    coverage_rows = [
        {
            "split": name,
            "records": len(part),
            "crop_identities": part["crop_identity"].nunique(),
            "districts": part["district_identity"].nunique(),
            "unseen_crops_vs_train": (
                len(set(part["crop_identity"]) - train_crops)
                if name != "train"
                else 0
            ),
            "unseen_districts_vs_train": (
                len(set(part["district_identity"]) - train_districts)
                if name != "train"
                else 0
            ),
            "outlier_flags": int(part["statistical_outlier_flag"].sum()),
        }
        for name, part in [
            ("train", train),
            ("validation", validation),
            ("test", test),
        ]
    ]
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(
        REPORT_DIR / "temporal_split_coverage.csv",
        index=False,
    )

    print("\nTEMPORAL DATASETS CREATED")
    print("=" * 75)
    print(coverage.to_string(index=False))

    print("\nMODEL FEATURES")
    print("=" * 75)
    print(feature_columns)

    print("\nTARGET:", TARGET)
    print("Production excluded from predictors.")
    print("Calculated yield excluded from predictors.")
    print("\nOutput directory:", OUTPUT_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()
