from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "data" / "processed" / "crop_yield_training_v1.csv"
OUTLIER_PATH = ROOT / "data" / "processed" / "crop_yield_modeling_candidates.csv"

SPLIT_DIR = ROOT / "data" / "processed" / "splits_v1"
REPORT_DIR = ROOT / "reports"

EXPECTED = {
    "train": 303_570,
    "validation": 36_737,
    "test": 54_772,
}

FEATURES = [
    "start_year",
    "area",
    "state_code",
    "district_code",
    "crop_code",
    "crop_name",
    "crop_type",
    "season",
]

AUDIT_COLUMNS = [
    "id",
    "crop_identity",
    "district_identity",
    "statistical_outlier_flag",
]


def main():
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH, dtype={"id": "string"}, low_memory=False)

    outliers = pd.read_csv(
        OUTLIER_PATH,
        usecols=["id", "statistical_outlier_flag"],
        dtype={"id": "string"},
        low_memory=False,
    )

    if df["id"].duplicated().any():
        raise ValueError("Duplicate IDs found in v1.0 training dataset.")

    if outliers["id"].duplicated().any():
        raise ValueError("Duplicate IDs found in outlier audit dataset.")

    df = df.merge(
        outliers,
        on="id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not df["_merge"].eq("both").all():
        missing = int(df["_merge"].ne("both").sum())
        raise ValueError(f"{missing} v1.0 records lack an outlier audit match.")

    df = df.drop(columns="_merge")

    # Normalize model features.
    df["start_year"] = pd.to_numeric(df["start_year"], errors="coerce")
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    df["yield"] = pd.to_numeric(df["yield"], errors="coerce")

    for column in ["state_code", "district_code", "crop_code",
                   "crop_name", "crop_type", "season"]:
        df[column] = df[column].astype("string").fillna("MISSING")

    df["crop_code"] = (
        df["crop_code_normalized"].astype("string").fillna("MISSING")
    )

    df["start_year"] = df["start_year"].astype("Int64")

    df["crop_identity"] = (
        df["crop_code"] + "|" + df["crop_name"]
    )
    df["district_identity"] = (
        df["state_code"] + "|" + df["district_code"]
    )

    if df[FEATURES].isna().any().any():
        raise ValueError("Missing values found in model features.")

    if df["yield"].isna().any() or (df["yield"] <= 0).any():
        raise ValueError("Invalid target values found.")

    # Chronological partitioning.
    conditions = [
        df["start_year"].between(1997, 2017),
        df["start_year"].between(2018, 2019),
        df["start_year"].between(2020, 2022),
    ]

    df["split"] = pd.NA
    df.loc[conditions[0], "split"] = "train"
    df.loc[conditions[1], "split"] = "validation"
    df.loc[conditions[2], "split"] = "test"

    if df["split"].isna().any():
        unexpected = sorted(
            df.loc[df["split"].isna(), "start_year"].dropna().unique().tolist()
        )
        raise ValueError(f"Unassigned years found: {unexpected}")

    print("\nAGROYIELD AI — TEMPORAL SPLITS v1.0")
    print("=" * 65)

    actual_counts = df["split"].value_counts().to_dict()

    for split_name, expected_count in EXPECTED.items():
        actual_count = actual_counts.get(split_name, 0)
        status = "PASS" if actual_count == expected_count else "MISMATCH"
        print(
            f"{split_name:15s} {actual_count:>10,} "
            f"(expected {expected_count:,})  {status}"
        )

    if actual_counts != EXPECTED:
        raise ValueError(
            f"Split count mismatch. Actual: {actual_counts}; "
            f"Expected: {EXPECTED}"
        )

    output_columns = FEATURES + ["yield"] + AUDIT_COLUMNS

    for split_name in ["train", "validation", "test"]:
        split_df = df.loc[
            df["split"].eq(split_name), output_columns
        ].copy()

        output_path = SPLIT_DIR / f"{split_name}.csv"
        split_df.to_csv(output_path, index=False)

        print(f"Saved {split_name}: {len(split_df):,} rows")

    coverage = (
        df.groupby(["split", "start_year"], observed=True)
        .agg(
            records=("id", "count"),
            crop_identities=("crop_identity", "nunique"),
            districts=("district_identity", "nunique"),
            outlier_records=("statistical_outlier_flag", "sum"),
        )
        .reset_index()
    )

    coverage.to_csv(
        REPORT_DIR / "temporal_split_coverage_v1.csv",
        index=False,
    )

    print("\nSplit year coverage:")
    print(coverage.to_string(index=False))

    print("\nAll v1.0 temporal split checks passed.")


if __name__ == "__main__":
    main()
