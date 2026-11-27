
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "crop_yield_modeling_candidates.csv"
TRAIN_PATH = ROOT / "data" / "processed" / "splits" / "train.csv"
VAL_PATH = ROOT / "data" / "processed" / "splits" / "validation.csv"
REPORT_DIR = ROOT / "reports"

data = pd.read_csv(
    DATA_PATH,
    dtype={"crop_code": "string", "id": "string"},
)

train_ids = pd.read_csv(TRAIN_PATH, usecols=["id"], dtype={"id": "string"})
val_ids = pd.read_csv(VAL_PATH, usecols=["id"], dtype={"id": "string"})

data["split"] = "other"
data.loc[data["id"].isin(train_ids["id"]), "split"] = "train"
data.loc[data["id"].isin(val_ids["id"]), "split"] = "validation"

# Select Coconut using both crop name and crop code.
# Normalize crop codes before filtering.
data["crop_code_normalized"] = (
    pd.to_numeric(data["crop_code"], errors="coerce")
    .astype("Int64")
)

# Select Coconut using both crop name and crop code.
coconut = data[
    data["crop_name"].str.strip().str.lower().eq("coconut")
    & data["crop_code_normalized"].eq(1006)
].copy()

print("\nCOCONUT DATASET OVERVIEW")
print("=" * 75)
print("Records:", len(coconut))
print(coconut["split"].value_counts().to_string())

# Distribution across training and validation.
summary = (
    coconut[coconut["split"].isin(["train", "validation"])]
    .groupby("split")["yield"]
    .describe(percentiles=[0.50, 0.90, 0.95, 0.99])
)

print("\nYIELD DISTRIBUTION BY SPLIT")
print("=" * 75)
print(summary.round(3).to_string())

# Outlier concentration.
print("\nOUTLIER STATUS BY SPLIT")
print("=" * 75)
print(
    pd.crosstab(
        coconut["split"],
        coconut["statistical_outlier_flag"],
        margins=True,
    ).to_string()
)

# State-level validation distribution.
validation = coconut[coconut["split"] == "validation"].copy()

state_summary = (
    validation.groupby("state_name")
    .agg(
        records=("yield", "size"),
        mean_yield=("yield", "mean"),
        median_yield=("yield", "median"),
        max_yield=("yield", "max"),
        mean_area=("area", "mean"),
        outlier_records=("statistical_outlier_flag", "sum"),
    )
    .sort_values("mean_yield", ascending=False)
)

print("\nVALIDATION YIELD BY STATE")
print("=" * 75)
print(state_summary.round(3).to_string())

# Highest-yield validation observations.
columns = [
    "id",
    "year",
    "state_name",
    "district_name",
    "area",
    "production",
    "yield",
    "statistical_outlier_flag",
]

print("\nTOP 20 COCONUT VALIDATION YIELDS")
print("=" * 75)
print(
    validation.nlargest(20, "yield")[columns]
    .round(3)
    .to_string(index=False)
)

# Save detailed reports.
summary.to_csv(REPORT_DIR / "coconut_yield_split_summary.csv")
state_summary.to_csv(REPORT_DIR / "coconut_validation_by_state.csv")

coconut.to_csv(
    REPORT_DIR / "coconut_full_record_audit.csv",
    index=False,
)

print("\nSaved:")
print("- reports/coconut_yield_split_summary.csv")
print("- reports/coconut_validation_by_state.csv")
print("- reports/coconut_full_record_audit.csv")
