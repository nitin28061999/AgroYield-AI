from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data/raw/crop-wise-area-production-yield.csv"
REPORT_DIR = ROOT / "reports"

OUTPUT_FILE = REPORT_DIR / "cross_crop_extreme_records.csv"
SUMMARY_FILE = REPORT_DIR / "cross_crop_extreme_summary.csv"

df = pd.read_csv(INPUT_FILE, low_memory=False)

# Normalize numeric columns
for column in ["area", "production", "yield"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Exclude annual aggregate records
df = df[df["season"] != "Total"].copy()

# Calculate yield independently
df["calculated_yield"] = df["production"] / df["area"]

# Screening conditions — not automatic data-error classifications
df["extreme_yield_flag"] = df["calculated_yield"] > 100
df["small_area_flag"] = df["area"] < 10
df["substantial_production_flag"] = df["production"] > 100

# Combined high-priority screening group
extreme = df[
    df["extreme_yield_flag"]
    & df["small_area_flag"]
    & df["substantial_production_flag"]
].copy()

extreme["crop_code_normalized"] = (
    pd.to_numeric(extreme["crop_code"], errors="coerce")
    .astype("Int64")
    .astype("string")
)

columns = [
    "id",
    "year",
    "state_name",
    "district_name",
    "crop_name",
    "crop_code",
    "season",
    "area",
    "production",
    "yield",
    "calculated_yield",
    "area_unit",
    "production_unit",
    "yield_unit",
]

extreme = extreme[columns].sort_values(
    "calculated_yield",
    ascending=False,
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
extreme.to_csv(OUTPUT_FILE, index=False)

# Crop-level summary
summary = (
    extreme.groupby(
        ["crop_name", "crop_code"],
        dropna=False,
    )
    .agg(
        flagged_records=("id", "count"),
        affected_states=("state_name", "nunique"),
        affected_districts=("district_name", "nunique"),
        affected_years=("year", "nunique"),
        median_yield=("calculated_yield", "median"),
        maximum_yield=("calculated_yield", "max"),
        median_area=("area", "median"),
        maximum_production=("production", "max"),
    )
    .reset_index()
    .sort_values("flagged_records", ascending=False)
)

summary.to_csv(SUMMARY_FILE, index=False)

print("=" * 100)
print("AGROYIELD AI — CROSS-CROP EXTREME RECORD SCREENING")
print("=" * 100)

print("\nSeasonal records examined:", len(df))
print("Extreme yield records (>100 t/ha):", df["extreme_yield_flag"].sum())
print("Records with area <10 ha:", df["small_area_flag"].sum())
print("Records with production >100 tonnes:",
      df["substantial_production_flag"].sum())

print("\nCombined high-priority records:", len(extreme))
print("Distinct crops:", extreme["crop_name"].nunique())
print("Distinct states:", extreme["state_name"].nunique())
print("Distinct districts:", extreme["district_name"].nunique())

print("\nTOP 30 EXTREME RECORDS")
print(extreme.head(30).to_string(index=False))

print("\nCROP-LEVEL SUMMARY")
print(summary.to_string(index=False))

print("\nSaved records:", OUTPUT_FILE)
print("Saved summary:", SUMMARY_FILE)
