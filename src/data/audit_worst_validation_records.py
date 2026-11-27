
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data" / "raw" / "crop-wise-area-production-yield.csv"
WORST = ROOT / "reports" / "catboost_v1_crop_geographic_worst_records.csv"
OUTPUT = ROOT / "reports" / "catboost_v1_worst_records_raw_audit.csv"

worst = pd.read_csv(WORST, dtype={"id": "string"})

worst = worst.sort_values(
    "absolute_error",
    ascending=False,
).head(20)

raw = pd.read_csv(
    RAW,
    dtype={"id": "string"},
    low_memory=False,
)

raw_columns = [
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

audit = worst.merge(
    raw[raw_columns],
    on="id",
    how="left",
    validate="one_to_one",
    suffixes=("_prediction", "_raw"),
    indicator=True,
)

assert (audit["_merge"] == "both").all(), (
    "Some validation IDs were not found in raw data"
)

audit["calculated_yield"] = (
    audit["production"] / audit["area"]
)

audit["yield_difference"] = (
    audit["yield"] - audit["calculated_yield"]
)

audit["prediction_actual_difference"] = (
    audit["actual_yield"] - audit["yield"]
)

audit["identity_match"] = (
    (audit["crop_name_prediction"] == audit["crop_name_raw"])
    & (audit["state_code_prediction"] == audit["state_code_raw"])
    & (audit["district_code_prediction"] == audit["district_code_raw"])
)

audit.to_csv(OUTPUT, index=False)

display_columns = [
    "id",
    "crop_name_raw",
    "state_name",
    "district_name",
    "year_raw",
    "season_raw",
    "area",
    "area_unit",
    "production",
    "production_unit",
    "yield",
    "yield_unit",
    "calculated_yield",
    "predicted_yield",
    "absolute_error",
    "statistical_outlier_flag",
    "identity_match",
]

print("\nWORST VALIDATION RECORDS — RAW DATA AUDIT")
print("=" * 120)

print(
    audit[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nAUDIT CHECKS")
print("=" * 80)
print("Records matched:", len(audit))
print("Identity matches:", audit["identity_match"].sum())
print(
    "Yield arithmetic matches:",
    audit["yield_difference"].abs().le(0.001).sum(),
    "of",
    len(audit),
)
print(
    "Prediction actual matches:",
    audit["prediction_actual_difference"].abs().le(0.000001).sum(),
    "of",
    len(audit),
)

print("\nSaved:", OUTPUT)
print("Raw source data was not modified.")
