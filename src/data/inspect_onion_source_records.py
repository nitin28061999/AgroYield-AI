from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "raw" / "crop-wise-area-production-yield.csv"
OUTPUT_FILE = ROOT / "reports" / "onion_source_record_inspection.csv"

df = pd.read_csv(INPUT_FILE, low_memory=False)

# Normalize crop code
crop_code = pd.to_numeric(df["crop_code"], errors="coerce")

onion = df.loc[crop_code.eq(708)].copy()

# Keep Andhra Pradesh records from 2015-16 to 2019-20
onion = onion[
    onion["state_name"].eq("Andhra Pradesh")
    & onion["year"].isin(
        ["2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020"]
    )
].copy()

# Calculate yield independently
onion["calculated_yield"] = (
    onion["production"] / onion["area"]
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
    "area_unit",
    "production",
    "production_unit",
    "yield",
    "yield_unit",
    "calculated_yield",
]

onion = onion[columns].sort_values(
    ["district_name", "year", "season", "id"]
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
onion.to_csv(OUTPUT_FILE, index=False)

print("=" * 100)
print("ONION — ORIGINAL SOURCE RECORD INSPECTION")
print("=" * 100)

print("Total records:", len(onion))
print("\nSeason distribution:")
print(onion["season"].value_counts(dropna=False).to_string())

print("\nRecords with yield above 50 tonnes/hectare:")
print(
    onion.loc[
        onion["calculated_yield"].gt(50)
    ].to_string(index=False)
)

print("\nSaved:", OUTPUT_FILE)
