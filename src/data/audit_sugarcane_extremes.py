from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data/raw/crop-wise-area-production-yield.csv"
OUTPUT = ROOT / "reports/sugarcane_extreme_history.csv"

df = pd.read_csv(INPUT, low_memory=False)

df["crop_code_num"] = pd.to_numeric(
    df["crop_code"], errors="coerce"
)

df["area"] = pd.to_numeric(df["area"], errors="coerce")
df["production"] = pd.to_numeric(
    df["production"], errors="coerce"
)
df["yield"] = pd.to_numeric(df["yield"], errors="coerce")

sugarcane = df[df["crop_code_num"] == 401].copy()

sugarcane["calculated_yield"] = (
    sugarcane["production"] / sugarcane["area"]
)

# Select records matching the high-priority area/production/yield criteria
extreme = sugarcane[
    (sugarcane["calculated_yield"] > 100)
    & (sugarcane["area"] < 10)
    & (sugarcane["production"] > 100)
].copy()

# Retrieve all seasonal and Total records for the affected districts and years
keys = extreme[
    ["state_name", "district_name", "year"]
].drop_duplicates()

history = sugarcane.merge(
    keys,
    on=["state_name", "district_name", "year"],
    how="inner",
)

columns = [
    "id",
    "year",
    "state_name",
    "district_name",
    "season",
    "area",
    "production",
    "yield",
    "calculated_yield",
    "area_unit",
    "production_unit",
    "yield_unit",
]

history = history[columns].sort_values(
    ["state_name", "district_name", "year", "season"]
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
history.to_csv(OUTPUT, index=False)

print("=" * 100)
print("SUGARCANE — EXTREME RECORD HISTORY")
print("=" * 100)

print("Extreme records:", len(extreme))
print("Affected district-year groups:", len(keys))
print("Full history records:", len(history))

print("\nExtreme records:")
print(
    extreme[columns].sort_values(
        "calculated_yield", ascending=False
    ).to_string(index=False)
)

print("\nSeason distribution in retrieved history:")
print(history["season"].value_counts(dropna=False).to_string())

print("\nSaved:", OUTPUT)
