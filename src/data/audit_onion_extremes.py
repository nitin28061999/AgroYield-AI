from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "crop_yield_modeling_candidates.csv"
OUTPUT = ROOT / "reports" / "onion_extreme_history.csv"

df = pd.read_csv(INPUT, low_memory=False)

# Normalize crop codes safely.
df["crop_code"] = (
    pd.to_numeric(df["crop_code"], errors="coerce")
    .astype("Int64")
    .astype("string")
)

onion = df[df["crop_code"] == "708"].copy()

# Extract starting year from labels such as 2018-2019.
onion["start_year"] = (
    onion["year"].astype(str).str.extract(r"(\d{4})")[0]
)
onion["start_year"] = pd.to_numeric(
    onion["start_year"], errors="coerce"
)

# Focus on Andhra Pradesh and nearby years around the extreme records.
focus = onion[
    (onion["state_name"] == "Andhra Pradesh")
    & onion["start_year"].between(2015, 2022)
].copy()

columns = [
    "year",
    "state_name",
    "district_name",
    "area",
    "production",
    "yield",
    "statistical_outlier_flag",
]

focus = focus.sort_values(
    ["district_name", "start_year"]
)

print("\nONION — ANDHRA PRADESH DISTRICT HISTORY")
print("=" * 100)
print(focus[columns].to_string(index=False))

print("\nTOP 25 ONION YIELDS — ALL STATES")
print("=" * 100)

top = onion.nlargest(25, "yield")
print(top[columns].to_string(index=False))

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
focus.to_csv(OUTPUT, index=False)

print(f"\nSaved: {OUTPUT}")
