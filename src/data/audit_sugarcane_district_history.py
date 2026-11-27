
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data/raw/crop-wise-area-production-yield.csv"
OUTPUT = ROOT / "reports/sugarcane_affected_district_history.csv"
SUMMARY = ROOT / "reports/sugarcane_affected_district_summary.csv"

df = pd.read_csv(INPUT, low_memory=False)

df["crop_code_num"] = pd.to_numeric(
    df["crop_code"], errors="coerce"
)

for col in ["area", "production", "yield"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

sugarcane = df[
    (df["crop_code_num"] == 401)
    & (df["season"] != "Total")
].copy()

sugarcane["calculated_yield"] = (
    sugarcane["production"] / sugarcane["area"]
)

# Identify districts from the previous extreme-record audit
extreme = sugarcane[
    (sugarcane["calculated_yield"] > 100)
    & (sugarcane["area"] < 10)
    & (sugarcane["production"] > 100)
]

affected_districts = extreme[
    ["state_name", "district_name"]
].drop_duplicates()

# Retrieve every available year for these districts
history = sugarcane.merge(
    affected_districts,
    on=["state_name", "district_name"],
    how="inner",
)

history = history.sort_values(
    ["state_name", "district_name", "year"]
)

history["extreme_flag"] = (
    (history["calculated_yield"] > 100)
    & (history["area"] < 10)
    & (history["production"] > 100)
)

history.to_csv(OUTPUT, index=False)

summary = (
    history.groupby(
        ["state_name", "district_name"],
        dropna=False,
    )
    .agg(
        years_reported=("year", "nunique"),
        total_records=("id", "count"),
        extreme_records=("extreme_flag", "sum"),
        median_area=("area", "median"),
        median_production=("production", "median"),
        median_yield=("calculated_yield", "median"),
        maximum_yield=("calculated_yield", "max"),
        minimum_yield=("calculated_yield", "min"),
    )
    .reset_index()
    .sort_values("extreme_records", ascending=False)
)

summary.to_csv(SUMMARY, index=False)

print("=" * 100)
print("SUGARCANE — COMPLETE AFFECTED DISTRICT HISTORY")
print("=" * 100)

print("\nAffected districts:", len(affected_districts))
print("Historical records:", len(history))
print("Historical years:", history["year"].nunique())

print("\nDISTRICT SUMMARY")
print(summary.to_string(index=False))

print("\nKERALA — KANNUR HISTORY")
print(
    history[
        (history["state_name"] == "Kerala")
        & (history["district_name"] == "Kannur")
    ][
        [
            "id", "year", "season", "area",
            "production", "calculated_yield", "extreme_flag"
        ]
    ].to_string(index=False)
)

print("\nSaved:", OUTPUT)
print("Saved:", SUMMARY)
