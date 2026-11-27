from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "crop_yield_modeling_candidates.csv"
OUTPUT = ROOT / "reports" / "coconut_state_season_audit.csv"

df = pd.read_csv(
    INPUT,
    dtype={"crop_code": "string"},
)

df["crop_code_normalized"] = (
    pd.to_numeric(df["crop_code"], errors="coerce")
    .astype("Int64")
)

coconut = df[
    df["crop_name"].str.strip().str.lower().eq("coconut")
    & df["crop_code_normalized"].eq(1006)
].copy()

# Focus on the two validation years and the three states
# containing some of the highest reported yields.
focus = coconut[
    coconut["year"].isin(["2018-2019", "2019-2020"])
    & coconut["state_name"].isin(
        ["Assam", "Andhra Pradesh", "Tamil Nadu"]
    )
].copy()

summary = (
    focus.groupby(
        ["year", "state_name", "season"],
        dropna=False,
    )
    .agg(
        records=("id", "size"),
        districts=("district_name", "nunique"),
        total_area=("area", "sum"),
        total_production=("production", "sum"),
        mean_reported_yield=("yield", "mean"),
        median_reported_yield=("yield", "median"),
        maximum_reported_yield=("yield", "max"),
    )
    .reset_index()
)

summary["recomputed_yield"] = (
    summary["total_production"] / summary["total_area"]
)

summary = summary.sort_values(
    ["year", "state_name", "season"]
)

summary.to_csv(OUTPUT, index=False)

print("\nCOCONUT STATE-SEASON UNIT AUDIT")
print("=" * 100)
print(summary.round(3).to_string(index=False))

print("\nORIGINAL HIGH-YIELD RECORDS")
print("=" * 100)

columns = [
    "id", "year", "state_name", "district_name",
    "season", "area", "production", "yield",
    "area_unit", "production_unit", "yield_unit",
]

print(
    focus.nlargest(15, "yield")[columns]
    .to_string(index=False)
)

print(f"\nSaved: {OUTPUT}")
