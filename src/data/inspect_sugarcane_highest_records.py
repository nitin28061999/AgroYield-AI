
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data/raw/crop-wise-area-production-yield.csv"
OUTPUT = ROOT / "reports/sugarcane_highest_historical_records.csv"

df = pd.read_csv(INPUT, low_memory=False)

df["crop_code_num"] = pd.to_numeric(
    df["crop_code"], errors="coerce"
)

for col in ["area", "production", "yield"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[
    (df["crop_code_num"] == 401)
    & (df["season"] != "Total")
].copy()

df["calculated_yield"] = (
    df["production"] / df["area"]
)

districts = [
    ("Tamil Nadu", "The Nilgiris"),
    ("Tamil Nadu", "Thoothukkudi"),
    ("Chhattisgarh", "Narayanpur"),
    ("Kerala", "Kannur"),
]

for state, district in districts:
    subset = df[
        (df["state_name"] == state)
        & (df["district_name"] == district)
    ].sort_values("calculated_yield", ascending=False)

    print("\n" + "=" * 100)
    print(f"{state} — {district}")
    print("=" * 100)

    print(
        subset[
            [
                "id", "year", "season", "area",
                "production", "yield", "calculated_yield",
                "area_unit", "production_unit", "yield_unit"
            ]
        ].head(10).to_string(index=False)
    )

    print("\nAll records:", len(subset))
    print("Median yield:", round(subset["calculated_yield"].median(), 3))

    print("\nChronological history:")
    print(
        subset.sort_values("year")[
            ["year", "season", "area", "production", "calculated_yield"]
        ].to_string(index=False)
    )

df.sort_values(
    "calculated_yield", ascending=False
).head(100).to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
