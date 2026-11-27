
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/raw/crop-wise-area-production-yield.csv"

df = pd.read_csv(INPUT, low_memory=False)

targets = [
    ("Tamil Nadu", "The Nilgiris", "1997-1998"),
    ("Tamil Nadu", "Thoothukkudi", "1997-1998"),
    ("Chhattisgarh", "Narayanpur", "2021-2022"),
    ("Kerala", "Kannur", "2020-2021"),
    ("Kerala", "Kannur", "2021-2022"),
    ("Kerala", "Kannur", "2022-2023"),
]

for state, district, year in targets:
    subset = df[
        (df["state_name"] == state)
        & (df["district_name"] == district)
        & (df["year"] == year)
    ].copy()

    print("\n" + "=" * 100)
    print(f"{state} | {district} | {year}")
    print("=" * 100)

    if subset.empty:
        print("No matching records")
        continue

    print(
        subset[
            [
                "id", "crop_name", "crop_code", "season",
                "area", "production", "yield",
                "area_unit", "production_unit", "yield_unit"
            ]
        ].to_string(index=False)
    )

    print("\nDistinct crop names:", subset["crop_name"].nunique())
    print("Distinct crop codes:", subset["crop_code"].nunique())
    print("Distinct seasons:", subset["season"].nunique())
