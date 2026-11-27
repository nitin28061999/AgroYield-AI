# sourcery skip: for-append-to-extend
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data" / "processed" / "crop_yield_modeling_candidates.csv"
OUTPUT = ROOT / "reports" / "extreme_crop_record_investigation.csv"

df = pd.read_csv(
    INPUT,
    dtype={"crop_code": "string"},
)

# Crop codes and approximate official national reference yields
# in tonnes per hectare.
benchmarks = {
    "619": {"crop": "Cashewnut", "2018-2019": 0.672, "2019-2020": 0.625},
    "708": {"crop": "Onion", "2018-2019": 18.705, "2019-2020": 18.245},
    "401": {"crop": "Sugarcane", "2018-2019": 80.0, "2019-2020": 80.0},
    "104": {"crop": "Maize", "2018-2019": 2.700, "2019-2020": 3.066},
}

df["crop_code_normalized"] = (
    pd.to_numeric(df["crop_code"], errors="coerce")
    .astype("Int64")
    .astype("string")
)

records = []

for code, details in benchmarks.items():
    crop_data = df[
        (df["crop_code_normalized"] == code)
        & (df["year"].isin(["2018-2019", "2019-2020"]))
    ].copy()

    for year, benchmark in details.items():
        year_data = crop_data[crop_data["year"] == year]

        if year_data.empty:
            continue

        for _, row in year_data.nlargest(10, "yield").iterrows():
            records.append(
                {
                    "crop": details["crop"],
                    "crop_code": code,
                    "year": year,
                    "state": row["state_name"],
                    "district": row["district_name"],
                    "area": row["area"],
                    "production": row["production"],
                    "reported_yield": row["yield"],
                    "national_reference_yield": benchmark,
                    "multiple_of_reference": (
                        row["yield"] / benchmark
                        if benchmark else None
                    ),
                    "outlier_flag": row["statistical_outlier_flag"],
                    "production_unit": row["production_unit"],
                    "yield_unit": row["yield_unit"],
                }
            )

result = pd.DataFrame(records)

result = result.sort_values(
    ["crop", "year", "reported_yield"],
    ascending=[True, True, False],
)

result.to_csv(OUTPUT, index=False)

print("\nEXTREME CROP RECORD INVESTIGATION")
print("=" * 100)

if result.empty:
    print("No matching records found.")
else:
    print(
        result.round(3).to_string(index=False)
    )

print(f"\nSaved: {OUTPUT}")
