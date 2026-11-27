
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/raw/crop-wise-area-production-yield.csv"
OUTPUT = ROOT / "reports/onion_seasonal_reconciliation.csv"

df = pd.read_csv(INPUT, low_memory=False)

df["crop_code_num"] = pd.to_numeric(
    df["crop_code"], errors="coerce"
)

onion = df[
    (df["crop_code_num"] == 708)
    & (df["state_name"] == "Andhra Pradesh")
    & (df["year"].isin([
        "2015-2016",
        "2016-2017",
        "2017-2018",
        "2018-2019",
        "2019-2020",
    ]))
].copy()

keys = ["year", "state_name", "district_name"]

seasonal = onion[onion["season"] != "Total"].copy()
totals = onion[onion["season"] == "Total"].copy()

season_sums = (
    seasonal.groupby(keys, as_index=False)
    .agg(
        seasonal_area=("area", "sum"),
        seasonal_production=("production", "sum"),
        seasonal_records=("id", "count"),
    )
)

total_values = totals[
    keys + ["area", "production", "id"]
].rename(columns={
    "area": "total_area",
    "production": "total_production",
    "id": "total_record_id",
})

result = season_sums.merge(
    total_values,
    on=keys,
    how="outer",
    indicator=True,
)

result["area_difference"] = (
    result["seasonal_area"] - result["total_area"]
)

result["production_difference"] = (
    result["seasonal_production"] - result["total_production"]
)

result["area_match"] = (
    result["area_difference"].abs() <= 0.01
)

result["production_match"] = (
    result["production_difference"].abs() <= 0.01
)

result["seasonal_yield"] = (
    result["seasonal_production"] / result["seasonal_area"]
)

result["total_yield"] = (
    result["total_production"] / result["total_area"]
)

result.to_csv(OUTPUT, index=False)

print("=" * 100)
print("ONION — SEASONAL VS TOTAL RECONCILIATION")
print("=" * 100)

print("\nDistrict-year groups:", len(result))
print("\nMerge status:")
print(result["_merge"].value_counts().to_string())

print("\nArea matches:", result["area_match"].sum())
print("Production matches:", result["production_match"].sum())

print("\nGroups where seasonal production matches Total but area does not:")
print(
    result[
        result["production_match"]
        & ~result["area_match"]
    ].to_string(index=False)
)

print("\nGroups where neither area nor production matches:")
print(
    result[
        ~result["area_match"]
        & ~result["production_match"]
    ].to_string(index=False)
)

print("\nSaved:", OUTPUT)
