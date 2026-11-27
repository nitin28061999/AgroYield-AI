
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data" / "raw" / "crop-wise-area-production-yield.csv"
WORST = ROOT / "reports" / "catboost_v1_worst_records_raw_audit.csv"
OUTPUT = ROOT / "reports" / "catboost_v1_worst_record_history.csv"

raw = pd.read_csv(
    RAW,
    dtype={"id": "string"},
    low_memory=False,
)

worst = pd.read_csv(
    WORST,
    dtype={"id": "string"},
)

# Only use the 20 investigated validation records as anchors.
anchors = worst.sort_values(
    "absolute_error",
    ascending=False,
).copy()

raw["start_year"] = (
    raw["year"].str.extract(r"^(\d{4})")[0].astype(int)
)

history = []

for _, target in anchors.iterrows():
    mask = (
        (raw["state_code"] == target["state_code_raw"])
        & (raw["district_code"] == target["district_code_raw"])
        & (raw["crop_code"] == target["crop_code_raw"])
        & (raw["crop_name"] == target["crop_name_raw"])
        & (raw["season"] == target["season_raw"])
        & (
            raw["start_year"].between(
                int(target["year_raw"][:4]) - 5,
                int(target["year_raw"][:4]) + 1,
            )
        )
    )

    context = raw.loc[mask].copy()

    context["anchor_id"] = target["id"]
    context["anchor_error"] = target["absolute_error"]
    context["anchor_actual_yield"] = target["actual_yield"]
    context["anchor_predicted_yield"] = target["predicted_yield"]

    context["relative_to_anchor"] = context["id"].map(
        lambda value: "TARGET" if value == target["id"] else "CONTEXT"
    )

    history.append(context)

result = pd.concat(history, ignore_index=True)

columns = [
    "anchor_id",
    "anchor_error",
    "relative_to_anchor",
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
]

result = result[columns].sort_values(
    ["anchor_id", "year", "id"]
)

result.to_csv(OUTPUT, index=False)

print("WORST RECORD HISTORICAL CONTEXT AUDIT")
print("=" * 100)
print("Anchor records:", anchors["id"].nunique())
print("Context records:", len(result))
print("Output:", OUTPUT)

for anchor_id, group in result.groupby("anchor_id", sort=False):
    target = group[group["relative_to_anchor"] == "TARGET"]

    print("\n" + "-" * 100)
    print(
        f"Anchor ID: {anchor_id} | "
        f"Crop: {target.iloc[0]['crop_name']} | "
        f"District: {target.iloc[0]['district_name']} | "
        f"Season: {target.iloc[0]['season']}"
    )

    print(
        group[
            [
                "relative_to_anchor",
                "year",
                "area",
                "production",
                "yield",
            ]
        ].to_string(index=False)
    )

print("\nRaw dataset was not modified.")
