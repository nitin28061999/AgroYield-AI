from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS = (
    ROOT / "reports" / "catboost_v1_validation_predictions.csv"
)
VALIDATION = (
    ROOT / "data" / "processed" / "splits_v1" / "validation.csv"
)
OUTPUT = ROOT / "reports" / "catboost_v1_crop_state_errors.csv"

focus_crops = [
    "Sugarcane",
    "Banana",
    "Onion",
    "Cashewnut",
    "Jute",
    "Turmeric",
]

pred = pd.read_csv(PREDICTIONS, dtype={"id": "string"})
val = pd.read_csv(VALIDATION, dtype={"id": "string"})

# Verify unique record identifiers before joining.
assert pred["id"].is_unique, "Duplicate IDs in prediction report"
assert val["id"].is_unique, "Duplicate IDs in validation split"

# Restore geographic information from the original validation split.
geo = val[["id", "state_code", "district_code"]]

df = pred.merge(
    geo,
    on="id",
    how="left",
    validate="one_to_one",
    indicator=True,
)

assert (df["_merge"] == "both").all(), "Some prediction IDs did not match"
df = df.drop(columns="_merge")

df["absolute_error"] = (
    df["actual_yield"] - df["predicted_yield"]
).abs()

df["squared_error"] = (
    df["actual_yield"] - df["predicted_yield"]
) ** 2

results = (
    df[df["crop_name"].isin(focus_crops)]
    .groupby(["crop_name", "state_code"], dropna=False)
    .agg(
        records=("actual_yield", "size"),
        MAE=("absolute_error", "mean"),
        RMSE=("squared_error", lambda x: x.mean() ** 0.5),
        mean_actual=("actual_yield", "mean"),
        mean_predicted=("predicted_yield", "mean"),
    )
    .reset_index()
)

results = results[results["records"] >= 10]

results = results.sort_values(
    ["crop_name", "MAE"],
    ascending=[True, False],
)

results.to_csv(OUTPUT, index=False)

print("\nCROP-STATE ERROR INVESTIGATION")
print("=" * 90)

for crop in focus_crops:
    subset = results[results["crop_name"] == crop]

    print(f"\n{crop}")
    print("-" * 80)

    print(
        subset.head(10).to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

print(f"\nSaved: {OUTPUT}")
