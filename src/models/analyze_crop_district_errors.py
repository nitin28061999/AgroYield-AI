
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS = ROOT / "reports" / "catboost_v1_validation_predictions.csv"
VALIDATION = ROOT / "data" / "processed" / "splits_v1" / "validation.csv"

focus = {
    "Sugarcane": [23],
    "Banana": [28, 14],
    "Onion": [28],
    "Cashewnut": [29],
    "Jute": [19, 10],
    "Turmeric": [28, 6],
}

pred = pd.read_csv(PREDICTIONS, dtype={"id": "string"})
val = pd.read_csv(VALIDATION, dtype={"id": "string"})

assert pred["id"].is_unique
assert val["id"].is_unique

df = pred.merge(
    val[["id", "state_code", "district_code", "season"]],
    on="id",
    how="left",
    validate="one_to_one",
)

assert df["district_code"].notna().all()

df["absolute_error"] = (
    df["actual_yield"] - df["predicted_yield"]
).abs()

df["squared_error"] = df["absolute_error"] ** 2

# Keep only the state/crop combinations identified in the previous audit.
mask = pd.Series(False, index=df.index)

for crop, states in focus.items():
    mask |= (
        (df["crop_name"] == crop)
        & (df["state_code"].isin(states))
    )

subset = df[mask].copy()

# District-level performance.
districts = (
    subset.groupby(
        ["crop_name", "state_code", "district_code"],
        dropna=False,
    )
    .agg(
        records=("actual_yield", "size"),
        MAE=("absolute_error", "mean"),
        RMSE=("squared_error", lambda x: x.mean() ** 0.5),
        mean_actual=("actual_yield", "mean"),
        mean_predicted=("predicted_yield", "mean"),
    )
    .reset_index()
)

districts = districts[districts["records"] >= 3]
districts = districts.sort_values(
    ["crop_name", "MAE"],
    ascending=[True, False],
)

# Year and season concentration.
periods = (
    subset.groupby(
        ["crop_name", "state_code", "year", "season"],
        dropna=False,
    )
    .agg(
        records=("actual_yield", "size"),
        MAE=("absolute_error", "mean"),
        RMSE=("squared_error", lambda x: x.mean() ** 0.5),
    )
    .reset_index()
    .sort_values(["crop_name", "MAE"], ascending=[True, False])
)

# Individual largest errors.
worst = subset.sort_values(
    "absolute_error",
    ascending=False,
).head(40)

districts.to_csv(
    ROOT / "reports" / "catboost_v1_crop_district_errors.csv",
    index=False,
)

periods.to_csv(
    ROOT / "reports" / "catboost_v1_crop_period_errors.csv",
    index=False,
)

worst.to_csv(
    ROOT / "reports" / "catboost_v1_crop_geographic_worst_records.csv",
    index=False,
)

for crop in focus:
    print(f"\n{'=' * 85}\n{crop}\n{'=' * 85}")

    print("\nWorst districts:")
    print(
        districts[districts["crop_name"] == crop]
        .head(8)
        .to_string(index=False)
    )

    print("\nWorst year-season combinations:")
    print(
        periods[periods["crop_name"] == crop]
        .head(8)
        .to_string(index=False)
    )

print("\nSaved three geographic error investigation reports.")
