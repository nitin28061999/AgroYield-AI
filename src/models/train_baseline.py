"""Compare crop-median baseline with CatBoost on validation data."""

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = ROOT / "data" / "processed" / "splits"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TARGET = "yield"

NUMERIC_FEATURES = [
    "start_year",
    "area",
]

CATEGORICAL_FEATURES = [
    "state_code",
    "district_code",
    "crop_code",
    "crop_name",
    "crop_type",
    "season",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data():
    train = pd.read_csv(SPLIT_DIR / "train.csv")
    validation = pd.read_csv(SPLIT_DIR / "validation.csv")

    for df in [train, validation]:
        for col in CATEGORICAL_FEATURES:
            df[col] = df[col].fillna("MISSING").astype(str)

    return train, validation


def evaluate(y_true, predictions):
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_true, predictions)),
        "R2": r2_score(y_true, predictions),
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train, validation = load_data()

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_val = validation[FEATURES]
    y_val = validation[TARGET]

    results = []

    # 1. Global median baseline.
    global_median = float(y_train.median())

    dummy = DummyRegressor(strategy="median")
    dummy.fit(np.zeros((len(train), 1)), y_train)

    dummy_predictions = dummy.predict(
        np.zeros((len(validation), 1))
    )

    results.append({
        "model": "Global median",
        **evaluate(y_val, dummy_predictions),
    })

    # 2. Crop-specific median baseline.
    crop_medians = (
        train.groupby("crop_identity")[TARGET]
        .median()
    )

    crop_predictions = (
        validation["crop_identity"]
        .map(crop_medians)
        .fillna(global_median)
        .to_numpy()
    )

    results.append({
        "model": "Crop median",
        **evaluate(y_val, crop_predictions),
    })

    # 3. CatBoost regression.
    model = CatBoostRegressor(
        iterations=1500,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="MAE",
        random_seed=42,
        verbose=100,
        early_stopping_rounds=100,
        allow_writing_files=False,
        thread_count=-1,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=CATEGORICAL_FEATURES,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    predictions = model.predict(X_val)

    results.append({
        "model": "CatBoost",
        **evaluate(y_val, predictions),
    })

    # Save model and validation predictions.
    model.save_model(MODEL_DIR / "catboost_baseline.cbm")

    prediction_output = validation[
        ["id", "crop_name", "crop_code",
         "yield", "statistical_outlier_flag"]
    ].copy()
    prediction_output["year"] = validation.get(
        "year", validation["start_year"]
    )
    prediction_output["predicted_yield"] = predictions
    prediction_output["absolute_error"] = np.abs(
        prediction_output["yield"] - predictions
    )
    prediction_output = prediction_output[
        ["id", "crop_name", "crop_code", "year",
         "yield", "statistical_outlier_flag",
         "predicted_yield", "absolute_error"]
    ]

    prediction_output.to_csv(
        REPORT_DIR / "validation_predictions.csv",
        index=False,
    )

    # Performance by outlier review status.
    subgroup_results = []

    for flag, group in prediction_output.groupby(
        "statistical_outlier_flag"
    ):
        subgroup_results.append({
            "outlier_flag": flag,
            "records": len(group),
            "MAE": mean_absolute_error(
                group["yield"], group["predicted_yield"]
            ),
            "RMSE": np.sqrt(mean_squared_error(
                group["yield"], group["predicted_yield"]
            )),
        })

    results_df = pd.DataFrame(results)
    subgroup_df = pd.DataFrame(subgroup_results)

    results_df.to_csv(
        REPORT_DIR / "baseline_model_comparison.csv",
        index=False,
    )

    subgroup_df.to_csv(
        REPORT_DIR / "validation_outlier_performance.csv",
        index=False,
    )

    print("\nMODEL COMPARISON")
    print("=" * 75)
    print(results_df.to_string(index=False))

    print("\nCATBOOST BEST ITERATION")
    print("=" * 75)
    print(model.get_best_iteration())

    print("\nVALIDATION PERFORMANCE BY OUTLIER STATUS")
    print("=" * 75)
    print(subgroup_df.to_string(index=False))

    print("\nSaved model:", "models/catboost_baseline.cbm")
    print("Saved reports:")
    print("- reports/baseline_model_comparison.csv")
    print("- reports/validation_predictions.csv")
    print("- reports/validation_outlier_performance.csv")


if __name__ == "__main__":
    main()
