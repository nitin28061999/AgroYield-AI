from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "processed" / "splits_v1"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TARGET = "yield"

COMMON_FEATURES = [
    "start_year",
    "area",
    "state_code",
    "district_code",
    "crop_type",
    "season",
]

EXPERIMENTS = {
    "A_separate_crop_features": {
        "features": [
            "start_year",
            "area",
            "state_code",
            "district_code",
            "crop_code",
            "crop_name",
            "crop_type",
            "season",
        ],
        "categorical": [
            "state_code",
            "district_code",
            "crop_code",
            "crop_name",
            "crop_type",
            "season",
        ],
    },
    "B_combined_crop_identity": {
        "features": [
            "start_year",
            "area",
            "state_code",
            "district_code",
            "crop_identity",
            "crop_type",
            "season",
        ],
        "categorical": [
            "state_code",
            "district_code",
            "crop_identity",
            "crop_type",
            "season",
        ],
    },
}

def calculate_metrics(actual, predicted):
    return {
        "records": len(actual),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": mean_squared_error(actual, predicted) ** 0.5,
        "R2": r2_score(actual, predicted),
        "mean_actual": actual.mean(),
        "mean_predicted": predicted.mean(),
        "bias": (predicted - actual).mean(),
        "median_absolute_error": (predicted - actual).abs().median(),
    }


def train_experiment(name, config, train, validation):
    features = config["features"]
    categorical = config["categorical"]

    X_train = train[features].copy()
    X_val = validation[features].copy()
    y_train = train[TARGET]
    y_val = validation[TARGET]

    for column in categorical:
        X_train[column] = X_train[column].astype(str)
        X_val[column] = X_val[column].astype(str)

    model = CatBoostRegressor(
        iterations=1500,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="MAE",
        random_seed=42,
        early_stopping_rounds=100,
        allow_writing_files=False,
        thread_count=-1,
        verbose=100,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    predictions = pd.Series(model.predict(X_val), index=validation.index)
    metrics = calculate_metrics(y_val, predictions)
    metrics["experiment"] = name
    metrics["best_iteration"] = model.get_best_iteration()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model.save_model(MODEL_DIR / f"catboost_identity_{name}.cbm")

    prediction_report = pd.DataFrame({
        "id": validation["id"],
        "crop_name": validation["crop_name"],
        "crop_code": validation["crop_code"],
        "crop_identity": validation["crop_identity"],
        "year": validation["start_year"],
        "actual_yield": y_val,
        "predicted_yield": predictions,
        "absolute_error": (y_val - predictions).abs(),
        "statistical_outlier_flag": validation["statistical_outlier_flag"],
    })

    prediction_report.to_csv(
        REPORT_DIR / f"catboost_identity_{name}_predictions.csv",
        index=False,
    )

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        REPORT_DIR / f"catboost_identity_{name}_importance.csv",
        index=False,
    )

    # Crop-level performance for identities with at least 30 validation rows.
    crop_results = []
    for identity, group in prediction_report.groupby("crop_identity"):
        if len(group) < 30:
            continue

        crop_metrics = calculate_metrics(
            group["actual_yield"],
            group["predicted_yield"],
        )
        crop_metrics["crop_identity"] = identity
        crop_results.append(crop_metrics)

    pd.DataFrame(crop_results).to_csv(
        REPORT_DIR / f"catboost_identity_{name}_crop_performance.csv",
        index=False,
    )

    # Outlier subgroup performance.
    outlier_results = []
    for flag, group in prediction_report.groupby("statistical_outlier_flag"):
        subgroup = calculate_metrics(
            group["actual_yield"],
            group["predicted_yield"],
        )
        subgroup["statistical_outlier_flag"] = bool(flag)
        outlier_results.append(subgroup)

    pd.DataFrame(outlier_results).to_csv(
        REPORT_DIR / f"catboost_identity_{name}_outlier_performance.csv",
        index=False,
    )

    print(f"\n{name} RESULTS")
    print("-" * 65)
    for key, value in metrics.items():
        print(f"{key:25s}: {value}")

    print("\nFeature importance:")
    print(importance.to_string(index=False))

    return metrics


def main():  # sourcery skip: for-append-to-extend, list-comprehension
    train = pd.read_csv(SPLIT_DIR / "train.csv", low_memory=False)
    validation = pd.read_csv(
        SPLIT_DIR / "validation.csv",
        low_memory=False,
    )

    print("AGROYIELD AI — CROP IDENTITY A/B EXPERIMENT")
    print("=" * 70)
    print(f"Training records   : {len(train):,}")
    print(f"Validation records : {len(validation):,}")

    for name, config in EXPERIMENTS.items():
        required = config["features"] + [TARGET]
        for split_name, data in [
            ("train", train),
            ("validation", validation),
        ]:
            missing = data[required].isna().sum()
            if missing.any():
                raise ValueError(
                    f"Missing values in {split_name} for {name}:\n"
                    f"{missing[missing > 0]}"
                )

    results = []
    for name, config in EXPERIMENTS.items():
        results.append(
            train_experiment(name, config, train, validation)
        )

    comparison = pd.DataFrame(results)
    comparison.to_csv(
        REPORT_DIR / "catboost_crop_identity_ab_comparison.csv",
        index=False,
    )

    print("\nFINAL A/B COMPARISON")
    print("=" * 70)
    print(
        comparison[
            ["experiment", "MAE", "RMSE", "R2", "bias",
             "median_absolute_error", "best_iteration"]
        ].to_string(index=False)
    )

    print("\nSaved comparison: reports/catboost_crop_identity_ab_comparison.csv")
    print("Original CatBoost v1.0 model and reports were not modified.")
    print("Test data was not loaded or evaluated.")


if __name__ == "__main__":
    main()
