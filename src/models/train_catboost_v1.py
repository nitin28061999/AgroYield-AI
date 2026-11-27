
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = ROOT / "data" / "processed" / "splits_v1"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

FEATURES = [
    "start_year",
    "area",
    "state_code",
    "district_code",
    "crop_code",
    "crop_name",
    "crop_type",
    "season",
]

CATEGORICAL_FEATURES = [
    "state_code",
    "district_code",
    "crop_code",
    "crop_name",
    "crop_type",
    "season",
]


def calculate_metrics(actual, predicted):
    return {
        "records": len(actual),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": mean_squared_error(actual, predicted) ** 0.5,
        "R2": r2_score(actual, predicted),
        "mean_actual": actual.mean(),
        "mean_predicted": predicted.mean(),
        "bias": (predicted - actual).mean(),
        "median_absolute_error": (
            (predicted - actual).abs().median()
        ),
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(
        SPLIT_DIR / "train.csv",
        low_memory=False,
    )
    validation = pd.read_csv(
        SPLIT_DIR / "validation.csv",
        low_memory=False,
    )

    print("AGROYIELD AI — CATBOOST v1.0")
    print("=" * 65)
    print(f"Training records   : {len(train):,}")
    print(f"Validation records : {len(validation):,}")

    for name, data in [("train", train), ("validation", validation)]:
        missing = data[FEATURES + ["yield"]].isna().sum()
        if missing.any():
            raise ValueError(
                f"Missing values in {name} data:\n{missing[missing > 0]}"
            )

    X_train = train[FEATURES].copy()
    y_train = train["yield"]

    X_val = validation[FEATURES].copy()
    y_val = validation["yield"]

    for column in CATEGORICAL_FEATURES:
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
        cat_features=CATEGORICAL_FEATURES,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    predictions = model.predict(X_val)
    prediction_series = pd.Series(predictions, index=validation.index)

    overall = calculate_metrics(y_val, prediction_series)
    overall["best_iteration"] = model.get_best_iteration()

    print("\nVALIDATION PERFORMANCE")
    print("-" * 65)
    for key, value in overall.items():
        if isinstance(value, float):
            print(f"{key:28s}: {value:,.4f}")
        else:
            print(f"{key:28s}: {value}")

    # Outlier subgroup performance.
    outlier_results = []
    for flag, group in validation.groupby("statistical_outlier_flag"):
        idx = group.index
        metrics = calculate_metrics(
            group["yield"],
            prediction_series.loc[idx],
        )
        metrics["outlier_status"] = bool(flag)
        outlier_results.append(metrics)

    outlier_report = pd.DataFrame(outlier_results)
    outlier_report.to_csv(
        REPORT_DIR / "catboost_v1_outlier_performance.csv",
        index=False,
    )

    # Validation predictions for subsequent error analysis.
    prediction_report = pd.DataFrame({
        "id": validation["id"],
        "crop_name": validation["crop_name"],
        "crop_code": validation["crop_code"],
        "year": validation["start_year"],
        "actual_yield": y_val,
        "predicted_yield": predictions,
        "absolute_error": (y_val - predictions).abs(),
        "statistical_outlier_flag": validation[
            "statistical_outlier_flag"
        ],
    })

    prediction_report.to_csv(
        REPORT_DIR / "catboost_v1_validation_predictions.csv",
        index=False,
    )

    # Feature importance.
    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        REPORT_DIR / "catboost_v1_feature_importance.csv",
        index=False,
    )

    pd.DataFrame([overall]).to_csv(
        REPORT_DIR / "catboost_v1_validation_metrics.csv",
        index=False,
    )

    model.save_model(MODEL_DIR / "catboost_v1.cbm")

    print("\nOUTLIER SUBGROUP PERFORMANCE")
    print(outlier_report.to_string(index=False))

    print("\nFEATURE IMPORTANCE")
    print(importance.to_string(index=False))

    print("\nSaved model: models/catboost_v1.cbm")
    print("Test data was not loaded or evaluated.")


if __name__ == "__main__":
    main()
