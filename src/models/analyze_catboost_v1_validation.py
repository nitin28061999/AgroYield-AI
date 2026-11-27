
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_PATH = (
    ROOT / "reports" / "catboost_v1_validation_predictions.csv"
)
OUTPUT_DIR = ROOT / "reports"


def calculate_metrics(group):
    actual = group["actual_yield"]
    predicted = group["predicted_yield"]
    errors = predicted - actual

    return pd.Series({
        "records": len(group),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted) if len(group) > 1 else np.nan,
        "mean_actual": actual.mean(),
        "mean_predicted": predicted.mean(),
        "bias": errors.mean(),
        "median_absolute_error": errors.abs().median(),
        "p95_absolute_error": errors.abs().quantile(0.95),
        "outlier_records": int(group["statistical_outlier_flag"].sum()),
    })


def main():  # sourcery skip: extract-duplicate-method
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PREDICTIONS_PATH, low_memory=False)

    required = {
        "crop_code",
        "crop_name",
        "actual_yield",
        "predicted_yield",
        "absolute_error",
        "statistical_outlier_flag",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Normalize crop codes for consistent grouping.
    numeric_code = pd.to_numeric(df["crop_code"], errors="coerce")
    df["crop_code"] = numeric_code.astype("Int64").astype("string")

    # Crop-level validation performance.
    crop_metrics = (
        df.groupby(["crop_code", "crop_name"], dropna=False)
        .apply(calculate_metrics, include_groups=False) # pyright: ignore[reportCallIssue]
        .reset_index()
    )

    crop_metrics = crop_metrics.sort_values(
        ["MAE", "records"], ascending=[False, False]
    )

    crop_metrics.to_csv(
        OUTPUT_DIR / "catboost_v1_crop_performance.csv",
        index=False,
    )

    # Focus on crops with at least 30 validation records.
    established = crop_metrics[crop_metrics["records"] >= 30]

    established.to_csv(
        OUTPUT_DIR / "catboost_v1_crop_performance_min30.csv",
        index=False,
    )

    # Largest individual prediction errors.
    worst = df.sort_values(
        "absolute_error", ascending=False
    ).head(50)

    worst.to_csv(
        OUTPUT_DIR / "catboost_v1_worst_predictions.csv",
        index=False,
    )

    # Crop-level outlier concentration.
    outlier_summary = (
        df.groupby(["crop_code", "crop_name"], dropna=False)
        .agg(
            records=("actual_yield", "size"),
            outlier_records=("statistical_outlier_flag", "sum"),
            mean_actual=("actual_yield", "mean"),
            mean_predicted=("predicted_yield", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            max_absolute_error=("absolute_error", "max"),
        )
        .reset_index()
    )

    outlier_summary["outlier_rate_pct"] = (
        outlier_summary["outlier_records"]
        / outlier_summary["records"] * 100
    )

    outlier_summary = outlier_summary.sort_values(
        ["outlier_records", "mean_absolute_error"],
        ascending=False,
    )

    outlier_summary.to_csv(
        OUTPUT_DIR / "catboost_v1_crop_outlier_concentration.csv",
        index=False,
    )

    print("\nCATBOOST v1.0 — CROP-LEVEL VALIDATION ANALYSIS")
    print("=" * 90)

    print("\nCROPS WITH AT LEAST 30 VALIDATION RECORDS — HIGHEST MAE")
    print(
        established.head(20).round(4).to_string(index=False)
    )

    print("\nCROPS WITH AT LEAST 30 VALIDATION RECORDS — LOWEST MAE")
    print(
        established.sort_values("MAE").head(15)
        .round(4).to_string(index=False)
    )

    print("\nCROPS WITH MOST FLAGGED OUTLIERS")
    print(
        outlier_summary.head(15).round(4).to_string(index=False)
    )

    print("\nTOP 20 INDIVIDUAL PREDICTION ERRORS")
    print(
        worst.head(20).to_string(index=False)
    )

    print("\nReports saved:")
    print("  catboost_v1_crop_performance.csv")
    print("  catboost_v1_crop_performance_min30.csv")
    print("  catboost_v1_worst_predictions.csv")
    print("  catboost_v1_crop_outlier_concentration.csv")


if __name__ == "__main__":
    main()
