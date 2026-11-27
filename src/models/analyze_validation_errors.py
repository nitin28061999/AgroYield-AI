
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
PREDICTIONS_PATH = ROOT_DIR / "reports" / "validation_predictions.csv"
REPORT_DIR = ROOT_DIR / "reports"

df = pd.read_csv(
    PREDICTIONS_PATH,
    dtype={"crop_code": "string"},
)

# Normalize crop identity and calculate prediction residuals.
df["crop_code"] = df["crop_code"].fillna("MISSING")
df["crop_identity"] = df["crop_code"] + " | " + df["crop_name"]

df["prediction_error"] = df["predicted_yield"] - df["yield"]
df["absolute_error"] = df["prediction_error"].abs()

# Percentage error is supplementary; avoid division by zero.
df["absolute_percentage_error"] = np.where(
    df["yield"] != 0,
    df["absolute_error"] / df["yield"] * 100,
    np.nan,
)


def calculate_metrics(group):
    return pd.Series(
        {
            "records": len(group),
            "MAE": group["absolute_error"].mean(),
            "RMSE": np.sqrt(
                np.mean(group["prediction_error"] ** 2)
            ),
            "mean_actual_yield": group["yield"].mean(),
            "mean_predicted_yield": group["predicted_yield"].mean(),
            "prediction_bias": group["prediction_error"].mean(),
            "median_absolute_error": group["absolute_error"].median(),
        }
    )


# 1. Overall validation error summary.
overall = calculate_metrics(df)
overall.to_frame("value").to_csv(
    REPORT_DIR / "validation_error_summary.csv"
)

# 2. Performance by crop identity.
crop_metrics = (
    df.groupby("crop_identity")
    .apply(calculate_metrics, include_groups=False) # pyright: ignore[reportCallIssue]
    .reset_index()
)

crop_metrics = crop_metrics.sort_values(
    "MAE", ascending=False
)

crop_metrics.to_csv(
    REPORT_DIR / "crop_validation_error_analysis.csv",
    index=False,
)

# 3. Performance by validation year.
year_metrics = (
    df.groupby("year")
    .apply(calculate_metrics, include_groups=False) # pyright: ignore[reportCallIssue]
    .reset_index()
)

year_metrics.to_csv(
    REPORT_DIR / "year_validation_error_analysis.csv",
    index=False,
)

# 4. Performance by statistical outlier status.
outlier_metrics = (
    df.groupby("statistical_outlier_flag")
    .apply(calculate_metrics, include_groups=False) # pyright: ignore[reportCallIssue]
    .reset_index()
)

outlier_metrics.to_csv(
    REPORT_DIR / "validation_error_by_outlier_status.csv",
    index=False,
)

# 5. Twenty largest absolute prediction errors.
worst_predictions = df.nlargest(
    20, "absolute_error"
)

worst_predictions.to_csv(
    REPORT_DIR / "worst_validation_predictions.csv",
    index=False,
)

# 6. Display key findings.
print("\nOVERALL VALIDATION ERROR")
print("=" * 75)
print(overall.round(4).to_string())

print("\nWORST-PERFORMING CROPS (minimum 30 records)")
print("=" * 75)

print(
    crop_metrics[crop_metrics["records"] >= 30]
    .head(15)
    .round(3)
    .to_string(index=False)
)

print("\nPERFORMANCE BY YEAR")
print("=" * 75)
print(year_metrics.round(3).to_string(index=False))

print("\nPERFORMANCE BY OUTLIER STATUS")
print("=" * 75)
print(outlier_metrics.round(3).to_string(index=False))

print("\n20 LARGEST PREDICTION ERRORS")
print("=" * 75)

columns = [
    "crop_name",
    "crop_code",
    "year",
    "yield",
    "predicted_yield",
    "absolute_error",
    "statistical_outlier_flag",
]

print(worst_predictions[columns].round(3).to_string(index=False))

print("\nSaved reports:")
print("- reports/validation_error_summary.csv")
print("- reports/crop_validation_error_analysis.csv")
print("- reports/year_validation_error_analysis.csv")
print("- reports/validation_error_by_outlier_status.csv")
print("- reports/worst_validation_predictions.csv")
