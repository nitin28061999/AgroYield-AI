
"""Audit extreme yields among primary training candidates."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT / "data" / "processed"
    / "crop_yield_quality_controlled.csv"
)
OUTPUT_PATH = ROOT / "reports" / "training_candidate_outliers.csv"
SUMMARY_PATH = ROOT / "reports" / "training_candidate_outlier_summary.csv"


def main():
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    candidates = df[
        df["eligible_for_primary_training"].eq(True)
    ].copy()

    candidates["yield"] = pd.to_numeric(
        candidates["yield"], errors="coerce"
    )

    candidates = candidates.dropna(subset=["yield"])

    keys = ["crop_code", "crop_name"]

    stats = (
        candidates.groupby(keys, dropna=False)["yield"]
        .agg(
            records="count",
            median="median",
            p95=lambda x: x.quantile(0.95),
            p99=lambda x: x.quantile(0.99),
            p995=lambda x: x.quantile(0.995),
            maximum="max",
        )
        .reset_index()
    )

    candidates = candidates.merge(
        stats[keys + ["records", "p99", "p995"]],
        on=keys,
        how="left",
    )

    # Screen only sufficiently represented crop identities.
    candidates["outlier_flag"] = (
        candidates["records"].ge(30)
        & candidates["yield"].gt(candidates["p995"])
    )

    outliers = candidates[
        candidates["outlier_flag"]
    ].copy()

    outliers = outliers.sort_values(
        "yield", ascending=False
    )

    summary = stats.sort_values(
        "maximum", ascending=False
    )

    print("\nTRAINING CANDIDATE OUTLIER AUDIT")
    print("=" * 75)
    print(f"Eligible records: {len(candidates):,}")
    print(f"Crop identities: {len(stats):,}")
    print(f"Flagged outliers: {len(outliers):,}")

    print("\nTOP 25 EXTREME YIELD RECORDS")
    print("=" * 75)

    columns = [
        "state_name",
        "district_name",
        "crop_name",
        "crop_code",
        "year",
        "season",
        "area",
        "production",
        "yield",
        "p99",
        "p995",
    ]

    print(outliers[columns].head(25).to_string(index=False))

    outliers.to_csv(OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print(f"\nOutliers saved: {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Summary saved: {SUMMARY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
