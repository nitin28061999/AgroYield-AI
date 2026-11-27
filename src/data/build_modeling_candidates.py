
"""Build modeling candidates while preserving outlier review flags."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT / "data" / "processed"
    / "crop_yield_quality_controlled.csv"
)
OUTLIER_PATH = ROOT / "reports" / "training_candidate_outliers.csv"

OUTPUT_PATH = (
    ROOT / "data" / "processed"
    / "crop_yield_modeling_candidates.csv"
)
SUMMARY_PATH = (
    ROOT / "reports"
    / "modeling_candidate_summary.csv"
)


def main():
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    outliers = pd.read_csv(OUTLIER_PATH, low_memory=False)

    if df["id"].duplicated().any():
        raise ValueError("Duplicate IDs found in quality-controlled data.")

    if outliers["id"].duplicated().any():
        raise ValueError("Duplicate IDs found in outlier audit.")

    outlier_ids = set(outliers["id"])

    df["statistical_outlier_flag"] = df["id"].isin(outlier_ids)

    df["review_status"] = "not_flagged"
    df.loc[
        df["statistical_outlier_flag"],
        "review_status",
    ] = "statistical_outlier_review"

    # Preserve only eligible records for this modeling candidate file.
    candidates = df[
        df["eligible_for_primary_training"].eq(True)
    ].copy()

    summary = (
        candidates.groupby("review_status")
        .agg(
            records=("id", "size"),
            crop_identities=("crop_code", "nunique"),
            states=("state_name", "nunique"),
        )
        .reset_index()
    )

    print("\nMODELING CANDIDATE SUMMARY")
    print("=" * 75)
    print(summary.to_string(index=False))

    print("\nTOTAL CANDIDATES:", len(candidates))
    print(
        "OUTLIER FLAGS:",
        int(candidates["statistical_outlier_flag"].sum()),
    )

    candidates.to_csv(OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print(f"\nSaved: {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Saved: {SUMMARY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
