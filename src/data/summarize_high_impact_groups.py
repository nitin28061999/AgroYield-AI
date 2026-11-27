
"""Create a one-row-per-group review of high-impact repetitions."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "reports" / "high_impact_original_records.csv"
OUTPUT_PATH = ROOT / "reports" / "high_impact_group_review.csv"


def summarize_groups():
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    summary = (
        df.groupby("audit_group_id", dropna=False)
        .agg(
            year=("year", "first"),
            state=("state_name", "first"),
            crop=("crop_name", "first"),
            crop_code=("crop_code", "first"),
            season=("season", "first"),
            area=("area", "first"),
            area_unit=("area_unit", "first"),
            area_unit_count=("area_unit", "nunique"),
            production=("production", "first"),
            production_unit=("production_unit", "first"),
            production_unit_count=("production_unit", "nunique"),
            yield_value=("yield", "first"),
            yield_unit=("yield_unit", "first"),
            yield_unit_count=("yield_unit", "nunique"),
            record_count=("id", "size"),
            district_count=("district_name", "nunique"),
            missing_production_records=("production", lambda s: s.isna().sum()),
            zero_production_records=(
                "production", lambda s: (s == 0).sum()
            ),
            districts=(
                "district_name",
                lambda s: " | ".join(sorted(s.dropna().unique())),
            ),
            record_ids=(
                "id",
                lambda s: " | ".join(s.astype(str)),
            ),
        )
        .reset_index()
    )

    summary.to_csv(OUTPUT_PATH, index=False)

    print("\nHIGH-IMPACT GROUP REVIEW")
    print("=" * 75)
    print("Groups reviewed:", len(summary))
    print(
        "Groups with inconsistent area units:",
        (summary["area_unit_count"] > 1).sum(),
    )
    print(
        "Groups with inconsistent production units:",
        (summary["production_unit_count"] > 1).sum(),
    )
    print(
        "Groups with inconsistent yield units:",
        (summary["yield_unit_count"] > 1).sum(),
    )
    print(
        "Groups with missing production records:",
        (summary["missing_production_records"] > 0).sum(),
    )

    print("\nGROUP REVIEW")
    print("=" * 75)
    print(
        summary[
            [
                "audit_group_id",
                "year",
                "state",
                "crop",
                "season",
                "area",
                "production",
                "record_count",
                "district_count",
            ]
        ].to_string(index=False)
    )

    print(f"\nReport saved: {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    summarize_groups()
