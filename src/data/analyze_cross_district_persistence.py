
"""Identify districts repeatedly involved in cross-district matches."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "reports" / "high_impact_cross_district.csv"
REPORT_DIR = ROOT / "reports"


def analyze_persistence():  # sourcery skip: extract-duplicate-method
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    # Expand the pipe-separated district names into individual rows.
    df["district"] = df["districts"].fillna("").str.split(
        " | ",
        regex=False,
    )
    expanded = df.explode("district")

    expanded["district"] = expanded["district"].str.strip()
    expanded = expanded[expanded["district"] != ""]

    # Count each district's participation in repeated groups.
    district_summary = (
        expanded.groupby(
            [
                "crop_name",
                "crop_code",
                "state_name",
                "district",
            ],
            dropna=False,
        )
        .agg(
            repeated_group_count=("year", "size"),
            distinct_years=("year", "nunique"),
            first_year=("year", "min"),
            last_year=("year", "max"),
        )
        .reset_index()
        .sort_values(
            ["repeated_group_count", "distinct_years"],
            ascending=False,
        )
    )

    # Summarize the number of distinct districts involved per crop/state.
    state_summary = (
        expanded.groupby(
            ["crop_name", "crop_code", "state_name"],
            dropna=False,
        )
        .agg(
            affected_districts=("district", "nunique"),
            district_participations=("district", "size"),
            distinct_years=("year", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["affected_districts", "district_participations"],
            ascending=False,
        )
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    district_path = REPORT_DIR / "cross_district_persistence_by_district.csv"
    state_path = REPORT_DIR / "cross_district_persistence_by_state.csv"

    district_summary.to_csv(district_path, index=False)
    state_summary.to_csv(state_path, index=False)

    print("\nDISTRICT PERSISTENCE SUMMARY")
    print("=" * 100)
    print(district_summary.to_string(index=False))

    print("\nCROP-STATE SUMMARY")
    print("=" * 100)
    print(state_summary.to_string(index=False))

    print("\nReports saved:")
    print(district_path.relative_to(ROOT))
    print(state_path.relative_to(ROOT))


if __name__ == "__main__":
    analyze_persistence()
