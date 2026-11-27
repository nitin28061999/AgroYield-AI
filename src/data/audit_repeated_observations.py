
"""Quantify repeated crop observations across different years."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "data" / "raw"
    / "crop-wise-area-production-yield.csv"
)

REPORT_DIR = ROOT / "reports"


def audit_repeated_observations():
    df = pd.read_csv(DATA_PATH, low_memory=False)

    df = df[df["season"] != "Total"].copy()

    keys = [
        "state_name",
        "district_name",
        "crop_name",
        "season",
    ]

    print("=" * 70)
    print("AGROYIELD AI - REPEATED OBSERVATION AUDIT")
    print("=" * 70)

    # Group identical area and production observations across years.
    grouped = (
        df.groupby(
            keys + ["area", "production"],
            dropna=False,
        )
        .agg(
            occurrences=("year", "size"),
            years=("year", lambda x: ", ".join(sorted(x.unique()))),
        )
        .reset_index()
    )

    repeated = grouped[grouped["occurrences"] >= 3].copy()

    non_missing = repeated[repeated["production"].notna()]
    missing = repeated[repeated["production"].isna()]

    print("\n1. REPEATED VALUE SUMMARY")

    print(f"All repeated groups: {len(repeated):,}")
    print(f"Repeated non-missing production groups: {len(non_missing):,}")
    print(f"Repeated missing production groups: {len(missing):,}")

    print(
        "Records belonging to repeated non-missing groups:",
        f"{non_missing['occurrences'].sum():,}",
    )

    print("\n2. MOST FREQUENT NON-MISSING REPEATED VALUES")

    print(
        non_missing.sort_values(
            "occurrences", ascending=False
        ).head(20).to_string(index=False)
    )

    # Identify repeated values with very long year spans.
    repeated["year_count"] = repeated["years"].str.count(",") + 1

    print("\n3. LONGEST REPEATED YEAR SPANS")

    print(
        repeated.sort_values(
            "year_count", ascending=False
        ).head(20).to_string(index=False)
    )

    # Investigate suspicious small-area observations.
    print("\n4. SMALL-AREA RECORD SUMMARY")

    small = df[df["area"] <= 10]

    print(f"Records with area <= 10 hectares: {len(small):,}")
    print(f"Percentage of dataset: {len(small) / len(df) * 100:.2f}%")

    print("\nSmall-area records by crop (top 15):")

    print(
        small["crop_name"]
        .value_counts()
        .head(15)
        .to_string()
    )

    # Save detailed reports.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    repeated.to_csv(
        REPORT_DIR / "repeated_observation_groups.csv",
        index=False,
    )

    non_missing.to_csv(
        REPORT_DIR / "repeated_nonmissing_groups.csv",
        index=False,
    )

    print("\nReports saved:")
    print("reports/repeated_observation_groups.csv")
    print("reports/repeated_nonmissing_groups.csv")

    print("\nAUDIT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    audit_repeated_observations()
