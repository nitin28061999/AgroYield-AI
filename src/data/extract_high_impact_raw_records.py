
"""Extract original records belonging to high-impact repeated groups."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "data" / "raw" / "crop-wise-area-production-yield.csv"
GROUPS_PATH = ROOT / "reports" / "high_impact_cross_district.csv"
OUTPUT_PATH = ROOT / "reports" / "high_impact_original_records.csv"

KEYS = [
    "year",
    "state_name",
    "crop_name",
    "crop_code",
    "season",
    "area",
    "production",
]


def extract_original_records():
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    groups = pd.read_csv(GROUPS_PATH, low_memory=False)

    # Only include the high-impact groups.
    groups = groups[groups["district_count"] >= 5].copy()

    # Assign a stable identifier to each suspicious group.
    groups = groups.reset_index(drop=True)
    groups["audit_group_id"] = [
        f"HDG_{i:03d}" for i in range(1, len(groups) + 1)
    ]

    # Match original records to the repeated groups.
    matched = raw.merge(
        groups[KEYS + ["audit_group_id", "district_count"]],
        on=KEYS,
        how="inner",
        validate="many_to_one",
    )

    columns = [
        "audit_group_id",
        "id",
        "year",
        "state_name",
        "state_code",
        "district_name",
        "district_code",
        "crop_name",
        "crop_code",
        "season",
        "area",
        "area_unit",
        "production",
        "production_unit",
        "yield",
        "yield_unit",
        "district_count",
    ]

    matched = matched[columns].sort_values(
        ["audit_group_id", "district_name"]
    )

    matched.to_csv(OUTPUT_PATH, index=False)

    print("\nORIGINAL RECORD EXTRACTION")
    print("=" * 75)
    print("High-impact groups:", len(groups))
    print("Original records matched:", len(matched))
    print(
        "Distinct original record IDs:",
        matched["id"].nunique(),
    )
    print(
        "Distinct districts:",
        matched["district_name"].nunique(),
    )
    print(
        "Groups with missing production:",
        matched.groupby("audit_group_id")["production"]
        .first()
        .isna()
        .sum(),
    )

    print("\nRECORDS PER AUDIT GROUP")
    print("=" * 75)
    print(
        matched.groupby("audit_group_id")
        .agg(
            records=("id", "size"),
            districts=("district_name", "nunique"),
        )
        .to_string()
    )

    print(f"\nReport saved: {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    extract_original_records()
