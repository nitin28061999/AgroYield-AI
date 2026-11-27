"""Inspect original district-level records in high-impact groups."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "reports" / "high_impact_original_records.csv"


def inspect_records():
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    columns = [
        "audit_group_id",
        "id",
        "year",
        "state_name",
        "district_name",
        "district_code",
        "crop_name",
        "crop_code",
        "season",
        "area",
        "production",
        "yield",
    ]

    df = df[columns].sort_values(
        ["audit_group_id", "district_name"]
    )

    print("\nHIGH-IMPACT DISTRICT-LEVEL RECORD INSPECTION")
    print("=" * 100)

    for group_id, group in df.groupby("audit_group_id", sort=True):
        first = group.iloc[0]

        print(
            f"\n{group_id} | {first['year']} | "
            f"{first['state_name']} | {first['crop_name']} | "
            f"{first['season']}"
        )
        print("-" * 100)

        print(
            group[
                [
                    "id",
                    "district_name",
                    "district_code",
                    "area",
                    "production",
                    "yield",
                ]
            ].to_string(index=False)
        )

    print("\nTotal records inspected:", len(df))
    print("Distinct record IDs:", df["id"].nunique())


if __name__ == "__main__":
    inspect_records()
