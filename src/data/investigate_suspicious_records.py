
"""Investigate suspicious crop production and yield records."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "data" / "raw"
    / "crop-wise-area-production-yield.csv"
)

REPORT_DIR = ROOT / "reports"


def investigate():
    df = pd.read_csv(DATA_PATH, low_memory=False)

    for column in ["area", "production", "yield"]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    print("=" * 75)
    print("AGROYIELD AI - SUSPICIOUS RECORD INVESTIGATION")
    print("=" * 75)

    # 1. Small-area records with unusually high yields.
    print("\n1. HIGH YIELD RECORDS WITH AREA <= 10 HECTARES")

    small_area = df[
        (df["area"] <= 10)
        & (df["yield"] > 0)
        & (df["season"] != "Total")
    ].copy()

    columns = [
        "year",
        "state_name",
        "district_name",
        "crop_name",
        "season",
        "area",
        "production",
        "yield",
    ]

    top_small_area = small_area.nlargest(30, "yield")

    print(f"Records with area <= 10 hectares: {len(small_area):,}")
    print("\nTop 30 yields:")
    print(top_small_area[columns].to_string(index=False))

    # 2. Check repeated production and area values across years.
    print("\n2. REPEATED AREA AND PRODUCTION VALUES")

    group_columns = [
        "state_name",
        "district_name",
        "crop_name",
        "season",
        "area",
        "production",
    ]

    repeated = (
        df[df["season"] != "Total"]
        .groupby(group_columns, dropna=False)
        .agg(
            occurrences=("year", "size"),
            years=("year", lambda x: ", ".join(sorted(x.astype(str).unique()))),
        )
        .reset_index()
    )

    repeated = repeated[repeated["occurrences"] >= 3]

    repeated = repeated.sort_values(
        "occurrences", ascending=False
    )

    print(f"Groups repeated across at least 3 records: {len(repeated):,}")

    print("\nTop 25 repeated groups:")
    print(repeated.head(25).to_string(index=False))

    # 3. Focused Coconut investigation.
    print("\n3. COCONUT - ASSAM - TINSUKIA")

    coconut = df[
        (df["crop_name"] == "Coconut")
        & (df["state_name"] == "Assam")
        & (df["district_name"] == "Tinsukia")
    ].sort_values("year")

    print(
        coconut[columns].to_string(index=False)
    )

    # 4. Save findings.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    top_small_area[columns].to_csv(
        REPORT_DIR / "small_area_high_yield_records.csv",
        index=False,
    )

    repeated.to_csv(
        REPORT_DIR / "repeated_area_production_records.csv",
        index=False,
    )

    print("\nReports saved:")
    print("reports/small_area_high_yield_records.csv")
    print("reports/repeated_area_production_records.csv")

    print("\nINVESTIGATION COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    investigate()
