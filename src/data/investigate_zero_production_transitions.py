
"""Check whether 2002-04 zero-production crop groups recover in adjacent years."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "data" / "raw" / "crop-wise-area-production-yield.csv"
OUTPUT_PATH = ROOT / "reports" / "zero_production_transition_audit.csv"


def main():  # sourcery skip: hoist-statement-from-if
    df = pd.read_csv(RAW_PATH, low_memory=False)
    df = df[df["season"] != "Total"].copy()
    df = df[df["production"].notna()].copy()

    # Keep crop name AND code to avoid merging distinct crop identities.
    keys = ["crop_name", "crop_code", "year"]

    summary = (
        df.assign(is_zero=df["production"].eq(0))
        .groupby(keys, dropna=False)
        .agg(
            total_records=("id", "size"),
            zero_records=("is_zero", "sum"),
            districts=("district_name", "nunique"),
            states=("state_name", "nunique"),
        )
        .reset_index()
    )

    summary["zero_rate_pct"] = (
        summary["zero_records"] / summary["total_records"] * 100
    ).round(2)

    # Derive chronological order from the first year in each label.
    years = sorted(
        df["year"].dropna().unique(),
        key=lambda value: int(str(value)[:4]),
    )
    year_positions = {year: i for i, year in enumerate(years)}

    suspicious = summary[
        summary["year"].isin(["2002-2003", "2003-2004"])
        & (summary["total_records"] >= 10)
        & (summary["zero_rate_pct"] >= 80)
    ].copy()

    rows = []

    for _, group in suspicious.iterrows():
        position = year_positions[group["year"]]

        record = group.to_dict()

        for label, offset in [
            ("previous", -1),
            ("next", 1),
        ]:
            adjacent_position = position + offset

            if 0 <= adjacent_position < len(years):
                adjacent_year = years[adjacent_position]

                match = summary[
                    (summary["crop_name"] == group["crop_name"])
                    & (summary["crop_code"] == group["crop_code"])
                    & (summary["year"] == adjacent_year)
                ]

                if not match.empty:
                    adjacent = match.iloc[0]
                    record[f"{label}_year"] = adjacent_year
                    record[f"{label}_records"] = adjacent["total_records"]
                    record[f"{label}_zero_records"] = adjacent["zero_records"]
                    record[f"{label}_zero_rate_pct"] = adjacent["zero_rate_pct"]
                else:
                    _extracted_from_main_67(adjacent_year, record, label)
            else:
                _extracted_from_main_67(None, record, label)
        rows.append(record)

    result = pd.DataFrame(rows)

    columns = [
        "crop_name",
        "crop_code",
        "year",
        "total_records",
        "zero_records",
        "zero_rate_pct",
        "previous_year",
        "previous_records",
        "previous_zero_rate_pct",
        "next_year",
        "next_records",
        "next_zero_rate_pct",
        "districts",
        "states",
    ]

    result = result[columns].sort_values(
        ["zero_rate_pct", "total_records"],
        ascending=False,
    )

    print("\nZERO-PRODUCTION TRANSITION AUDIT")
    print("=" * 90)
    print("Suspicious crop-year groups:", len(result))
    print("\n")
    print(result.to_string(index=False))

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nReport saved: {OUTPUT_PATH.relative_to(ROOT)}")


# TODO Rename this here and in `main`
def _extracted_from_main_67(arg0, record, label):
    record[f"{label}_year"] = arg0
    record[f"{label}_records"] = 0
    record[f"{label}_zero_records"] = 0
    record[f"{label}_zero_rate_pct"] = None


if __name__ == "__main__":
    main()
