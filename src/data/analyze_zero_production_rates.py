
"""Compare zero-production rates across years, crops, and states."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "data" / "raw" / "crop-wise-area-production-yield.csv"

YEAR_OUTPUT = ROOT / "reports" / "zero_production_rates_by_year.csv"
CROP_YEAR_OUTPUT = ROOT / "reports" / "zero_production_rates_by_crop_year.csv"
STATE_YEAR_OUTPUT = ROOT / "reports" / "zero_production_rates_by_state_year.csv"


def summarize_rates(df, keys):
    result = (
        df.assign(is_zero=df["production"].eq(0))
        .groupby(keys, dropna=False)
        .agg(
            total_records=("id", "size"),
            zero_records=("is_zero", "sum"),
            districts=("district_name", "nunique"),
        )
        .reset_index()
    )

    result["zero_rate_pct"] = (
        result["zero_records"] / result["total_records"] * 100
    ).round(2)

    return result


def main():  # sourcery skip: extract-duplicate-method
    df = pd.read_csv(RAW_PATH, low_memory=False)
    df = df[df["season"] != "Total"].copy()

    # Restrict to records with a reported production value.
    # Missing production is not the same as zero production.
    df = df[df["production"].notna()].copy()

    print("\nZERO PRODUCTION RATE ANALYSIS")
    print("=" * 75)

    year_rates = summarize_rates(df, ["year"])
    year_rates = year_rates.sort_values("year")

    print("\nYEAR-WISE ZERO PRODUCTION RATES")
    print(year_rates.to_string(index=False))
    year_rates.to_csv(YEAR_OUTPUT, index=False)

    crop_year_rates = summarize_rates(
        df,
        ["crop_name", "crop_code", "year"],
    )

    suspicious_crop_years = crop_year_rates[
        crop_year_rates["zero_records"] >= 10
    ].sort_values(
        ["zero_rate_pct", "zero_records"],
        ascending=False,
    )

    print("\nCROP-YEAR GROUPS WITH AT LEAST 10 ZERO RECORDS")
    print(suspicious_crop_years.head(35).to_string(index=False))
    suspicious_crop_years.to_csv(CROP_YEAR_OUTPUT, index=False)

    state_year_rates = summarize_rates(
        df,
        ["state_name", "year"],
    )

    suspicious_state_years = state_year_rates[
        state_year_rates["year"].isin(
            ["2002-2003", "2003-2004", "2020-2021", "2021-2022", "2022-2023"]
        )
    ].sort_values(
        ["year", "zero_rate_pct"],
        ascending=[True, False],
    )

    print("\nSTATE-YEAR ZERO PRODUCTION RATES")
    print(suspicious_state_years.to_string(index=False))
    suspicious_state_years.to_csv(STATE_YEAR_OUTPUT, index=False)

    print("\nReports saved:")
    print(YEAR_OUTPUT.relative_to(ROOT))
    print(CROP_YEAR_OUTPUT.relative_to(ROOT))
    print(STATE_YEAR_OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
