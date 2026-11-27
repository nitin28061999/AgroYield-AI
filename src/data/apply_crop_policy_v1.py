from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "data" / "raw" / "crop-wise-area-production-yield.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

POLICY_VERSION = "1.0"

EXPECTED_RAW_ROWS = 455_359
EXPECTED_SEASONAL_ROWS = 407_315
EXPECTED_TARGET_ELIGIBLE_ROWS = 398_343
EXPECTED_COCONUT_ROWS = 3_261
EXPECTED_SUGARCANE_EXCLUSIONS = 3
EXPECTED_FINAL_ROWS = 395_079


def normalize_crop_code(series):
    """Normalize numeric crop codes, including values such as '1006.0'."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.astype("Int64").astype("string")


def normalize_start_year(series):
    """Extract the first four-digit year from labels such as 1997-1998."""
    return (
        series.astype("string")
        .str.extract(r"(\d{4})", expand=False)
        .pipe(pd.to_numeric, errors="coerce")
        .astype("Int64")
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH, low_memory=False)

    required_columns = {
        "id",
        "year",
        "state_name",
        "district_name",
        "crop_code",
        "crop_name",
        "season",
        "area",
        "production",
        "yield",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if len(df) != EXPECTED_RAW_ROWS:
        raise ValueError(
            f"Raw row-count mismatch: expected {EXPECTED_RAW_ROWS:,}, "
            f"found {len(df):,}"
        )

    df["crop_code_normalized"] = normalize_crop_code(df["crop_code"])
    df["start_year"] = normalize_start_year(df["year"])

    df["production_numeric"] = pd.to_numeric(
        df["production"], errors="coerce"
    )
    df["yield_numeric"] = pd.to_numeric(df["yield"], errors="coerce")

    # Every row receives exactly one primary disposition.
    df["policy_version"] = POLICY_VERSION
    df["exclusion_reason"] = pd.NA

    # Apply broad data-quality rules first.
    df.loc[
        df["season"].astype("string").str.strip().str.casefold().eq("total"),
        "exclusion_reason",
    ] = "total_season_overlap"

    df.loc[
        df["exclusion_reason"].isna()
        & df["production_numeric"].isna(),
        "exclusion_reason",
    ] = "missing_production"

    df.loc[
        df["exclusion_reason"].isna()
        & df["production_numeric"].le(0),
        "exclusion_reason",
    ] = "zero_or_nonpositive_production"

    df.loc[
        df["exclusion_reason"].isna()
        & (
            df["yield_numeric"].isna()
            | df["yield_numeric"].le(0)
        ),
        "exclusion_reason",
    ] = "missing_or_nonpositive_reported_yield"

    # Crop-level exclusion: Coconut is not pooled with tonnes/hectare crops.
    coconut_mask = df["crop_code_normalized"].eq("1006")

    df.loc[
        df["exclusion_reason"].isna() & coconut_mask,
        "exclusion_reason",
    ] = "excluded_coconut_unit_uncertainty"

    # Record-specific Sugarcane exclusions.
    sugarcane_mask = df["crop_code_normalized"].eq("401")
    sugarcane_year = df["start_year"]

    sugarcane_cases = (
        sugarcane_mask
        & (
            (
                df["state_name"].astype("string").str.strip().eq("Tamil Nadu")
                & df["district_name"].astype("string").str.strip().eq("The Nilgiris")
                & sugarcane_year.eq(1997)
            )
            | (
                df["state_name"].astype("string").str.strip().eq("Tamil Nadu")
                & df["district_name"].astype("string").str.strip().eq("Thoothukkudi")
                & sugarcane_year.eq(1997)
            )
            | (
                df["state_name"].astype("string").str.strip().eq("Chhattisgarh")
                & df["district_name"].astype("string").str.strip().eq("Narayanpur")
                & sugarcane_year.eq(2021)
            )
        )
    )

    df.loc[
        df["exclusion_reason"].isna() & sugarcane_cases,
        "exclusion_reason",
    ] = "excluded_sugarcane_record_pending_source_verification"

    # All remaining rows are eligible under policy v1.0.
    df["policy_eligible"] = df["exclusion_reason"].isna()
    df.loc[df["policy_eligible"], "exclusion_reason"] = "included"

    # Reconciliation checks.
    seasonal_rows = ~(
        df["season"].astype("string").str.strip().str.casefold().eq("total")
    )
    target_eligible = (
        seasonal_rows
        & df["production_numeric"].gt(0)
        & df["yield_numeric"].gt(0)
    )

    checks = {
        "raw_rows": len(df),
        "seasonal_rows": int(seasonal_rows.sum()),
        "target_eligible_before_crop_policy": int(target_eligible.sum()),
        "coconut_exclusions": int(
            df["exclusion_reason"].eq(
                "excluded_coconut_unit_uncertainty"
            ).sum()
        ),
        "sugarcane_record_exclusions": int(
            df["exclusion_reason"].eq(
                "excluded_sugarcane_record_pending_source_verification"
            ).sum()
        ),
        "final_eligible_rows": int(df["policy_eligible"].sum()),
    }

    expected = {
        "raw_rows": EXPECTED_RAW_ROWS,
        "seasonal_rows": EXPECTED_SEASONAL_ROWS,
        "target_eligible_before_crop_policy": EXPECTED_TARGET_ELIGIBLE_ROWS,
        "coconut_exclusions": EXPECTED_COCONUT_ROWS,
        "sugarcane_record_exclusions": EXPECTED_SUGARCANE_EXCLUSIONS,
        "final_eligible_rows": EXPECTED_FINAL_ROWS,
    }

    print("\n" + "=" * 72)
    print(f"AGROYIELD AI — CROP POLICY v{POLICY_VERSION}")
    print("=" * 72)

    for key, actual in checks.items():
        expected_value = expected[key]
        status = "PASS" if actual == expected_value else "MISMATCH"
        print(
            f"{key:42s} {actual:>10,} "
            f"(expected {expected_value:,})  {status}"
        )

    mismatches = {
        key: (checks[key], expected[key])
        for key in expected
        if checks[key] != expected[key]
    }
    if mismatches:
        raise ValueError(
            "Policy v1.0 count reconciliation failed. "
            f"Investigate before training: {mismatches}"
        )

    # Save complete row-level audit and eligible training data.
    audit_path = REPORT_DIR / "crop_policy_v1_audit.csv"
    training_path = OUTPUT_DIR / "crop_yield_training_v1.csv"
    summary_path = REPORT_DIR / "crop_policy_v1_summary.csv"

    df.to_csv(audit_path, index=False)

    training_df = df.loc[df["policy_eligible"]].copy()

    # Keep source columns and policy fields for traceability.
    training_df.to_csv(training_path, index=False)

    summary = (
        df.groupby("exclusion_reason", dropna=False)
        .size()
        .rename("records")
        .reset_index()
        .sort_values("records", ascending=False)
    )
    summary.insert(0, "policy_version", POLICY_VERSION)
    summary.to_csv(summary_path, index=False)

    print("\nDisposition summary:")
    print(summary.to_string(index=False))

    print("\nFiles created:")
    print(f"  Training data : {training_path}")
    print(f"  Full audit    : {audit_path}")
    print(f"  Summary       : {summary_path}")

    print("\nPolicy v1.0 filtering completed successfully.")


if __name__ == "__main__":
    main()
