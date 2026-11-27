
"""Check feature quality before building the ML pipeline."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT / "data" / "processed"
    / "crop_yield_modeling_candidates.csv"
)


def main():
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    print("\nFEATURE READINESS AUDIT")
    print("=" * 75)

    features = [
        "year",
        "state_name",
        "state_code",
        "district_name",
        "district_code",
        "crop_name",
        "crop_code",
        "crop_type",
        "season",
        "area",
    ]

    print("\n1. MISSING VALUES")
    print("-" * 75)
    print(df[features].isna().sum().to_string())

    print("\n2. CATEGORY COUNTS")
    print("-" * 75)

    for col in [
        "state_name",
        "district_name",
        "crop_name",
        "crop_type",
        "season",
    ]:
        print(f"{col}: {df[col].nunique(dropna=True)}")

    print("\n3. CROP CODE COVERAGE")
    print("-" * 75)
    print("Missing crop codes:", int(df["crop_code"].isna().sum()))
    print(
        "Distinct crop identities:",
        df[["crop_code", "crop_name"]].drop_duplicates().shape[0],
    )

    print("\n4. AREA DISTRIBUTION")
    print("-" * 75)
    print(
        df["area"].describe(
            percentiles=[0.01, 0.05, 0.50, 0.95, 0.99]
        ).to_string()
    )

    print("\n5. DISTRICT CODE CONSISTENCY")
    print("-" * 75)

    district_mapping = (
        df.groupby("district_code")
        .agg(
            district_names=("district_name", "nunique"),
            states=("state_name", "nunique"),
        )
        .sort_values("states", ascending=False)
    )

    print(
        "District codes associated with multiple states:",
        int((district_mapping["states"] > 1).sum()),
    )
    print(district_mapping.head(15).to_string())

    print("\n6. CROP-YEAR COVERAGE")
    print("-" * 75)

    coverage = (
        df.groupby(["crop_code", "crop_name"])["year"]
        .nunique()
        .describe()
    )
    print(coverage.to_string())

    print("\n7. DUPLICATE GEOGRAPHIC CROP-SEASON-YEAR KEYS")
    print("-" * 75)

    keys = [
        "state_name",
        "district_name",
        "crop_code",
        "crop_name",
        "season",
        "year",
    ]

    duplicate_count = int(df.duplicated(subset=keys).sum())
    print("Repeated geographic crop-season-year keys:", duplicate_count)


if __name__ == "__main__":
    main()
