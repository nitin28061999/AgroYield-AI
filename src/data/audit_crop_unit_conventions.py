
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "data"
    / "processed"
    / "crop_yield_modeling_candidates.csv"
)

OUTPUT = ROOT / "reports" / "crop_unit_convention_audit.csv"

df = pd.read_csv(
    INPUT,
    dtype={"crop_code": "string"},
)

# Keep only valid crop identities.
df["crop_code"] = df["crop_code"].fillna("MISSING")
df["crop_identity"] = df["crop_code"] + " | " + df["crop_name"]

# Summarize each crop's reported units and yield distribution.
summary = (
    df.groupby(
        ["crop_identity", "crop_name", "crop_code"],
        dropna=False,
    )
    .agg(
        records=("id", "size"),
        area_units=("area_unit", "nunique"),
        production_units=("production_unit", "nunique"),
        yield_units=("yield_unit", "nunique"),
        area_unit_values=("area_unit", lambda x: " | ".join(sorted(x.dropna().unique()))),
        production_unit_values=("production_unit", lambda x: " | ".join(sorted(x.dropna().unique()))),
        yield_unit_values=("yield_unit", lambda x: " | ".join(sorted(x.dropna().unique()))),
        median_yield=("yield", "median"),
        p95_yield=("yield", lambda x: x.quantile(0.95)),
        p99_yield=("yield", lambda x: x.quantile(0.99)),
        maximum_yield=("yield", "max"),
        median_production=("production", "median"),
    )
    .reset_index()
)

summary["unit_label_inconsistency"] = (
    (summary["area_units"] > 1)
    | (summary["production_units"] > 1)
    | (summary["yield_units"] > 1)
)

summary = summary.sort_values(
    "maximum_yield",
    ascending=False,
)

summary.to_csv(OUTPUT, index=False)

print("\nCROP UNIT CONVENTION AUDIT")
print("=" * 100)

print("Crop identities:", len(summary))
print(
    "Crop identities with inconsistent unit labels:",
    summary["unit_label_inconsistency"].sum(),
)

print("\nCROPS WITH INCONSISTENT UNIT LABELS")
print("=" * 100)

inconsistent = summary[summary["unit_label_inconsistency"]]

if inconsistent.empty:
    print("No inconsistent unit labels found.")
else:
    print(
        inconsistent[
            [
                "crop_identity",
                "records",
                "area_unit_values",
                "production_unit_values",
                "yield_unit_values",
            ]
        ].to_string(index=False)
    )

print("\nTOP 25 CROPS BY MAXIMUM REPORTED YIELD")
print("=" * 100)

print(
    summary[
        [
            "crop_identity",
            "records",
            "median_yield",
            "p95_yield",
            "p99_yield",
            "maximum_yield",
            "production_unit_values",
            "yield_unit_values",
        ]
    ].head(25).round(3).to_string(index=False)
)

print(f"\nSaved: {OUTPUT}")
