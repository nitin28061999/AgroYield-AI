
"""Audit chronological train, validation and test coverage."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT / "data" / "processed"
    / "crop_yield_modeling_candidates.csv"
)


def main():
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    # Extract the starting year from labels such as 1997-1998.
    df["start_year"] = (
        df["year"].str.extract(r"^(\d{4})")[0].astype(int)
    )

    # Treat missing crop codes as a distinct identity value.
    df["crop_code_key"] = (
        df["crop_code"].astype("Int64").astype("string")
        .fillna("MISSING")
    )

    df["crop_identity"] = (
        df["crop_code_key"] + "|" + df["crop_name"]
    )

    df["district_identity"] = (
        df["state_code"].astype(str)
        + "|"
        + df["district_code"].astype(str)
    )

    conditions = [
        df["start_year"].between(1997, 2017),
        df["start_year"].between(2018, 2019),
        df["start_year"].between(2020, 2022),
    ]

    labels = ["train", "validation", "test"]

    df["split"] = "outside_split"
    for condition, label in zip(conditions, labels):
        df.loc[condition, "split"] = label

    if (df["split"] == "outside_split").any():
        raise ValueError("Some records fall outside the defined split.")

    print("\nCHRONOLOGICAL SPLIT SUMMARY")
    print("=" * 75)

    summary = (
        df.groupby("split")
        .agg(
            records=("id", "size"),
            first_year=("start_year", "min"),
            last_year=("start_year", "max"),
            crop_identities=("crop_identity", "nunique"),
            states=("state_name", "nunique"),
            districts=("district_identity", "nunique"),
        )
        .reindex(labels)
    )

    print(summary.to_string())

    train = df[df["split"] == "train"]
    train_crops = set(train["crop_identity"])
    train_districts = set(train["district_identity"])

    print("\nUNSEEN CATEGORY CHECK")
    print("=" * 75)

    for label in ["validation", "test"]:
        part = df[df["split"] == label]

        unseen_crops = set(part["crop_identity"]) - train_crops
        unseen_districts = (
            set(part["district_identity"]) - train_districts
        )

        print(f"\n{label.upper()}")
        print("Unseen crop identities:", len(unseen_crops))
        print("Unseen district identities:", len(unseen_districts))

        if unseen_crops:
            print("Unseen crops:", sorted(unseen_crops)[:20])

    print("\nRECORD COUNTS BY YEAR AND SPLIT")
    print("=" * 75)

    yearly = (
        df.groupby(["year", "split"])
        .size()
        .unstack(fill_value=0)
    )

    print(yearly.to_string())


if __name__ == "__main__":
    main()
