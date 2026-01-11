import argparse
from pathlib import Path
import pandas as pd

from src.config import RAW_ATHLETES_CSV, PROCESSED_DIR, PROCESSED_CSV


def build_country_year_table(df: pd.DataFrame) -> pd.DataFrame:
    # Keep Summer Olympics (simpler and most common in medal prediction projects)
    df = df[df["Season"] == "Summer"].copy()

    # Medal flag
    df["medal_flag"] = df["Medal"].notna().astype(int)

    # Athletes per country-year (unique athlete IDs)
    athletes = (
        df.groupby(["Year", "NOC"])["ID"]
        .nunique()
        .reset_index(name="athletes")
    )

    # Medals per country-year (count medal rows)
    medals = (
        df[df["medal_flag"] == 1]
        .groupby(["Year", "NOC"])["medal_flag"]
        .sum()
        .reset_index(name="medals")
    )

    # Merge and fill missing medal counts with 0
    out = athletes.merge(medals, on=["Year", "NOC"], how="left")
    out["medals"] = out["medals"].fillna(0).astype(int)

    # Previous medals feature (previous Olympics entry for that country)
    out = out.sort_values(["NOC", "Year"]).reset_index(drop=True)
    out["prev_medals"] = out.groupby("NOC")["medals"].shift(1).fillna(0).astype(int)

    # Keep only what we need
    out = out[["Year", "NOC", "athletes", "prev_medals", "medals"]]

    return out


def main(raw_path: Path, output_path: Path) -> None:
    df = pd.read_csv(raw_path)

    required = {"ID", "Year", "Season", "NOC", "Medal"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    clean = build_country_year_table(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_path, index=False)

    print(f"Saved clean dataset -> {output_path}")
    print(f"Rows: {len(clean):,} | Years: {clean['Year'].min()} to {clean['Year'].max()}")
    print(clean.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build clean country-year dataset for medal prediction.")
    parser.add_argument("--raw", type=str, default=str(RAW_ATHLETES_CSV), help="Path to athlete_events.csv")
    parser.add_argument("--out", type=str, default=str(PROCESSED_CSV), help="Output path for clean.csv")
    args = parser.parse_args()

    main(Path(args.raw), Path(args.out))
