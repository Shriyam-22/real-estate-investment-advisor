"""
Step 1 — Data Preprocessing & Feature Engineering
Input : data/india_housing_prices.csv
Output: data/processed_housing.csv, models/city_median_ppsf.json

Run:  python src/preprocess.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from features import add_features, add_targets

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "india_housing_prices.csv"
OUT = ROOT / "data" / "processed_housing.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "ID"])
    print(f"Duplicates removed: {n0 - len(df)}")

    # Missing values: median for numeric, mode for categorical
    num_cols = df.select_dtypes("number").columns
    for c in df.columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median() if c in num_cols else df[c].mode()[0])
    print(f"Missing values remaining: {int(df.isna().sum().sum())}")

    # Floor_No can't exceed Total_Floors -> cap it
    bad = (df["Floor_No"] > df["Total_Floors"]).sum()
    df["Floor_No"] = np.minimum(df["Floor_No"], df["Total_Floors"])
    print(f"Floor_No > Total_Floors fixed: {bad}")
    return df


def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    # IQR on price/sqft — flagged, not dropped, so any user input can still be scored
    q1, q3 = df["Price_per_SqFt"].quantile([0.25, 0.75])
    iqr = q3 - q1
    df["PPSF_Outlier"] = ((df["Price_per_SqFt"] < q1 - 1.5 * iqr) |
                          (df["Price_per_SqFt"] > q3 + 1.5 * iqr)).astype(int)
    print(f"Price/SqFt IQR outliers flagged: {df['PPSF_Outlier'].sum()}")
    return df


def main():
    df = pd.read_csv(RAW)
    print("Raw shape:", df.shape)
    df = clean(df)
    df = add_features(df)               # Price_per_SqFt recomputed in ₹ (raw col was lakhs/sqft)
    df = flag_outliers(df)
    df = add_targets(df)
    print("Good_Investment balance:", df["Good_Investment"].value_counts(normalize=True).round(3).to_dict())

    city_med = df.groupby("City")["Price_per_SqFt"].median().round(2).to_dict()
    (MODELS / "city_median_ppsf.json").write_text(json.dumps(city_med, indent=1))
    df.to_csv(OUT, index=False)
    print("Processed shape:", df.shape, "->", OUT)


if __name__ == "__main__":
    main()
