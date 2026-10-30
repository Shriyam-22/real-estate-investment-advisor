"""
Shared feature engineering + target rules.
Used by preprocess.py (training data) AND app.py (single-property inference),
so the model always sees features built exactly the same way.
"""
import pandas as pd

HORIZON_YEARS = 5
BASE_GROWTH = 0.08
AMENITY_LIST = ["Gym", "Pool", "Clubhouse", "Garden", "Playground"]

CAT_FEATURES = ["City", "Property_Type", "Furnished_Status", "Facing", "Owner_Type",
                "Public_Transport_Accessibility", "Availability_Status"]
NUM_FEATURES = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Age_of_Property",
                "Floor_No", "Total_Floors", "Nearby_Schools", "Nearby_Hospitals",
                "Parking_Flag", "Security_Flag", "Ready_Flag", "Transport_Score",
                "Amenity_Count", *[f"Has_{a}" for a in AMENITY_LIST],
                "School_Density_Score", "Hospital_Density_Score", "Infrastructure_Score",
                "Floor_Ratio", "SqFt_per_BHK", "City_Median_PPSF", "PPSF_vs_City"]
MODEL_FEATURES = CAT_FEATURES + NUM_FEATURES


def add_features(df: pd.DataFrame, city_median_ppsf: dict | None = None) -> pd.DataFrame:
    """Engineer features. Pass city_median_ppsf at inference time (saved from training)."""
    df = df.copy()
    df["Price_per_SqFt"] = (df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]).round(2)

    for a in AMENITY_LIST:
        df[f"Has_{a}"] = df["Amenities"].str.contains(a, case=False).astype(int)
    df["Amenity_Count"] = df[[f"Has_{a}" for a in AMENITY_LIST]].sum(axis=1)

    df["Parking_Flag"] = (df["Parking_Space"] == "Yes").astype(int)
    df["Security_Flag"] = (df["Security"] == "Yes").astype(int)
    df["Ready_Flag"] = (df["Availability_Status"] == "Ready_to_Move").astype(int)
    df["Transport_Score"] = df["Public_Transport_Accessibility"].map({"Low": 1, "Medium": 2, "High": 3})

    df["School_Density_Score"] = df["Nearby_Schools"] / 10
    df["Hospital_Density_Score"] = df["Nearby_Hospitals"] / 10
    df["Infrastructure_Score"] = (0.35 * df["School_Density_Score"]
                                  + 0.35 * df["Hospital_Density_Score"]
                                  + 0.30 * (df["Transport_Score"] - 1) / 2).round(3)

    df["Floor_Ratio"] = (df["Floor_No"] / df["Total_Floors"]).round(3)
    df["SqFt_per_BHK"] = (df["Size_in_SqFt"] / df["BHK"]).round(1)

    if city_median_ppsf is None:
        city_med = df.groupby("City")["Price_per_SqFt"].transform("median")
    else:
        overall = pd.Series(city_median_ppsf).median()
        city_med = df["City"].map(city_median_ppsf).fillna(overall)
    df["City_Median_PPSF"] = city_med.round(2)
    df["PPSF_vs_City"] = (df["Price_per_SqFt"] / city_med).round(4)
    return df


def growth_rate(df: pd.DataFrame) -> pd.Series:
    """Feature-based annual appreciation: 8% base, adjusted by investor drivers, clipped 3–14%."""
    r = pd.Series(BASE_GROWTH, index=df.index)
    r += 0.010 * (df["Transport_Score"] - 2)              # connectivity ±1%
    r += 0.020 * (df["Infrastructure_Score"] - 0.5)       # schools / hospitals
    r += 0.010 * (1 - df["Ready_Flag"])                   # under-construction premium
    r -= 0.0006 * (df["Age_of_Property"] - 15)            # depreciation with age
    r += 0.003 * (df["Amenity_Count"] - 3)                # amenities
    r += 0.005 * df["Security_Flag"]
    r += 0.015 * (1 - df["PPSF_vs_City"]).clip(-1, 1)     # under-priced vs city catches up
    r += df["Property_Type"].map({"Villa": 0.005, "Independent House": 0.003, "Apartment": 0.0}).fillna(0)
    return r.clip(0.03, 0.14).round(4)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Annual_Growth_Rate"] = growth_rate(df)
    df["Future_Price_5Y"] = (df["Price_in_Lakhs"] * (1 + df["Annual_Growth_Rate"]) ** HORIZON_YEARS).round(2)
    df["Expected_Return_5Y_pct"] = ((df["Future_Price_5Y"] / df["Price_in_Lakhs"] - 1) * 100).round(2)
    # Multi-factor rule (7 checks, need >= 4)
    score = ((df["PPSF_vs_City"] <= 1.0).astype(int)
             + (df["BHK"] >= 3).astype(int)
             + df["Ready_Flag"]
             + (df["Transport_Score"] >= 2).astype(int)
             + (df["Amenity_Count"] >= 3).astype(int)
             + df["Security_Flag"]
             + (df["Annual_Growth_Rate"] >= BASE_GROWTH).astype(int))
    df["Investment_Score"] = score
    df["Good_Investment"] = (score >= 4).astype(int)
    return df
