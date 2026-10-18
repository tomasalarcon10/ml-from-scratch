"""
Data cleaning and feature engineering for the Ames Housing dataset (De Cock, 2011).
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "ames_housing.csv"

# Categorical columns where a missing value means "the house does not have it"
NONE_MEANS_ABSENT = [
    "Alley", "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2",
    "Fireplace Qu", "Garage Type", "Garage Finish", "Garage Qual", "Garage Cond",
    "Pool QC", "Fence", "Misc Feature", "Mas Vnr Type",
]
# Numeric columns where missing means zero (no basement, no garage, no veneer)
ZERO_MEANS_ABSENT = [
    "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF",
    "Bsmt Full Bath", "Bsmt Half Bath", "Garage Cars", "Garage Area",
]
QUALITY_SCALE = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ORDINAL_QUALITY = [
    "Exter Qual", "Exter Cond", "Bsmt Qual", "Bsmt Cond", "Heating QC", "Kitchen Qual",
    "Fireplace Qu", "Garage Qual", "Garage Cond", "Pool QC",
]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    return df.drop(columns=[c for c in ["Unnamed: 0", "Order", "PID"] if c in df.columns])


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Returns the cleaned frame and a log of every decision taken."""
    log = {"rows_raw": len(df)}
    df = df.copy()

    # 1. Outliers recommended by the dataset author: very large houses sold as partial sales
    outliers = df["Gr Liv Area"] > 4000
    log["outliers_removed"] = int(outliers.sum())
    df = df[~outliers]

    # 2. Structural missing values
    for c in NONE_MEANS_ABSENT:
        df[c] = df[c].fillna("None")
    for c in ZERO_MEANS_ABSENT:
        df[c] = df[c].fillna(0)

    # 3. Lot frontage: median of the same neighborhood
    log["lot_frontage_imputed"] = int(df["Lot Frontage"].isna().sum())
    df["Lot Frontage"] = df.groupby("Neighborhood")["Lot Frontage"].transform(lambda s: s.fillna(s.median()))
    df["Lot Frontage"] = df["Lot Frontage"].fillna(df["Lot Frontage"].median())

    # 4. Garage year built: when there is no garage, use the house year
    df["Garage Yr Blt"] = df["Garage Yr Blt"].fillna(df["Year Built"])

    # 5. Any remaining gaps: mode for categoricals, median for numerics
    for c in df.columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median() if pd.api.types.is_numeric_dtype(df[c]) else df[c].mode()[0])

    log["rows_clean"] = len(df)
    log["missing_after"] = int(df.isna().sum().sum())
    return df.reset_index(drop=True), log


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ORDINAL_QUALITY:
        df[c] = df[c].map(QUALITY_SCALE).fillna(0)

    df["Total SF"] = df["Total Bsmt SF"] + df["1st Flr SF"] + df["2nd Flr SF"]
    df["Total Baths"] = (df["Full Bath"] + 0.5 * df["Half Bath"]
                         + df["Bsmt Full Bath"] + 0.5 * df["Bsmt Half Bath"])
    df["House Age"] = (df["Yr Sold"] - df["Year Built"]).clip(lower=0)
    df["Years Since Remodel"] = (df["Yr Sold"] - df["Year Remod/Add"]).clip(lower=0)
    df["Total Porch SF"] = (df["Open Porch SF"] + df["Enclosed Porch"] + df["3Ssn Porch"]
                            + df["Screen Porch"] + df["Wood Deck SF"])
    df["Quality x Area"] = df["Overall Qual"] * df["Gr Liv Area"]
    df["Has Pool"] = (df["Pool Area"] > 0).astype(int)
    df["Has Garage"] = (df["Garage Area"] > 0).astype(int)
    df["Is New"] = (df["Sale Type"] == "New").astype(int)
    df["MS SubClass"] = df["MS SubClass"].astype(str)
    df["Mo Sold"] = df["Mo Sold"].astype(str)
    return df


def design_matrix(df: pd.DataFrame, skew_threshold: float = 0.75):
    """Log-transform skewed numerics, one-hot encode nominals. Returns X, y (log price), feature names."""
    y = np.log(df["SalePrice"].values)
    X = df.drop(columns=["SalePrice"])
    numeric = X.select_dtypes(include=[np.number]).columns
    skewed = [c for c in numeric if X[c].min() >= 0 and abs(X[c].skew()) > skew_threshold]
    X[skewed] = np.log1p(X[skewed])
    X = pd.get_dummies(X, drop_first=True, dtype=float)
    X = X.loc[:, X.std() > 0]
    return X, y, skewed
