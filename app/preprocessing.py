"""
Data preprocessing and feature engineering utilities.
"""
import pandas as pd
import numpy as np
from typing import List

from app.constants import (
    LISTING_ID_COL, COUNTY_COL, CITY_COL, PROPERTY_TYPE_COL,
    LISTED_DATE_COL, DAYS_ON_MARKET_COL, CURRENT_PRICE_COL,
    HOA_FEE_COL, PRICE_PER_SQ_FT_COL
)


def normalize_listings_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize listings DataFrame schema and data types.
    
    Args:
        df: Raw listings DataFrame
        
    Returns:
        Normalized DataFrame
    """
    df = df.copy()
    
    # Ensure key columns exist
    required_cols = [COUNTY_COL, CITY_COL, PROPERTY_TYPE_COL, LISTING_ID_COL]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
    
    # Normalize listing_id to string
    df[LISTING_ID_COL] = df[LISTING_ID_COL].astype(str).str.strip()
    
    # Convert date columns
    if LISTED_DATE_COL in df.columns:
        df[LISTED_DATE_COL] = pd.to_datetime(df[LISTED_DATE_COL], errors="coerce")
    
    # Convert numeric columns
    numeric_cols = [
        DAYS_ON_MARKET_COL, CURRENT_PRICE_COL, HOA_FEE_COL, PRICE_PER_SQ_FT_COL,
        "bedrooms", "bathrooms", "square_footage", "lot_size", "year_built",
        "zip_code", "latitude", "longitude"
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


def derive_time_features(df: pd.DataFrame, date_col: str = LISTED_DATE_COL) -> pd.DataFrame:
    """
    Derive time-based features from a date column.
    
    Args:
        df: DataFrame with date column
        date_col: Name of the date column
        
    Returns:
        DataFrame with added time features
    """
    df = df.copy()
    
    if date_col not in df.columns:
        return df
    
    date_series = pd.to_datetime(df[date_col], errors="coerce")
    df["listed_year"] = date_series.dt.year
    df["listed_month"] = date_series.dt.month
    df["listed_dayofweek"] = date_series.dt.dayofweek
    
    return df


def prepare_prediction_input(
    input_row: dict,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Prepare a single prediction input aligned to training features.
    
    Args:
        input_row: Dictionary of input values
        feature_cols: List of feature column names from training
        
    Returns:
        Single-row DataFrame ready for prediction
    """
    # Build DataFrame with exact feature columns
    X_pred = pd.DataFrame([{c: input_row.get(c, np.nan) for c in feature_cols}])
    
    # Replace infinities
    X_pred = X_pred.replace([np.inf, -np.inf], np.nan)
    
    # Identify column types
    num_cols = X_pred.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_pred.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    
    # Fill missing values for numeric columns with 0 (safer than median for single row)
    for col in num_cols:
        X_pred[col] = X_pred[col].fillna(0)
    
    # Fill missing values for categorical columns and ensure they're strings
    for col in cat_cols:
        X_pred[col] = X_pred[col].fillna("Unknown").astype(str)
    
    # Ensure all categorical columns are strings (no mixed types)
    for col in cat_cols:
        X_pred[col] = X_pred[col].astype(str)
    
    return X_pred


def build_input_from_listing(row: pd.Series) -> dict:
    """
    Build prediction input dictionary from a listing row.
    
    Args:
        row: Pandas Series representing a listing
        
    Returns:
        Dictionary of input features
    """
    input_row = row.to_dict()
    
    # Derive time features if listed_date exists
    listed_date = pd.to_datetime(input_row.get(LISTED_DATE_COL), errors="coerce")
    if pd.notna(listed_date):
        input_row["listed_year"] = listed_date.year
        input_row["listed_month"] = listed_date.month
        input_row["listed_dayofweek"] = listed_date.dayofweek
    else:
        input_row["listed_year"] = np.nan
        input_row["listed_month"] = np.nan
        input_row["listed_dayofweek"] = np.nan
    
    return input_row
