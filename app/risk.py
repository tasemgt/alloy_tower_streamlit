"""
Risk scoring and analysis utilities.
"""
import pandas as pd
from typing import Dict, Any, List

from app.constants import (
    LISTING_ID_COL, COUNTY_COL, CITY_COL, PROPERTY_TYPE_COL, CURRENT_PRICE_COL
)
from app.data_io import normalize_risk_id_column
from app.filters import apply_filters


def merge_risk_with_listings(
    risk_df: pd.DataFrame,
    listings_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge risk output with listing metadata.
    
    Args:
        risk_df: Risk scoring DataFrame
        listings_df: Listings DataFrame
        
    Returns:
        Merged DataFrame
    """
    # Normalize ID columns
    risk_df = normalize_risk_id_column(risk_df)
    listings_df = listings_df.copy()
    listings_df[LISTING_ID_COL] = listings_df[LISTING_ID_COL].astype(str).str.strip()
    
    # Select metadata columns that exist
    meta_cols = [LISTING_ID_COL, COUNTY_COL, CITY_COL, PROPERTY_TYPE_COL, CURRENT_PRICE_COL]
    available_meta_cols = [c for c in meta_cols if c in listings_df.columns]
    
    # Merge
    merged = risk_df.merge(
        listings_df[available_meta_cols],
        on=LISTING_ID_COL,
        how="left"
    )
    
    return merged


def filter_risk_output(
    risk_df: pd.DataFrame,
    filters: Dict[str, Any]
) -> pd.DataFrame:
    """
    Apply filters to risk output DataFrame.
    
    Args:
        risk_df: Risk DataFrame
        filters: Dictionary of filter selections
        
    Returns:
        Filtered risk DataFrame
    """
    return apply_filters(risk_df, filters)


def sort_by_risk(risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort risk DataFrame by risk score if available.
    
    Args:
        risk_df: Risk DataFrame
        
    Returns:
        Sorted DataFrame
    """
    if "risk_score_0_100" in risk_df.columns:
        return risk_df.sort_values("risk_score_0_100", ascending=False)
    return risk_df


def get_risk_display_columns(risk_df: pd.DataFrame) -> List[str]:
    """
    Get the list of columns to display in risk output.
    
    Args:
        risk_df: Risk DataFrame
        
    Returns:
        List of column names to display
    """
    base_cols = [
        LISTING_ID_COL, COUNTY_COL, CITY_COL, PROPERTY_TYPE_COL,
        CURRENT_PRICE_COL, "pred_dom", "risk_level"
    ]
    optional_cols = ["risk_score_0_100", "recommended_action"]
    
    display_cols = [c for c in base_cols if c in risk_df.columns]
    display_cols += [c for c in optional_cols if c in risk_df.columns]
    
    return display_cols
