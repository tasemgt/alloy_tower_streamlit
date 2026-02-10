"""
Dynamic filter generation and application utilities.
"""
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any

from app.constants import COUNTY_COL, CITY_COL, PROPERTY_TYPE_COL, CURRENT_PRICE_COL


def create_categorical_filter(
    df: pd.DataFrame,
    column: str,
    label: str,
    key: str = None
) -> Any:
    """
    Create a categorical filter widget.
    
    Args:
        df: DataFrame to filter
        column: Column name to filter on
        label: Display label for the filter
        key: Unique key for the widget
        
    Returns:
        Selected value
    """
    if column not in df.columns:
        return "All"
    
    options = ["All"] + sorted(df[column].dropna().astype(str).unique().tolist())
    return st.sidebar.selectbox(label, options, key=key)


def create_price_range_filter(
    df: pd.DataFrame,
    price_col: str = CURRENT_PRICE_COL
) -> tuple:
    """
    Create a price range slider filter.
    
    Args:
        df: DataFrame to filter
        price_col: Name of the price column
        
    Returns:
        Tuple of (min_price, max_price)
    """
    if price_col not in df.columns or df[price_col].isna().all():
        return (0.0, 0.0)
    
    min_price = float(df[price_col].min()) if df[price_col].notna().any() else 0.0
    max_price = float(df[price_col].max()) if df[price_col].notna().any() else 0.0
    
    if min_price > max_price or np.isnan(min_price) or np.isnan(max_price):
        min_price, max_price = 0.0, 0.0
    
    # If min and max are the same, add a small buffer to make slider work
    if min_price == max_price:
        if min_price == 0.0:
            max_price = 1.0
        else:
            min_price = min_price * 0.99
            max_price = max_price * 1.01
    
    return st.sidebar.slider(
        "Current Price Range",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price)
    )


def apply_filters(
    df: pd.DataFrame,
    filters: Dict[str, Any]
) -> pd.DataFrame:
    """
    Apply multiple filters to a DataFrame.
    
    Args:
        df: DataFrame to filter
        filters: Dictionary of {column: value} filters
        
    Returns:
        Filtered DataFrame
    """
    filtered = df.copy()
    
    for column, value in filters.items():
        if value == "All" or value is None:
            continue
        
        if column == "price_range":
            min_price, max_price = value
            if CURRENT_PRICE_COL in filtered.columns:
                filtered = filtered[
                    (filtered[CURRENT_PRICE_COL].fillna(0) >= min_price) &
                    (filtered[CURRENT_PRICE_COL].fillna(0) <= max_price)
                ]
        else:
            if column in filtered.columns:
                filtered = filtered[filtered[column].astype(str) == str(value)]
    
    return filtered


def build_sidebar_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build all sidebar filters and return selected values.
    
    Args:
        df: DataFrame to build filters for
        
    Returns:
        Dictionary of filter selections
    """
    st.sidebar.header("Filters")
    
    filters = {}
    
    # County filter
    county = create_categorical_filter(df, COUNTY_COL, "County", "filter_county")
    filters[COUNTY_COL] = county
    
    # Apply county filter for cascading
    temp_df = apply_filters(df, {COUNTY_COL: county})
    
    # City filter (cascading)
    city = create_categorical_filter(temp_df, CITY_COL, "City", "filter_city")
    filters[CITY_COL] = city
    
    # Apply city filter for cascading
    temp_df = apply_filters(temp_df, {CITY_COL: city})
    
    # Property type filter (cascading)
    property_type = create_categorical_filter(
        temp_df, PROPERTY_TYPE_COL, "Property Type", "filter_property_type"
    )
    filters[PROPERTY_TYPE_COL] = property_type
    
    # Apply property type filter
    temp_df = apply_filters(temp_df, {PROPERTY_TYPE_COL: property_type})
    
    # Price range filter
    price_range = create_price_range_filter(temp_df)
    filters["price_range"] = price_range
    
    return filters
