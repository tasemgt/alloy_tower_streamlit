"""
Data loading and validation utilities.
"""
import glob
import os
import pandas as pd
import streamlit as st
from typing import Optional, Tuple

from app.constants import LISTINGS_DIR, RISK_CSV_PATH, POSSIBLE_ID_COLS, LISTING_ID_COL


@st.cache_data
def load_latest_listings(data_dir: str = LISTINGS_DIR) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load the most recent listings CSV from the data directory.
    
    Returns:
        Tuple of (DataFrame, file_path) or (None, None) if no files found
    """
    files = sorted(glob.glob(os.path.join(data_dir, "clean_sales_listings_*.csv")))
    if not files:
        return None, None
    
    path = files[-1]
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    return df, path


@st.cache_data
def load_risk_output(path: str = RISK_CSV_PATH) -> Optional[pd.DataFrame]:
    """
    Load risk scoring output CSV if it exists.
    
    Returns:
        DataFrame or None if file doesn't exist
    """
    if not os.path.exists(path):
        return None
    
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    return df


def normalize_risk_id_column(risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and normalize the listing ID column in risk output.
    
    Args:
        risk_df: Risk output DataFrame
        
    Returns:
        DataFrame with normalized listing_id column
        
    Raises:
        ValueError: If no ID column is found
    """
    id_col = None
    for col in POSSIBLE_ID_COLS:
        if col in risk_df.columns:
            id_col = col
            break
    
    if id_col is None:
        raise ValueError(
            f"Risk file is missing a listing ID column. "
            f"Columns found: {risk_df.columns.tolist()}"
        )
    
    if id_col != LISTING_ID_COL:
        risk_df = risk_df.rename(columns={id_col: LISTING_ID_COL})
    
    risk_df[LISTING_ID_COL] = risk_df[LISTING_ID_COL].astype(str).str.strip()
    return risk_df
