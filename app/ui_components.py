"""
Reusable UI components.
"""
import streamlit as st
import numpy as np
import pandas as pd
from typing import Any


def kpi_card(label: str, value: Any) -> None:
    """
    Display a KPI card with label and value.
    
    Args:
        label: KPI label
        value: KPI value to display
    """
    st.markdown(
        f"""
        <div style="padding:14px;border-radius:12px;background:#111827;color:white">
            <div style="font-size:12px;opacity:0.85">{label}</div>
            <div style="font-size:28px;font-weight:700">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def format_currency(value: float) -> str:
    """Format a numeric value as currency."""
    if np.isnan(value):
        return "N/A"
    return f"${value:,.0f}"


def format_number(value: float, suffix: str = "") -> str:
    """Format a numeric value with optional suffix."""
    if np.isnan(value):
        return "N/A"
    return f"{value:,.0f}{suffix}"


def paginated_dataframe(df: pd.DataFrame, page_size: int = 10, key: str = "page") -> None:
    """
    Display a paginated dataframe.
    
    Args:
        df: DataFrame to display
        page_size: Number of rows per page
        key: Unique key for the pagination widget
    """
    if len(df) == 0:
        st.info("No data to display")
        return
    
    total_pages = (len(df) - 1) // page_size + 1
    
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        page = st.number_input(
            f"Page (1-{total_pages})",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
            key=key
        )
    
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, len(df))
    
    st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True)
    st.caption(f"Showing {start_idx + 1}-{end_idx} of {len(df)} rows")
