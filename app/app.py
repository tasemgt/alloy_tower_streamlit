"""
AlloyTower Real Estate Dashboard - Main Application Entry Point

This application provides two main views:
1. Investment Analyzer: Predict future price & DOM to assess investment risk
2. Market Dashboard: Explore market data with KPIs, charts, and risk analysis

Note: Run from project root using: streamlit run run_app.py
"""
import streamlit as st
import sys
import os

# Ensure parent directory is in path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the application modules
from app import dashboard, investment


def main():
    """Main application router."""
    st.sidebar.title("Navigation")
    
    page = st.sidebar.radio(
        "Select View",
        ["Investment Analyzer", "Market Dashboard"],
        help="Analyze investments or explore market data"
    )
    
    st.sidebar.divider()
    
    if page == "Investment Analyzer":
        investment.main()
    else:
        dashboard.main()


if __name__ == "__main__":
    main()
