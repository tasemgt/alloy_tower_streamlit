"""
AlloyTower Real Estate Dashboard - Launcher Script

This is the main entry point for the Streamlit application.
Run with: streamlit run run_app.py
"""
import streamlit as st
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the application modules
from app import dashboard, investment


def main():
    """Main application router."""
    st.set_page_config(layout="wide")
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
