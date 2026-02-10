"""
Main dashboard application for market overview and risk analysis.
"""
import streamlit as st
import numpy as np
import pandas as pd
import sys
import os

# Ensure parent directory is in path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app.constants import (
    APP_TITLE, LISTINGS_DIR, RISK_CSV_PATH,
    COUNTY_COL, CURRENT_PRICE_COL, DAYS_ON_MARKET_COL,
    HOA_FEE_COL, PRICE_PER_SQ_FT_COL
)
from app.data_io import load_latest_listings, load_risk_output
from app.preprocessing import normalize_listings_schema
from app.filters import build_sidebar_filters, apply_filters
from app.plots import (
    plot_top_counties, plot_avg_price_by_property_type,
    plot_price_distribution, plot_dom_distribution,
    plot_price_vs_sqft_scatter, plot_correlation_heatmap
)
from app.risk import (
    merge_risk_with_listings, filter_risk_output,
    sort_by_risk, get_risk_display_columns
)
from app.ui_components import kpi_card, format_currency, format_number, paginated_dataframe


def render_kpis(df):
    """Render KPI cards for the dashboard."""
    total_listings = len(df)
    num_counties = df[COUNTY_COL].nunique()
    avg_price = df[CURRENT_PRICE_COL].mean()
    avg_dom = df[DAYS_ON_MARKET_COL].mean()
    median_ppsf = df[PRICE_PER_SQ_FT_COL].median()
    avg_hoa = df[HOA_FEE_COL].mean()
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        kpi_card("Total Listings", f"{total_listings:,}")
    with c2:
        kpi_card("Unique Counties", f"{num_counties:,}")
    with c3:
        kpi_card("Avg Price", format_currency(avg_price))
    with c4:
        kpi_card("Avg DOM", format_number(avg_dom, " days"))
    with c5:
        kpi_card("Median PPSF", format_currency(median_ppsf))
    with c6:
        kpi_card("Avg HOA Fee", format_currency(avg_hoa))


def render_geographic_heatmap(df):
    """Render geographic heatmap with risk overlay."""
    st.subheader("🗺️ Geographic Risk Heatmap")
    
    # Check if we have geographic data
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        st.info("Geographic data not available for mapping.")
        return
    
    # Prepare map data with explicit column names
    map_df = df[['latitude', 'longitude', CURRENT_PRICE_COL, DAYS_ON_MARKET_COL]].copy()
    map_df = map_df.dropna()
    
    if len(map_df) == 0:
        st.info("No geographic data available for selected filters.")
        return
    
    # Rename columns for easier reference in tooltip
    map_df = map_df.rename(columns={
        CURRENT_PRICE_COL: 'price',
        DAYS_ON_MARKET_COL: 'dom'
    })
    
    # Calculate risk score (normalized DOM)
    map_df['dom_numeric'] = pd.to_numeric(map_df['dom'], errors='coerce')
    map_df = map_df.dropna(subset=['dom_numeric'])
    
    # Normalize risk score to 0-100
    if map_df['dom_numeric'].max() > 0:
        map_df['risk_score'] = (map_df['dom_numeric'] / map_df['dom_numeric'].max() * 100).round(0)
    else:
        map_df['risk_score'] = 0
    
    # Color code by risk
    def get_color(risk):
        if risk <= 33:
            return [0, 255, 0, 160]  # Green - Low risk
        elif risk <= 66:
            return [255, 255, 0, 160]  # Yellow - Medium risk
        else:
            return [255, 0, 0, 160]  # Red - High risk
    
    map_df['color'] = map_df['risk_score'].apply(get_color)
    
    # Create map using pydeck
    try:
        import pydeck as pdk
        
        # Calculate center
        center_lat = map_df['latitude'].median()
        center_lon = map_df['longitude'].median()
        
        # Create layer
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=map_df,
            get_position='[longitude, latitude]',
            get_color='color',
            get_radius=200,
            pickable=True,
            opacity=0.6,
            stroked=True,
            filled=True,
            radius_scale=1,
            radius_min_pixels=5,
            radius_max_pixels=15,
        )
        
        # Set view
        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=8,
            pitch=0,
        )
        
        # Render map with improved tooltip
        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={
              "html": """
<div style='background-color: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 12px;
            border-radius: 8px;
            font-family: Arial, sans-serif;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);'>
    <div style='font-size: 14px; font-weight: bold; margin-bottom: 8px; border-bottom: 1px solid #666; padding-bottom: 4px;'>
        Property Details
    </div>
    <div style='font-size: 12px; line-height: 1.6;'>
        <div><strong>Price:</strong> ${price}</div>
        <div><strong>Days on Market:</strong> {dom} days</div>
        <div><strong>Risk Score:</strong> {risk_score}/100</div>
    </div>
</div>
""",
                "style": {
                    "backgroundColor": "transparent",
                    "color": "white",
                    "zIndex": "10000"
                }
            }
        )
        
        st.pydeck_chart(r)
        
        # Legend
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("🟢 **Low Risk** (Fast selling, ≤33% normalized)")
        with col2:
            st.markdown("🟡 **Medium Risk** (Moderate, 34-66% normalized)")
        with col3:
            st.markdown("🔴 **High Risk** (Slow selling, >66% normalized)")
        
        st.caption(f"Showing {len(map_df):,} properties on map. Hover over points for details.")
            
    except ImportError:
        st.warning("Pydeck not available. Install with: pip install pydeck")
        # Fallback to simple scatter plot
        st.scatter_chart(map_df, x='longitude', y='latitude', color='risk_score')


def render_advanced_charts(df):
    """Render advanced analytical charts."""
    tab1, tab2, tab3, tab4 = st.tabs([
        " Distributions", 
        " Correlations", 
        " Price Analysis",
        " Market Segments"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_price_distribution(df))
        with col2:
            st.pyplot(plot_dom_distribution(df))
    
    with tab2:
        st.pyplot(plot_correlation_heatmap(df))
        st.caption("Correlation matrix shows relationships between numeric features. "
                  "Green = positive correlation, Red = negative correlation.")
    
    with tab3:
        st.pyplot(plot_price_vs_sqft_scatter(df))
        st.caption("Scatter plot shows relationship between property size and price. "
                  "Trend line indicates general pricing pattern.")
    
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_top_counties(df))
        with col2:
            st.pyplot(plot_avg_price_by_property_type(df))


def render_risk_section(filtered_df, filters):
    """Render the risk analysis section with filtered data."""
    st.subheader(" High-Risk Listings Analysis")
    
    risk_out = load_risk_output()
    
    if risk_out is None:
        st.info(
            "Risk scoring data not available. "
            "This section shows properties with high days-on-market risk when risk data is generated."
        )
        return
    
    try:
        # Get listing IDs from filtered data
        filtered_listing_ids = filtered_df['listing_id'].astype(str).str.strip().unique()
        
        # Normalize risk data listing IDs
        from app.data_io import normalize_risk_id_column
        risk_out = normalize_risk_id_column(risk_out)
        risk_out['listing_id'] = risk_out['listing_id'].astype(str).str.strip()
        
        # Filter risk data to only include listings in filtered dataset
        risk_filtered = risk_out[risk_out['listing_id'].isin(filtered_listing_ids)]
        
        if len(risk_filtered) == 0:
            st.warning("No risk data available for the current filter selection.")
            return
        
        # Merge with filtered listings to get metadata
        merged = risk_filtered.merge(
            filtered_df[['listing_id', 'county', 'city', 'property_type', 'current_price']],
            on='listing_id',
            how='left'
        )
        
        # Sort by risk
        merged = sort_by_risk(merged)
        display_cols = get_risk_display_columns(merged)
        
        # Show summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            high_risk = len(merged[merged.get('risk_level', '') == 'High'])
            st.metric("High Risk Properties", f"{high_risk:,}")
        with col2:
            avg_risk_dom = merged.get('pred_dom', pd.Series([0])).mean()
            st.metric("Avg Predicted DOM", f"{avg_risk_dom:.0f} days")
        with col3:
            if 'risk_score_0_100' in merged.columns:
                avg_risk_score = merged['risk_score_0_100'].mean()
                st.metric("Avg Risk Score", f"{avg_risk_score:.0f}/100")
        
        st.caption(f"Showing {len(merged):,} properties with risk scores (from {len(filtered_df):,} filtered properties)")
        
        paginated_dataframe(merged[display_cols], page_size=10, key="risk_page")
        
    except ValueError as e:
        st.error(str(e))


def main():
    """Main dashboard application."""
    # st.set_page_config(page_title="Market Dashboard", layout="wide")
    st.title("Alloy tower Estate Market Dashboard")
    st.caption("Comprehensive market analysis with advanced visualizations")
    
    # Load data
    df, listings_path = load_latest_listings()
    
    if df is None:
        st.error(
            f"No listings CSV found in: {LISTINGS_DIR}\n\n"
            "Run ingest first (to generate clean_sales_listings_*.csv)."
        )
        st.stop()
    
    st.caption(f"Data source: {listings_path}")
    
    # Normalize schema
    df = normalize_listings_schema(df)
    
    # Build filters
    filters = build_sidebar_filters(df)
    
    # Apply filters
    filtered = apply_filters(df, filters)
    
    # Show filter impact
    if len(filtered) < len(df):
        st.info(f" Showing {len(filtered):,} of {len(df):,} properties (filters applied)")
    
    # Render KPIs
    render_kpis(filtered)
    st.divider()
    
    # Geographic heatmap
    render_geographic_heatmap(filtered)
    st.divider()
    
    # Advanced charts
    render_advanced_charts(filtered)
    st.divider()
    
    # Risk section (pass filtered data)
    render_risk_section(filtered, filters)


if __name__ == "__main__":
    main()
