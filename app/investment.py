"""
Investment Risk Analysis - Predict future price and days on market to assess investment risk.
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Ensure parent directory is in path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app.constants import (
    COUNTY_COL, CITY_COL, PROPERTY_TYPE_COL, LISTING_TYPE_COL,
    RISK_LOW_THRESHOLD, RISK_MEDIUM_THRESHOLD
)
from app.data_io import load_latest_listings
from app.preprocessing import normalize_listings_schema, prepare_prediction_input
from app.models import load_models, predict_dom, predict_price, get_risk_level


def calculate_investment_risk(predicted_price: float, predicted_dom: float, investment_amount: float) -> dict:
    """
    Calculate comprehensive investment risk based on predictions.
    
    Args:
        predicted_price: Predicted future property value
        predicted_dom: Predicted days on market
        investment_amount: User's investment amount
        
    Returns:
        Dictionary with risk assessment
    """
    # DOM Risk
    dom_risk = get_risk_level(predicted_dom)
    
    # Price vs Investment Risk
    if predicted_price < investment_amount * 0.9:
        price_risk = "High"
        price_risk_score = 3
    elif predicted_price < investment_amount:
        price_risk = "Medium"
        price_risk_score = 2
    else:
        price_risk = "Low"
        price_risk_score = 1
    
    # DOM Risk Score
    if dom_risk == "High":
        dom_risk_score = 3
    elif dom_risk == "Medium":
        dom_risk_score = 2
    else:
        dom_risk_score = 1
    
    # Overall Risk
    overall_score = (price_risk_score + dom_risk_score) / 2
    
    if overall_score >= 2.5:
        overall_risk = "High Risk"
        risk_color = "🔴"
        recommendation = "NOT RECOMMENDED - High risk of loss and extended market time"
    elif overall_score >= 1.5:
        overall_risk = "Medium Risk"
        risk_color = "🟡"
        recommendation = "PROCEED WITH CAUTION - Moderate risk factors present"
    else:
        overall_risk = "Low Risk"
        risk_color = "🟢"
        recommendation = "GOOD INVESTMENT - Favorable conditions for return"
    
    # Calculate potential return
    potential_return = predicted_price - investment_amount
    return_percentage = (potential_return / investment_amount * 100) if investment_amount > 0 else 0
    
    return {
        'overall_risk': overall_risk,
        'risk_color': risk_color,
        'recommendation': recommendation,
        'dom_risk': dom_risk,
        'price_risk': price_risk,
        'predicted_price': predicted_price,
        'predicted_dom': predicted_dom,
        'investment_amount': investment_amount,
        'potential_return': potential_return,
        'return_percentage': return_percentage,
        'overall_score': overall_score
    }


def render_risk_assessment(risk_data: dict):
    """Render comprehensive risk assessment display."""
    
    # Main risk indicator
    st.markdown(f"""
    <div style="padding:30px;border-radius:15px;background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);color:white;text-align:center;margin:20px 0;">
        <h1 style="margin:0;font-size:48px;">{risk_data['risk_color']}</h1>
        <h2 style="margin:10px 0;">{risk_data['overall_risk']}</h2>
        <p style="font-size:18px;margin:10px 0;">{risk_data['recommendation']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Predicted Property Value",
            f"${risk_data['predicted_price']:,.0f}",
            f"{risk_data['return_percentage']:+.1f}%"
        )
        st.caption(f"Investment: ${risk_data['investment_amount']:,.0f}")
    
    with col2:
        st.metric(
            "Days on Market",
            f"{risk_data['predicted_dom']:.0f} days",
            delta=None
        )
        dom_color = "🟢" if risk_data['dom_risk'] == "Low" else "🟡" if risk_data['dom_risk'] == "Medium" else "🔴"
        st.caption(f"{dom_color} {risk_data['dom_risk']} Risk")
    
    with col3:
        st.metric(
            "Potential Return",
            f"${risk_data['potential_return']:,.0f}",
            f"{risk_data['return_percentage']:.1f}%"
        )
        price_color = "🟢" if risk_data['price_risk'] == "Low" else "🟡" if risk_data['price_risk'] == "Medium" else "🔴"
        st.caption(f"{price_color} {risk_data['price_risk']} Price Risk")
    
    # Risk breakdown
    st.divider()
    st.subheader("Risk Analysis Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Market Liquidity Risk (Days on Market)**")
        if risk_data['predicted_dom'] <= RISK_LOW_THRESHOLD:
            st.success(f"✓ Property expected to sell quickly ({risk_data['predicted_dom']:.0f} days)")
        elif risk_data['predicted_dom'] <= RISK_MEDIUM_THRESHOLD:
            st.warning(f"⚠ Moderate time to sell ({risk_data['predicted_dom']:.0f} days)")
        else:
            st.error(f"✗ Extended time on market ({risk_data['predicted_dom']:.0f} days)")
    
    with col2:
        st.write("**Value Risk (Price vs Investment)**")
        if risk_data['potential_return'] > 0:
            st.success(f"✓ Positive return expected (+${risk_data['potential_return']:,.0f})")
        elif risk_data['potential_return'] > -risk_data['investment_amount'] * 0.1:
            st.warning(f"⚠ Minimal return (${risk_data['potential_return']:,.0f})")
        else:
            st.error(f"✗ Potential loss (${risk_data['potential_return']:,.0f})")


def main():
    """Main investment analysis application."""
    # st.set_page_config(page_title="Investment Risk Analysis", layout="wide")
    st.title("Alloy Tower Estate Investment Risk Analyzer")
    st.caption("Predict future value and market time to assess investment risk")
    
    # Load data for reference
    df, _ = load_latest_listings()
    
    if df is None:
        st.error("No listings data found. Run data ingestion first.")
        st.stop()
    
    df = normalize_listings_schema(df)
    
    # Load models
    try:
        dom_model, price_model, feature_cols = load_models()
    except Exception as e:
        st.error(f"Models not found: {e}")
        st.stop()
    
    # Input Section
    st.header(" Property Details")
    st.write("Enter the property characteristics to analyze investment risk:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Location")
        county = st.selectbox(
            "County",
            sorted(df[COUNTY_COL].dropna().unique().tolist()),
            key="inv_county"
        )
        city = st.selectbox(
            "City",
            sorted(df[df[COUNTY_COL] == county][CITY_COL].dropna().unique().tolist()),
            key="inv_city"
        )
        zip_code = st.number_input("Zip Code", value=75000, step=1, key="inv_zip")
    
    with col2:
        st.subheader("Property Features")
        property_type = st.selectbox(
            "Property Type",
            sorted(df[PROPERTY_TYPE_COL].dropna().unique().tolist()),
            key="inv_ptype"
        )
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, key="inv_beds")
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, key="inv_baths")
        square_footage = st.number_input("Square Footage", min_value=500, max_value=20000, value=2000, step=100, key="inv_sqft")
    
    with col3:
        st.subheader("Additional Details")
        year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2010, key="inv_year")
        lot_size = st.number_input("Lot Size (sq ft)", min_value=0, max_value=100000, value=7000, step=100, key="inv_lot")
        hoa_fee = st.number_input("HOA Fee (monthly)", min_value=0, max_value=5000, value=0, step=10, key="inv_hoa")
        
        st.divider()
        
        # Future listing date selector
        st.write("**📅 Expected Listing Date**")
        from datetime import datetime, timedelta
        
        today = datetime.now()
        max_date = today + timedelta(days=365*2)  # Allow up to 2 years in future
        
        listing_date = st.date_input(
            "When do you plan to list?",
            value=today,
            min_value=today,
            max_value=max_date,
            key="inv_listing_date",
            help="Select a future date to see how timing affects predictions"
        )
        
        st.divider()
        
        investment_amount = st.number_input(
            "💰 Your Investment Amount",
            min_value=10000,
            max_value=10000000,
            value=300000,
            step=10000,
            key="inv_amount",
            help="Enter the amount you plan to invest"
        )
    
    # Analyze button
    if st.button("🔍 Analyze Investment Risk", type="primary", use_container_width=True):
        
        with st.spinner("Analyzing investment risk..."):
            
            # Build input row with proper data types
            listing_type_value = "For Sale"  # Default
            if LISTING_TYPE_COL in df.columns:
                mode_result = df[LISTING_TYPE_COL].mode()
                if len(mode_result) > 0:
                    listing_type_value = str(mode_result.iloc[0])
            
            # Get median lat/lon for the city, with fallback
            city_data = df[df[CITY_COL] == city] if CITY_COL in df.columns else df
            latitude_value = float(city_data['latitude'].median()) if 'latitude' in df.columns and not city_data.empty else 30.0
            longitude_value = float(city_data['longitude'].median()) if 'longitude' in df.columns and not city_data.empty else -97.0
            
            # Calculate price per sq ft (use median from similar properties)
            price_per_sqft = 0.0
            if 'price_per_sq_ft' in df.columns:
                similar_props = df[
                    (df[PROPERTY_TYPE_COL] == property_type) & 
                    (df['square_footage'].notna())
                ]
                if not similar_props.empty:
                    price_per_sqft = float(similar_props['price_per_sq_ft'].median())
            
            # Extract temporal features from selected listing date
            from datetime import datetime
            if isinstance(listing_date, datetime):
                listing_dt = listing_date
            else:
                listing_dt = datetime.combine(listing_date, datetime.min.time())
            
            listed_year = listing_dt.year
            listed_month = listing_dt.month
            listed_dayofweek = listing_dt.weekday()
            
            input_row = {
                COUNTY_COL: str(county),
                CITY_COL: str(city),
                PROPERTY_TYPE_COL: str(property_type),
                'zip_code': float(zip_code),
                'bedrooms': float(bedrooms),
                'bathrooms': float(bathrooms),
                'square_footage': float(square_footage),
                'year_built': float(year_built),
                'lot_size': float(lot_size),
                'hoa_fee': float(hoa_fee),
                'price_per_sq_ft': price_per_sqft,
                'listing_type': listing_type_value,
                'latitude': latitude_value,
                'longitude': longitude_value,
                'listed_year': listed_year,
                'listed_month': listed_month,
                'listed_dayofweek': listed_dayofweek,
                'unit': "Unknown"  # Add missing categorical feature
            }
            
            # Prepare prediction input
            X_pred = prepare_prediction_input(input_row, feature_cols)
            
            # Make predictions
            predicted_dom = predict_dom(dom_model, X_pred)
            predicted_price = predict_price(price_model, X_pred)
            
            # Calculate risk
            risk_data = calculate_investment_risk(predicted_price, predicted_dom, investment_amount)
            
            # Display results
            st.divider()
            st.header("Investment Analysis Results")
            
            # Show listing date info
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June', 
                          'July', 'August', 'September', 'October', 'November', 'December']
            
            st.info(f" **Analysis for listing on:** {day_names[listed_dayofweek]}, "
                   f"{month_names[listed_month]} {listing_dt.day}, {listed_year}")
            
            render_risk_assessment(risk_data)
            
            # Additional insights
            st.divider()
            st.subheader(" Key Insights")
            
            insights = []
            
            # Timing insight
            if listed_month in [3, 4, 5, 6]:  # Spring/early summer
                insights.append("✓ Spring/early summer is typically a strong selling season")
            elif listed_month in [11, 12, 1]:  # Winter
                insights.append("⚠ Winter months may see slower market activity")
            
            if listed_dayofweek in [4, 5]:  # Friday, Saturday
                insights.append("✓ Weekend listings often get more initial views")
            
            if risk_data['predicted_dom'] <= RISK_LOW_THRESHOLD:
                insights.append("✓ Property is expected to sell quickly, indicating strong market demand")
            elif risk_data['predicted_dom'] > RISK_MEDIUM_THRESHOLD:
                insights.append("⚠ Extended time on market may indicate pricing or market challenges")
            
            if risk_data['return_percentage'] > 10:
                insights.append("✓ Strong potential return on investment")
            elif risk_data['return_percentage'] < 0:
                insights.append("⚠ Investment amount exceeds predicted value - consider negotiating")
            
            if risk_data['overall_risk'] == "Low Risk":
                insights.append("✓ Favorable investment conditions across multiple factors")
            elif risk_data['overall_risk'] == "High Risk":
                insights.append("⚠ Multiple risk factors present - thorough due diligence recommended")
            
            for insight in insights:
                st.write(insight)
            
            # Show input features used
            with st.expander("🔧 View Prediction Input Features"):
                st.dataframe(X_pred, use_container_width=True)


if __name__ == "__main__":
    main()
