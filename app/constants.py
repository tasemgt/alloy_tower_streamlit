"""
Constants and configuration for the AlloyTower Dashboard.
"""
import os

# Directory paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
LISTINGS_DIR = os.path.join(ROOT_DIR, "ml", "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

# File paths
RISK_CSV_PATH = os.path.join(REPORTS_DIR, "dom_risk_scoring_output.csv")
DOM_MODEL_PATH = os.path.join(MODELS_DIR, "dom_model.joblib")
PRICE_MODEL_PATH = os.path.join(MODELS_DIR, "price_model.joblib")
FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.joblib")


# Model download URLs (Google Drive links)
DOM_MODEL_URL = "https://drive.google.com/uc?id=1l70G_Rirzs88fXxTtsFQ4rpj-p-MTGlT"
PRICE_MODEL_URL = "https://drive.google.com/uc?id=1qc_zboKF0qNEjuLEQk2IJtIWIM23XgZe"
FEATURE_COLUMNS_URL = "https://drive.google.com/uc?id=14RyQ-tbjrd8cDVy4YNPFxs5lkqDan2-C"

# Column name constants
LISTING_ID_COL = "listing_id"
COUNTY_COL = "county"
CITY_COL = "city"
PROPERTY_TYPE_COL = "property_type"
LISTING_TYPE_COL = "listing_type"
CURRENT_PRICE_COL = "current_price"
DAYS_ON_MARKET_COL = "days_on_market"
LISTED_DATE_COL = "listed_date"
HOA_FEE_COL = "hoa_fee"
PRICE_PER_SQ_FT_COL = "price_per_sq_ft"

# Possible ID column variations for risk file
POSSIBLE_ID_COLS = ["listing_id", "listingid", "id", "listing id", "listing-id"]

# Risk levels
RISK_LOW_THRESHOLD = 60
RISK_MEDIUM_THRESHOLD = 120

# UI labels
APP_TITLE = "AlloyTower Real Estate Dashboard"
PREDICTION_TITLE = "AlloyTower Live Predictions"
