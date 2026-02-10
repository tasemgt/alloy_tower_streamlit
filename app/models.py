"""
Model loading and prediction utilities with auto-download + caching.
"""

import os
import joblib
import streamlit as st
from typing import Tuple, Any, List
import gdown

# Import sklearn to ensure it's available when unpickling models
try:
    import sklearn
    import sklearn.ensemble
    import sklearn.pipeline
    import sklearn.compose
    import sklearn.preprocessing
except ImportError as e:
    st.error(f"scikit-learn not found: {e}")
    st.info("Install it with: pip install scikit-learn")
    raise

from app.constants import (
    DOM_MODEL_PATH, PRICE_MODEL_PATH, FEATURE_COLUMNS_PATH,
    RISK_LOW_THRESHOLD, RISK_MEDIUM_THRESHOLD,
    DOM_MODEL_URL, PRICE_MODEL_URL, FEATURE_COLUMNS_URL  # ADD THESE
)


def download_if_missing(path: str, url: str, label: str):
    """Download model only if not present."""
    if not os.path.exists(path):
        st.warning(f"Downloading {label} model (first run only)...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        gdown.download(url, path, quiet=False)


@st.cache_resource
def load_models() -> Tuple[Any, Any, List[str]]:
    """
    Load trained models and feature columns (cached).
    Downloads models once if missing.
    """
    try:
        # Download if missing (only runs once)
        download_if_missing(DOM_MODEL_PATH, DOM_MODEL_URL, "DOM")
        download_if_missing(PRICE_MODEL_PATH, PRICE_MODEL_URL, "Price")
        download_if_missing(FEATURE_COLUMNS_PATH, FEATURE_COLUMNS_URL, "Feature Columns")

        # Load models
        dom_model = joblib.load(DOM_MODEL_PATH)
        price_model = joblib.load(PRICE_MODEL_PATH)
        feature_cols = joblib.load(FEATURE_COLUMNS_PATH)

        return dom_model, price_model, feature_cols

    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        raise


def predict_dom(model: Any, X) -> float:
    """Predict days on market."""
    return float(model.predict(X)[0])


def predict_price(model: Any, X) -> float:
    """Predict property price."""
    return float(model.predict(X)[0])


def get_risk_level(predicted_dom: float) -> str:
    """Determine risk level based on predicted DOM."""
    if predicted_dom <= RISK_LOW_THRESHOLD:
        return "Low"
    elif predicted_dom <= RISK_MEDIUM_THRESHOLD:
        return "Medium"
    return "High"