"""Real Estate ML Model Training Script.

This script trains machine learning models to predict:
1. Days on Market (DOM) - How long a property will stay on the market
2. Property Price - The expected sale price

The script:
- Loads and preprocesses real estate data
- Tests multiple ML algorithms
- Selects the best performing model
- Saves trained models for production use

Example:
    Run from command line:
        $ python train_models.py

Attributes:
    None

Todo:
    * Add cross-validation
    * Implement time-based validation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
from sklearn.ensemble import (
    ExtraTreesRegressor, 
    RandomForestRegressor, 
    GradientBoostingRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("REAL ESTATE ML MODEL TRAINING - ALGORITHM COMPARISON")
print("=" * 80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

print("Loading data...")
data_dir = "ml/data"
listings_file = [f for f in os.listdir(data_dir) if f.startswith("clean_sales_listings")][0]
df = pd.read_csv(os.path.join(data_dir, listings_file))

# Normalize column names to lowercase for consistency
df.columns = df.columns.str.lower()
print(f"Loaded {len(df)} records\n")

# Build training dataset
train_df = df.copy()

# Cap days_on_market at 365 days to handle outliers
# Rationale: Properties staying >365 days are outliers that can skew the model
train_df["days_on_market_capped"] = train_df["days_on_market"].clip(upper=365)

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

# Extract temporal features from listing date
# These capture seasonal and weekly patterns in real estate sales
if "listed_date" in train_df.columns:
    train_df["listed_date"] = pd.to_datetime(train_df["listed_date"], errors="coerce")
    train_df["listed_year"] = train_df["listed_date"].dt.year
    train_df["listed_month"] = train_df["listed_date"].dt.month
    train_df["listed_dayofweek"] = train_df["listed_date"].dt.dayofweek

# =============================================================================
# TARGET VARIABLES AND FEATURE SELECTION
# =============================================================================

# Define target variables
y_dom = train_df["days_on_market_capped"]  # Days on market (capped)
y_price = train_df["current_price"]  # Property price

# Drop columns to prevent data leakage and remove non-predictive features
# Data leakage prevention: Exclude any information not available at listing time
drop_cols = [
    "days_on_market",           # Raw target (use capped version)
    "days_on_market_capped",    # Target variable
    "current_price",            # Target variable for price model
    "listing_id",               # Identifier (no predictive value)
    "address",                  # Too specific (overfitting risk)
    "street",                   # Too specific (overfitting risk)
    "mls_number",               # Identifier
    "agent_id",                 # Post-listing information
    "office_id",                # Post-listing information
    "removed_date",             # Future information (data leakage)
    "last_seen_ts",             # Future information (data leakage)
    "status",                   # Post-listing information
]

X = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])

# Replace infinities with NaN for proper handling
X = X.replace([np.inf, -np.inf], np.nan)

# Remove raw date strings (temporal features already extracted)
date_cols = ["listed_date", "created_date"]
X = X.drop(columns=[c for c in date_cols if c in X.columns])

# Identify numeric and categorical columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

# Fill missing values
X[num_cols] = X[num_cols].fillna(X[num_cols].median())
X[cat_cols] = X[cat_cols].fillna("Unknown")

print(f"Features: {len(X.columns)} total")
print(f"  - Numeric: {len(num_cols)}")
print(f"  - Categorical: {len(cat_cols)}")
print(f"Categorical columns: {cat_cols}\n")

# Enhanced preprocessor with scaling for numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

# Train/test split
X_train, X_test, y_dom_train, y_dom_test = train_test_split(
    X, y_dom, test_size=0.2, random_state=42
)
_, _, y_price_train, y_price_test = train_test_split(
    X, y_price, test_size=0.2, random_state=42
)

print("=" * 80)
print("TRAINING DAYS ON MARKET (DOM) MODEL")
print("=" * 80)

# Define models to test (with reasonable default parameters)
models_to_test = {
    "ExtraTrees": ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=1),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=200, random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "ElasticNet": ElasticNet(alpha=1.0, random_state=42),
    "KNeighbors": KNeighborsRegressor(n_neighbors=5, n_jobs=1)
}

results = []

print("\nTesting algorithms (this may take a few minutes)...\n")
print(f"{'Algorithm':<25} {'MAE (days)':<15} {'RMSE (days)':<15} {'R²':<10}")
print("-" * 80)

for name, model in models_to_test.items():
    try:
        # Create pipeline
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        
        # Fit model
        pipeline.fit(X_train, y_dom_train)
        
        # Predict
        pred = pipeline.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_dom_test, pred)
        rmse = np.sqrt(mean_squared_error(y_dom_test, pred))
        r2 = r2_score(y_dom_test, pred)
        
        results.append({
            'Algorithm': name,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'Model': pipeline
        })
        
        print(f"{name:<25} {mae:<15.2f} {rmse:<15.2f} {r2:<10.3f}")
        
    except Exception as e:
        print(f"{name:<25} FAILED: {str(e)[:40]}")

# Sort by MAE (lower is better)
results_df = pd.DataFrame(results).sort_values('MAE')

print("\n" + "=" * 80)
print("RESULTS SUMMARY - RANKED BY MAE (BEST TO WORST)")
print("=" * 80)
print(results_df[['Algorithm', 'MAE', 'RMSE', 'R²']].to_string(index=False))

# Get best model
best_result = results_df.iloc[0]
dom_model = best_result['Model']

print("\n" + "=" * 80)
print("BEST DOM MODEL")
print("=" * 80)
print(f"Algorithm: {best_result['Algorithm']}")
print(f"MAE:       {best_result['MAE']:.2f} days")
print(f"RMSE:      {best_result['RMSE']:.2f} days")
print(f"R²:        {best_result['R²']:.3f}")
print("=" * 80)

print("\n" + "=" * 80)
print("HYPERPARAMETER TUNING")
print("=" * 80)
print("\nTuning the best DOM model for optimal performance...")

# Define hyperparameter grids for top algorithms
param_grids = {
    "ExtraTrees": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [10, 20, 30, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None]
    },
    "RandomForest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [10, 20, 30, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None]
    },
    "GradientBoosting": {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__max_depth": [3, 5, 7],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__subsample": [0.8, 0.9, 1.0]
    },
    "HistGradientBoosting": {
        "model__max_iter": [100, 200, 300],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__max_depth": [3, 5, 7, 10, None],
        "model__min_samples_leaf": [10, 20, 30],
        "model__l2_regularization": [0.0, 0.1, 1.0]
    },
    "AdaBoost": {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.01, 0.1, 0.5, 1.0]
    },
    "Ridge": {
        "model__alpha": [0.1, 1.0, 10.0, 100.0]
    },
    "Lasso": {
        "model__alpha": [0.1, 1.0, 10.0, 100.0]
    },
    "ElasticNet": {
        "model__alpha": [0.1, 1.0, 10.0],
        "model__l1_ratio": [0.1, 0.5, 0.7, 0.9]
    },
    "KNeighbors": {
        "model__n_neighbors": [3, 5, 7, 10, 15],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2]
    }
}

# Tune the best DOM model
best_algo = best_result['Algorithm']
if best_algo in param_grids:
    print(f"\nTuning {best_algo} for DOM prediction...")
    
    # Get model class
    model_class = type(models_to_test[best_algo])
    
    # Create base model
    if best_algo in ["ExtraTrees", "RandomForest"]:
        base_model = model_class(random_state=42, n_jobs=1)
    elif best_algo == "GradientBoosting":
        base_model = model_class(random_state=42)
    elif best_algo == "HistGradientBoosting":
        base_model = model_class(random_state=42)
    elif best_algo == "AdaBoost":
        base_model = model_class(random_state=42)
    elif best_algo == "KNeighbors":
        base_model = model_class(n_jobs=1)
    else:
        base_model = model_class(random_state=42) if 'random_state' in model_class().get_params() else model_class()
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", base_model)
    ])
    
    # Randomized search (faster than grid search)
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grids[best_algo],
        n_iter=20,  # Number of parameter settings sampled
        cv=3,  # 3-fold cross-validation
        scoring='neg_mean_absolute_error',
        n_jobs=1,
        random_state=42,
        verbose=1
    )
    
    print(f"Running randomized search with 20 iterations and 3-fold CV...")
    random_search.fit(X_train, y_dom_train)
    
    # Get tuned model
    tuned_dom_model = random_search.best_estimator_
    
    # Evaluate on test set
    tuned_pred = tuned_dom_model.predict(X_test)
    tuned_mae = mean_absolute_error(y_dom_test, tuned_pred)
    tuned_rmse = np.sqrt(mean_squared_error(y_dom_test, tuned_pred))
    tuned_r2 = r2_score(y_dom_test, tuned_pred)
    
    print("\n" + "=" * 80)
    print("TUNED DOM MODEL RESULTS")
    print("=" * 80)
    print(f"Algorithm:     {best_algo}")
    print(f"\nBest parameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nBefore tuning:")
    print(f"  MAE:  {best_result['MAE']:.2f} days")
    print(f"  RMSE: {best_result['RMSE']:.2f} days")
    print(f"  R²:   {best_result['R²']:.3f}")
    print(f"\nAfter tuning:")
    print(f"  MAE:  {tuned_mae:.2f} days")
    print(f"  RMSE: {tuned_rmse:.2f} days")
    print(f"  R²:   {tuned_r2:.3f}")
    
    improvement = ((best_result['MAE'] - tuned_mae) / best_result['MAE']) * 100
    print(f"\nImprovement: {improvement:+.2f}%")
    print("=" * 80)
    
    # Use tuned model
    dom_model = tuned_dom_model
else:
    print(f"\nNo hyperparameter grid defined for {best_algo}, using default model.")

# Train PRICE model (using top 3 algorithms from DOM testing)
print("\n" + "=" * 80)
print("TRAINING PRICE MODEL")
print("=" * 80)

# Use top 3 algorithms from DOM results
top_algorithms = results_df.head(3)['Algorithm'].tolist()
print(f"\nTesting top 3 algorithms from DOM: {', '.join(top_algorithms)}\n")

price_results = []
print(f"{'Algorithm':<25} {'MAE ($)':<20} {'RMSE ($)':<20} {'R²':<10}")
print("-" * 80)

for algo_name in top_algorithms:
    # Get the model class from original dict
    model_class = type(models_to_test[algo_name])
    
    # Create new instance with same params
    if algo_name in ["ExtraTrees", "RandomForest"]:
        model = model_class(n_estimators=200, random_state=42, n_jobs=1)
    elif algo_name == "GradientBoosting":
        model = model_class(n_estimators=200, random_state=42)
    elif algo_name == "HistGradientBoosting":
        model = model_class(max_iter=200, random_state=42)
    elif algo_name == "AdaBoost":
        model = model_class(n_estimators=100, random_state=42)
    elif algo_name == "KNeighbors":
        model = model_class(n_neighbors=5, n_jobs=1)
    else:
        model = model_class(random_state=42) if 'random_state' in model_class().get_params() else model_class()
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    pipeline.fit(X_train, y_price_train)
    pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_price_test, pred)
    rmse = np.sqrt(mean_squared_error(y_price_test, pred))
    r2 = r2_score(y_price_test, pred)
    
    price_results.append({
        'Algorithm': algo_name,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'Model': pipeline
    })
    
    print(f"{algo_name:<25} ${mae:<19,.0f} ${rmse:<19,.0f} {r2:<10.3f}")

# Get best price model
price_results_df = pd.DataFrame(price_results).sort_values('MAE')
best_price_result = price_results_df.iloc[0]
price_model = best_price_result['Model']

print("\n" + "=" * 80)
print("BEST PRICE MODEL")
print("=" * 80)
print(f"Algorithm: {best_price_result['Algorithm']}")
print(f"MAE:       ${best_price_result['MAE']:,.0f}")
print(f"RMSE:      ${best_price_result['RMSE']:,.0f}")
print(f"R²:        {best_price_result['R²']:.3f}")
print("=" * 80)

# Tune the best PRICE model
print("\n" + "=" * 80)
print("HYPERPARAMETER TUNING - PRICE MODEL")
print("=" * 80)

best_price_algo = best_price_result['Algorithm']
if best_price_algo in param_grids:
    print(f"\nTuning {best_price_algo} for price prediction...")
    
    # Get model class
    model_class = type(models_to_test[best_price_algo])
    
    # Create base model
    if best_price_algo in ["ExtraTrees", "RandomForest"]:
        base_model = model_class(random_state=42, n_jobs=1)
    elif best_price_algo == "GradientBoosting":
        base_model = model_class(random_state=42)
    elif best_price_algo == "HistGradientBoosting":
        base_model = model_class(random_state=42)
    elif best_price_algo == "AdaBoost":
        base_model = model_class(random_state=42)
    elif best_price_algo == "KNeighbors":
        base_model = model_class(n_jobs=1)
    else:
        base_model = model_class(random_state=42) if 'random_state' in model_class().get_params() else model_class()
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", base_model)
    ])
    
    # Randomized search
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grids[best_price_algo],
        n_iter=20,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=1,
        random_state=42,
        verbose=1
    )
    
    print(f"Running randomized search with 20 iterations and 3-fold CV...")
    random_search.fit(X_train, y_price_train)
    
    # Get tuned model
    tuned_price_model = random_search.best_estimator_
    
    # Evaluate on test set
    tuned_pred = tuned_price_model.predict(X_test)
    tuned_mae = mean_absolute_error(y_price_test, tuned_pred)
    tuned_rmse = np.sqrt(mean_squared_error(y_price_test, tuned_pred))
    tuned_r2 = r2_score(y_price_test, tuned_pred)
    
    print("\n" + "=" * 80)
    print("TUNED PRICE MODEL RESULTS")
    print("=" * 80)
    print(f"Algorithm:     {best_price_algo}")
    print(f"\nBest parameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nBefore tuning:")
    print(f"  MAE:  ${best_price_result['MAE']:,.0f}")
    print(f"  RMSE: ${best_price_result['RMSE']:,.0f}")
    print(f"  R²:   {best_price_result['R²']:.3f}")
    print(f"\nAfter tuning:")
    print(f"  MAE:  ${tuned_mae:,.0f}")
    print(f"  RMSE: ${tuned_rmse:,.0f}")
    print(f"  R²:   {tuned_r2:.3f}")
    
    improvement = ((best_price_result['MAE'] - tuned_mae) / best_price_result['MAE']) * 100
    print(f"\nImprovement: {improvement:+.2f}%")
    print("=" * 80)
    
    # Use tuned model
    price_model = tuned_price_model
else:
    print(f"\nNo hyperparameter grid defined for {best_price_algo}, using default model.")

# Save models
print("\n" + "=" * 80)
print("FINAL MODEL SUMMARY")
print("=" * 80)
print("\nDays on Market (DOM) Model:")
print(f"  Algorithm: {best_result['Algorithm']}")
print(f"  MAE:       {mean_absolute_error(y_dom_test, dom_model.predict(X_test)):.2f} days")
print(f"  R²:        {r2_score(y_dom_test, dom_model.predict(X_test)):.3f}")

print("\nPrice Model:")
print(f"  Algorithm: {best_price_result['Algorithm']}")
print(f"  MAE:       ${mean_absolute_error(y_price_test, price_model.predict(X_test)):,.0f}")
print(f"  R²:        {r2_score(y_price_test, price_model.predict(X_test)):.3f}")

print("\n" + "=" * 80)
print("SAVING MODELS")
print("=" * 80)
os.makedirs("models", exist_ok=True)

joblib.dump(dom_model, "models/dom_model.joblib")
joblib.dump(price_model, "models/price_model.joblib")
joblib.dump(list(X.columns), "models/feature_columns.joblib")

print("\nSaved:")
print("  - models/dom_model.joblib")
print("  - models/price_model.joblib")
print("  - models/feature_columns.joblib")

print("\n" + "=" * 80)
print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
