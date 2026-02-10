# Real Estate Prediction Model Documentation

## Executive Summary

This document provides comprehensive documentation of the machine learning models developed for predicting Days on Market (DOM) and property prices in the real estate domain.

**Key Results:**
- **DOM Model**: MAE = 36.63 days, R² = 0.584 (13.5% improvement over baseline)
- **Price Model**: MAE = $71,818, R² = 0.795
- **Algorithm**: ExtraTrees Regressor (Extra Trees Ensemble)

---

## 1. Problem Statement

### Objective
Develop predictive models to:
1. Predict how long a property will stay on the market (Days on Market)
2. Predict the future sale price of a property

### Business Value
- Help investors assess investment risk
- Enable data-driven pricing decisions
- Optimize listing timing strategies
- Identify high-risk properties

---

## 2. Dataset

### Source
- **Database**: Snowflake (ALLOY_TOWER_DB.SILVER schema)
- **Tables**: clean_sales_listings, clean_sales_history, clean_sales_agents, clean_sales_offices
- **Records**: 13,428 property listings
- **Time Period**: Historical real estate transactions

### Features (18 total)

#### Categorical Features (5)
1. **unit**: Unit number/identifier
2. **city**: City location
3. **county**: County location
4. **property_type**: Type of property (Single Family, Condo, etc.)
5. **listing_type**: Type of listing (For Sale, etc.)

#### Numeric Features (10)
1. **zip_code**: ZIP code
2. **latitude**: Geographic latitude
3. **longitude**: Geographic longitude
4. **bedrooms**: Number of bedrooms
5. **bathrooms**: Number of bathrooms
6. **square_footage**: Property square footage
7. **lot_size**: Lot size in square feet
8. **year_built**: Year property was built
9. **hoa_fee**: Monthly HOA fee
10. **price_per_sq_ft**: Price per square foot

#### Temporal Features (3)
1. **listed_year**: Year property was listed
2. **listed_month**: Month property was listed (1-12)
3. **listed_dayofweek**: Day of week listed (0=Monday, 6=Sunday)

### Target Variables
- **days_on_market_capped**: Days on market (capped at 365 days)
- **current_price**: Property sale price

---

## 3. Data Preprocessing

### 3.1 Data Cleaning

```python
# Column name normalization
df.columns = df.columns.str.lower()

# Outlier handling
train_df["days_on_market_capped"] = train_df["days_on_market"].clip(upper=365)
```

**Rationale**: Capping DOM at 365 days prevents extreme outliers from skewing the model.

### 3.2 Feature Engineering

#### Temporal Features
```python
train_df["listed_date"] = pd.to_datetime(train_df["listed_date"], errors="coerce")
train_df["listed_year"] = train_df["listed_date"].dt.year
train_df["listed_month"] = train_df["listed_date"].dt.month
train_df["listed_dayofweek"] = train_df["listed_date"].dt.dayofweek
```

**Rationale**: Temporal patterns (seasonality, day of week) significantly impact real estate sales.

### 3.3 Missing Value Imputation

```python
# Numeric columns: Median imputation
X[num_cols] = X[num_cols].fillna(X[num_cols].median())

# Categorical columns: "Unknown" category
X[cat_cols] = X[cat_cols].fillna("Unknown")
```

**Rationale**: 
- Median is robust to outliers for numeric features
- "Unknown" category preserves information about missingness

### 3.4 Feature Scaling and Encoding

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ]
)
```

**Rationale**:
- StandardScaler normalizes numeric features for consistent scale
- OneHotEncoder converts categorical variables to binary features
- `handle_unknown="ignore"` prevents errors on unseen categories

---

## 4. Data Leakage Prevention

### 4.1 Excluded Features

The following columns were explicitly excluded to prevent data leakage:

```python
drop_cols = [
    "days_on_market",           # Target variable (raw)
    "days_on_market_capped",    # Target variable (processed)
    "current_price",            # Target variable for price model
    "listing_id",               # Identifier (no predictive value)
    "address",                  # Too specific (overfitting risk)
    "street",                   # Too specific (overfitting risk)
    "mls_number",               # Identifier
    "agent_id",                 # Post-listing information
    "office_id",                # Post-listing information
    "removed_date",             # Future information (leakage)
    "last_seen_ts",             # Future information (leakage)
    "status",                   # Post-listing information
]
```

### 4.2 Temporal Validation

- **Train/Test Split**: 80/20 random split
- **Random State**: 42 (reproducibility)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Note**: For production, time-based splitting should be considered to respect temporal ordering.

### 4.3 Feature Availability Check

All features used in the model are available **at the time of listing**:
- Property characteristics (bedrooms, bathrooms, etc.)
- Location data (city, county, coordinates)
- Temporal features (listing date)

**No future information** is used in predictions.

---

## 5. Model Selection

### 5.1 Algorithms Tested

We evaluated 10 different algorithms:

| Algorithm | Type | DOM MAE | DOM R² |
|-----------|------|---------|--------|
| **ExtraTrees** | Ensemble | **36.63** | 0.584 |
| **RandomForest** | Ensemble | 36.65 | **0.617** |
| **HistGradientBoosting** | Ensemble | 39.57 | 0.612 |
| KNeighbors | Instance-based | 40.27 | 0.555 |
| GradientBoosting | Ensemble | 43.76 | 0.591 |
| DecisionTree | Tree | 44.17 | 0.346 |
| Ridge | Linear | 44.88 | 0.574 |
| Lasso | Linear | 58.52 | 0.431 |
| AdaBoost | Ensemble | 65.77 | 0.360 |
| ElasticNet | Linear | 73.61 | 0.201 |

### 5.2 Why ExtraTrees?

**Selected Algorithm**: Extra Trees Regressor

**Reasons**:

1. **Best MAE Performance**: 36.63 days (13.5% better than baseline 42.38 days)

2. **Handles Non-linearity**: Real estate pricing is highly non-linear
   - Location effects
   - Property size interactions
   - Seasonal patterns

3. **Robust to Outliers**: Randomized splits reduce overfitting

4. **Feature Interactions**: Automatically captures complex interactions
   - Example: Price per sq ft varies by location and property type

5. **No Feature Scaling Required**: Tree-based models are scale-invariant

6. **Fast Prediction**: Parallel prediction across trees

7. **Handles Mixed Data Types**: Works well with categorical and numeric features

**Comparison with RandomForest**:
- RandomForest had slightly better R² (0.617 vs 0.584)
- ExtraTrees had marginally better MAE (36.63 vs 36.65)
- ExtraTrees trains faster due to random splits
- **Decision**: Chose ExtraTrees for best MAE (primary metric)

---

## 6. Model Architecture

### 6.1 Pipeline Structure

```python
Pipeline(steps=[
    ("preprocessor", ColumnTransformer([
        ("num", StandardScaler(), numeric_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ])),
    ("model", ExtraTreesRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])
```

### 6.2 Hyperparameters

**Current Configuration** (default parameters):
- `n_estimators=200`: Number of trees in the forest
- `random_state=42`: Reproducibility
- `n_jobs=-1`: Use all CPU cores
- Other parameters: sklearn defaults

**Note**: Hyperparameter tuning was not performed in this iteration. Future work should explore:
- `max_depth`: Tree depth limit
- `min_samples_split`: Minimum samples to split node
- `min_samples_leaf`: Minimum samples in leaf node
- `max_features`: Features to consider for splits

---

## 7. Model Performance

### 7.1 Days on Market (DOM) Model

**Metrics**:
- **MAE (Mean Absolute Error)**: 36.63 days
  - On average, predictions are off by ±36.63 days
  - **Improvement**: 13.5% better than baseline (42.38 days)

- **RMSE (Root Mean Squared Error)**: 64.79 days
  - Penalizes larger errors more heavily
  - Indicates some predictions have larger deviations

- **R² (Coefficient of Determination)**: 0.584
  - Model explains 58.4% of variance in DOM
  - Moderate predictive power

**Interpretation**:
- Model performs well for typical properties
- Less accurate for unusual properties with extended market time
- Suitable for risk assessment and general guidance

### 7.2 Price Model

**Metrics**:
- **MAE**: $71,818
  - Average prediction error
  
- **RMSE**: $474,601
  - Higher RMSE indicates some large errors (expensive properties)

- **R²**: 0.795
  - Model explains 79.5% of price variance
  - Strong predictive power

**Interpretation**:
- Excellent performance for price prediction
- Captures most price-determining factors
- Suitable for investment analysis

### 7.3 Error Analysis

**DOM Model Error Distribution**:
```
Actual vs Predicted scatter plot shows:
- Good predictions for DOM < 120 days
- Underestimation for very long DOM (>200 days)
- Slight tendency to predict toward mean
```

**Recommendations**:
- Use with caution for luxury/unique properties
- Combine with domain expertise for edge cases
- Consider ensemble with other models for critical decisions

---

## 8. Feature Importance

### 8.1 Top Features (Expected)

Based on domain knowledge and model type:

**For DOM**:
1. **price_per_sq_ft**: Overpriced properties stay longer
2. **county/city**: Location is critical
3. **property_type**: Different types have different markets
4. **listed_month**: Seasonal effects
5. **square_footage**: Size affects marketability

**For Price**:
1. **square_footage**: Primary price driver
2. **county/city**: Location, location, location
3. **bedrooms/bathrooms**: Key property features
4. **year_built**: Age affects value
5. **lot_size**: Land value component

**Note**: Actual feature importance can be extracted using:
```python
model.named_steps['model'].feature_importances_
```

---

## 9. Model Validation

### 9.1 Cross-Validation

**Current**: Single train/test split (80/20)

**Recommendation**: Implement k-fold cross-validation
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, X, y, 
    cv=5, 
    scoring='neg_mean_absolute_error'
)
```

### 9.2 Temporal Validation

**Current**: Random split (may include future data in training)

**Recommendation**: Time-based split
```python
# Sort by listing date
df_sorted = df.sort_values('listed_date')

# Use last 20% as test set
split_idx = int(len(df_sorted) * 0.8)
train = df_sorted[:split_idx]
test = df_sorted[split_idx:]
```

**Rationale**: Prevents training on future data, more realistic evaluation

---

## 10. Model Deployment

### 10.1 Saved Artifacts

```
models/
├── dom_model.joblib           # DOM prediction pipeline
├── price_model.joblib          # Price prediction pipeline
└── feature_columns.joblib      # Feature column order
```

### 10.2 Prediction Interface

```python
from app.models import load_models, predict_dom, predict_price

# Load models
dom_model, price_model, feature_cols = load_models()

# Prepare input
X_pred = prepare_prediction_input(input_row, feature_cols)

# Predict
predicted_dom = predict_dom(dom_model, X_pred)
predicted_price = predict_price(price_model, X_pred)
```

### 10.3 Production Considerations

**Monitoring**:
- Track prediction accuracy over time
- Monitor for data drift
- Log predictions for analysis

**Retraining**:
- Retrain quarterly with new data
- Update when market conditions change significantly
- Version control for models

**Error Handling**:
- Validate input data
- Handle missing features gracefully
- Provide confidence intervals

---

## 11. Limitations and Assumptions

### 11.1 Limitations

1. **Temporal Assumptions**: Assumes market conditions remain similar to training period

2. **Geographic Coverage**: Limited to areas in training data

3. **Property Types**: May not generalize to rare property types

4. **Market Events**: Cannot predict impact of major economic events

5. **Data Quality**: Dependent on accuracy of input data

6. **Feature Completeness**: Missing features (e.g., property condition, renovations)

### 11.2 Assumptions

1. **IID Data**: Assumes training and test data are from same distribution

2. **Feature Availability**: All features available at prediction time

3. **No Manipulation**: Assumes honest reporting of property features

4. **Market Stability**: Assumes no major market disruptions

---

## 12. Future Improvements

### 12.1 Model Enhancements

1. **Hyperparameter Tuning**: GridSearchCV or RandomizedSearchCV
   ```python
   param_grid = {
       'model__n_estimators': [200, 300, 500],
       'model__max_depth': [None, 20, 30],
       'model__min_samples_split': [2, 5, 10]
   }
   ```

2. **Ensemble Stacking**: Combine multiple models
   ```python
   from sklearn.ensemble import StackingRegressor
   
   estimators = [
       ('rf', RandomForestRegressor()),
       ('et', ExtraTreesRegressor()),
       ('gb', HistGradientBoostingRegressor())
   ]
   ```

3. **Feature Engineering**:
   - Neighborhood-level aggregations
   - Market trend indicators
   - Property age categories
   - Price-to-income ratios

4. **Advanced Algorithms**:
   - XGBoost
   - LightGBM
   - CatBoost
   - Neural networks for complex patterns

### 12.2 Data Improvements

1. **Additional Features**:
   - Property condition/quality ratings
   - Recent renovations
   - School district ratings
   - Crime statistics
   - Walkability scores

2. **External Data**:
   - Economic indicators
   - Interest rates
   - Local employment data
   - Population trends

3. **Image Data**:
   - Property photos (CNN features)
   - Aerial imagery
   - Street view data

### 12.3 Validation Improvements

1. **Time-based Cross-Validation**
2. **Stratified Sampling** by property type and location
3. **Confidence Intervals** for predictions
4. **Prediction Intervals** for uncertainty quantification

---

## 13. Conclusion

### 13.1 Summary

We successfully developed machine learning models for real estate prediction:

- **DOM Model**: Predicts days on market with MAE of 36.63 days
- **Price Model**: Predicts property prices with R² of 0.795
- **Algorithm**: ExtraTrees Regressor chosen for best performance
- **Deployment**: Integrated into Streamlit dashboard for real-time predictions

### 13.2 Model Reliability

**Are the results accurate?**

**Yes, with caveats**:

✅ **Strengths**:
- 13.5% improvement over baseline
- Strong R² scores (0.584 for DOM, 0.795 for price)
- Tested on 10 different algorithms
- Proper train/test split
- Data leakage prevention measures

⚠️ **Limitations**:
- Moderate R² for DOM (58.4%) - room for improvement
- Some large errors for unusual properties
- No hyperparameter tuning yet
- Single train/test split (should use cross-validation)
- No temporal validation

**Recommendation**: 
- **Suitable for**: Investment screening, risk assessment, general guidance
- **Not suitable for**: Critical financial decisions without expert review
- **Use with**: Domain expertise and market knowledge

### 13.3 Next Steps

1. Implement hyperparameter tuning
2. Add cross-validation
3. Perform temporal validation
4. Monitor production performance
5. Retrain with new data quarterly

---

## Appendix A: Code References

### Training Script
- `train_models.py`: Main training script
- `app/preprocessing.py`: Data preprocessing functions
- `app/models.py`: Model loading and prediction

### Notebooks
- `notebooks/01_eda.ipynb`: Exploratory data analysis
- `notebooks/02_feature_engineering.ipynb`: Feature engineering
- `notebooks/03_model_training.ipynb`: Model training experiments

### Documentation
- `MODEL_TRAINING_SUMMARY.md`: Training results summary
- `FEATURE_FUTURE_DATES.md`: Future date prediction feature
- `TROUBLESHOOTING.md`: Common issues and solutions

---

## Appendix B: Reproducibility

### Environment
```
Python: 3.14
scikit-learn: 1.8.0
pandas: latest
numpy: latest
```

### Random Seeds
```python
random_state = 42  # Used throughout for reproducibility
```

### Data Version
```
Snapshot Date: 2026-02-01
Records: 13,428
File: clean_sales_listings_20260201_134434.csv
```

---

**Document Version**: 1.0  
**Last Updated**: February 10, 2026  
**Author**: Data Science Team  
**Review Status**: Initial Release
