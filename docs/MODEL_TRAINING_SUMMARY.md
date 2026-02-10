# Model Training Summary

## Date: February 10, 2026

## Objective
Improve the Days on Market (DOM) prediction model performance from baseline MAE of 42.38 days.

## Approach
Tested 10 different machine learning algorithms without hyperparameter tuning to identify the best performers for the real estate dataset.

## Algorithms Tested

### Tree-Based Ensemble Methods
1. **ExtraTrees** (Extra Trees Regressor)
2. **RandomForest** (Random Forest Regressor)
3. **GradientBoosting** (Gradient Boosting Regressor)
4. **HistGradientBoosting** (Histogram-based Gradient Boosting)
5. **AdaBoost** (Adaptive Boosting)
6. **DecisionTree** (Single Decision Tree)

### Linear Models
7. **Ridge** (Ridge Regression)
8. **Lasso** (Lasso Regression)
9. **ElasticNet** (Elastic Net Regression)

### Instance-Based
10. **KNeighbors** (K-Nearest Neighbors)

## Results - Days on Market (DOM) Model

### Top 3 Performers (Ranked by MAE)

| Rank | Algorithm | MAE (days) | RMSE (days) | R² Score |
|------|-----------|------------|-------------|----------|
| 🥇 1 | **ExtraTrees** | **36.63** | 64.79 | 0.584 |
| 🥈 2 | **RandomForest** | 36.65 | 62.15 | **0.617** |
| 🥉 3 | **HistGradientBoosting** | 39.57 | 62.55 | 0.612 |

### Key Achievement
- **13.5% improvement** in MAE (from 42.38 to 36.63 days)
- RandomForest shows best R² score (0.617), explaining 61.7% of variance

### Full Results

| Algorithm | MAE | RMSE | R² |
|-----------|-----|------|-----|
| ExtraTrees | 36.63 | 64.79 | 0.584 |
| RandomForest | 36.65 | 62.15 | 0.617 |
| HistGradientBoosting | 39.57 | 62.55 | 0.612 |
| KNeighbors | 40.27 | 67.02 | 0.555 |
| GradientBoosting | 43.76 | 64.24 | 0.591 |
| DecisionTree | 44.17 | 81.21 | 0.346 |
| Ridge | 44.88 | 65.56 | 0.574 |
| Lasso | 58.52 | 75.79 | 0.431 |
| AdaBoost | 65.77 | 80.36 | 0.360 |
| ElasticNet | 73.61 | 89.80 | 0.201 |

## Results - Price Model

### Top 3 Performers

| Rank | Algorithm | MAE ($) | RMSE ($) | R² Score |
|------|-----------|---------|----------|----------|
| 🥇 1 | **ExtraTrees** | **$71,818** | $474,601 | **0.795** |
| 🥈 2 | **RandomForest** | $74,528 | $492,116 | 0.780 |
| 🥉 3 | **HistGradientBoosting** | $113,358 | $577,393 | 0.697 |

## Key Findings

### 1. Tree-Based Ensembles Dominate
- ExtraTrees and RandomForest significantly outperform all other algorithms
- Both achieve similar MAE but RandomForest has better R² score
- These models handle non-linear relationships in real estate data effectively

### 2. Linear Models Perform Poorly
- Ridge, Lasso, and ElasticNet show poor performance
- Real estate pricing is highly non-linear
- Not suitable for this dataset

### 3. Gradient Boosting Methods
- HistGradientBoosting shows promise (3rd place)
- Faster training than traditional GradientBoosting
- Good balance of speed and accuracy

### 4. KNeighbors Surprise
- Achieved 4th place (MAE: 40.27 days)
- However, slower for predictions at scale
- Not recommended for production

## Feature Engineering Applied

1. **Date Features**: Extracted year, month, day of week from listing date
2. **Outlier Handling**: Capped days_on_market at 365 days
3. **Missing Value Imputation**: 
   - Numeric: Median imputation
   - Categorical: "Unknown" category
4. **Scaling**: StandardScaler for numeric features
5. **Encoding**: OneHotEncoder for categorical features

## Dataset Statistics

- **Total Records**: 13,428 properties
- **Features**: 18 total
  - Numeric: 10 features
  - Categorical: 5 features (unit, city, county, property_type, listing_type)
- **Train/Test Split**: 80/20

## Recommendations for Next Steps

### Immediate Actions
1. **Deploy Current Models**: ExtraTrees models are production-ready
2. **Monitor Performance**: Track predictions vs actual outcomes

### Future Improvements
1. **Hyperparameter Tuning**: Focus on top 3 algorithms
   - RandomForest (best R² score)
   - ExtraTrees (best MAE)
   - HistGradientBoosting (fast, modern)

2. **Advanced Techniques**:
   - Feature selection (remove low-importance features)
   - Ensemble stacking (combine top models)
   - Cross-validation for robust evaluation
   - Time-based validation (respect temporal nature)

3. **Feature Engineering**:
   - Add neighborhood-level aggregations
   - Include market trend indicators
   - Create interaction features
   - Add seasonal indicators

## Model Files Saved

- `models/dom_model.joblib` - Days on Market model (ExtraTrees)
- `models/price_model.joblib` - Price prediction model (ExtraTrees)
- `models/feature_columns.joblib` - Feature column order

## Technical Notes

- **Training Time**: ~5 minutes for all 10 algorithms
- **Model Size**: ~250MB each (due to ensemble size)
- **Dependencies**: scikit-learn 1.8.0, pandas, numpy, joblib
- **Python Version**: 3.14

## Conclusion

Successfully improved DOM prediction by 13.5% using ExtraTrees algorithm. The model is now production-ready and integrated into the Streamlit dashboard for real-time investment risk analysis.

**Current Performance**: MAE = 36.63 days, R² = 0.584
**Target for Tuning**: MAE < 35 days, R² > 0.65
