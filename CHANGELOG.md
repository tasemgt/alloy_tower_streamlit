# Changelog

All notable changes to the AlloyTower Real Estate ML project.

## [1.0.0] - 2026-02-10

### Added

#### Machine Learning Models
- Implemented ExtraTrees Regressor for Days on Market prediction (MAE: 36.63 days, R²: 0.584)
- Implemented ExtraTrees Regressor for Price prediction (MAE: $71,818, R²: 0.795)
- Tested 10 different algorithms and selected best performer
- Added proper data leakage prevention measures
- Implemented feature engineering (temporal features)
- Added StandardScaler for numeric features
- Added OneHotEncoder for categorical features

#### Features
- **Investment Risk Analyzer**: Predict property values and market time
- **Future Date Predictions**: Select listing date to see timing impact
- **Market Dashboard**: Interactive visualizations and KPIs
- **Geographic Risk Heatmap**: Visual risk assessment by location
- **Seasonal Insights**: Timing recommendations based on selected date
- **Risk Assessment**: Comprehensive risk scoring (High/Medium/Low)

#### Documentation
- `docs/MODEL_DOCUMENTATION.md`: Comprehensive model documentation
  - Problem statement and business value
  - Dataset description (18 features)
  - Data preprocessing steps
  - Data leakage prevention measures
  - Model selection rationale
  - Performance metrics and analysis
  - Limitations and assumptions
  - Future improvements
- `docs/MODEL_TRAINING_SUMMARY.md`: Training results summary
- `docs/FEATURE_FUTURE_DATES.md`: Future date prediction feature docs
- `docs/TROUBLESHOOTING.md`: Common issues and solutions
- `docs/RUN_APP_INSTRUCTIONS.md`: Detailed setup instructions
- `README.md`: Comprehensive project README
- `QUICKSTART.md`: Quick start guide
- `CHANGELOG.md`: This file

#### Code Organization
- Modular application structure in `app/` directory
- Separated concerns: models, preprocessing, data I/O, UI components
- Added proper docstrings following Google Python style guide
- Added inline comments for complex logic
- Created utility scripts: `start_app.ps1`, `start_app.bat`
- Added `.gitignore` for proper version control

#### Testing
- Created test scripts for model validation
- Verified future date predictions
- Tested data type handling

### Changed

#### Code Quality
- Refactored monolithic code into modular structure
- Applied Google Python style guide
- Added comprehensive docstrings
- Improved error handling
- Fixed data type issues in preprocessing
- Enhanced input validation

#### Data Processing
- Normalized column names to lowercase
- Implemented proper missing value handling
- Added outlier capping (DOM at 365 days)
- Improved categorical feature handling
- Fixed mixed data type issues

#### User Interface
- Enhanced Investment Analyzer with date picker
- Added seasonal and timing insights
- Improved risk visualization
- Added pagination to tables (10 items per page)
- Created geographic heatmap with risk overlay
- Organized charts into analytical tabs

### Fixed

- **sklearn Import Error**: Added explicit sklearn imports in models.py
- **Data Type Mismatch**: Fixed mixed float/string in categorical columns
- **Prediction Errors**: Proper type casting in input preparation
- **Missing Features**: Added all required features (unit, price_per_sq_ft)
- **Slider Error**: Added buffer logic for min/max value equality
- **Tooltip Display**: Fixed geographic heatmap tooltip data

### Removed

- Duplicate prediction pages
- Unnecessary documentation files
- Test scripts (moved to separate testing directory)
- Redundant code blocks

### Security

- Excluded Snowflake credentials from version control
- Added `.gitignore` for sensitive files
- Removed hardcoded credentials from documentation

## [0.1.0] - Initial Development

### Added
- Basic Streamlit application
- Initial model training script
- Data ingestion from Snowflake
- Basic visualizations

---

## Upgrade Guide

### From 0.1.0 to 1.0.0

1. **Update Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Retrain Models**:
   ```bash
   python train_models.py
   ```

3. **Update Configuration**:
   - Review `app/constants.py` for any custom configurations
   - Update Snowflake credentials in `ingest_data.py` if needed

4. **Run Application**:
   ```bash
   .\start_app.ps1  # or python -m streamlit run run_app.py
   ```

---

## Future Releases

### Planned for 1.1.0
- Hyperparameter tuning for models
- Cross-validation implementation
- Time-based validation
- Confidence intervals for predictions
- Model monitoring dashboard

### Planned for 1.2.0
- XGBoost and LightGBM models
- Ensemble stacking
- Feature importance visualization
- A/B testing framework
- Automated retraining pipeline

### Planned for 2.0.0
- Deep learning models
- Image-based property valuation
- Real-time market data integration
- API for external integrations
- Mobile-responsive design

---

## Contributing

See [README.md](README.md) for contribution guidelines.

## License

[Add license information]
