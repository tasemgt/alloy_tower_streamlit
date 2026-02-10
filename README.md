# AlloyTower Real Estate ML Dashboard

A machine learning-powered real estate investment analysis platform that predicts property values and days on market to assess investment risk.

## 🎯 Features

- **Investment Risk Analyzer**: Predict future property values and market time
- **Market Dashboard**: Interactive visualizations and market insights
- **Geographic Risk Heatmap**: Visual risk assessment by location
- **Future Date Predictions**: Plan optimal listing timing
- **Real-time Predictions**: ML-powered property valuation

## 📊 Model Performance

- **Days on Market**: MAE = 36.63 days, R² = 0.584
- **Price Prediction**: MAE = $71,818, R² = 0.795
- **Algorithm**: ExtraTrees Regressor (13.5% improvement over baseline)

## 🚀 Quick Start

### Prerequisites

- Python 3.14+
- Virtual environment recommended

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/alloy-tower-real-estate-ml.git
cd alloy-tower-real-estate-ml

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Running the Dashboard

**Option 1: Using Launcher (Recommended)**
```powershell
.\start_app.ps1  # PowerShell
start_app.bat    # Command Prompt
```

**Option 2: Manual Start**
```bash
python -m streamlit run run_app.py
```

The dashboard will open at `http://localhost:8501`

## 📁 Project Structure

```
alloy-tower-real-estate-ml/
├── app/                      # Application modules
│   ├── app.py               # Main app entry point
│   ├── dashboard.py         # Market dashboard view
│   ├── investment.py        # Investment analyzer view
│   ├── models.py            # Model loading and prediction
│   ├── preprocessing.py     # Data preprocessing
│   ├── data_io.py          # Data loading utilities
│   ├── filters.py          # Filter components
│   ├── plots.py            # Visualization functions
│   ├── risk.py             # Risk analysis
│   ├── ui_components.py    # UI helper components
│   └── constants.py        # Configuration constants
├── docs/                    # Documentation
│   ├── MODEL_DOCUMENTATION.md
│   ├── MODEL_TRAINING_SUMMARY.md
│   ├── FEATURE_FUTURE_DATES.md
│   ├── TROUBLESHOOTING.md
│   └── RUN_APP_INSTRUCTIONS.md
├── ml/data/                 # Training data
├── models/                  # Trained models
│   ├── dom_model.joblib
│   ├── price_model.joblib
│   └── feature_columns.joblib
├── notebooks/               # Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── reports/                 # Generated reports
├── train_models.py         # Model training script
├── ingest_data.py          # Data ingestion script
├── run_app.py              # App launcher
├── requirements.txt        # Python dependencies
├── QUICKSTART.md          # Quick start guide
└── README.md              # This file
```

## 🔧 Training Models

To retrain models with new data:

```bash
python train_models.py
```

This will:
- Test 10 different algorithms
- Select the best performer
- Save models to `models/` directory
- Take approximately 5 minutes

## 📚 Documentation

- **[Quick Start Guide](QUICKSTART.md)**: Get started quickly
- **[Model Documentation](docs/MODEL_DOCUMENTATION.md)**: Comprehensive model details
- **[Training Summary](docs/MODEL_TRAINING_SUMMARY.md)**: Training results
- **[Future Dates Feature](docs/FEATURE_FUTURE_DATES.md)**: Timing predictions
- **[Troubleshooting](docs/TROUBLESHOOTING.md)**: Common issues
- **[Run Instructions](docs/RUN_APP_INSTRUCTIONS.md)**: Detailed setup

## 🛠️ Technology Stack

- **ML Framework**: scikit-learn 1.8.0
- **Web Framework**: Streamlit
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, pydeck
- **Database**: Snowflake
- **Language**: Python 3.14

## 📈 Model Details

### Algorithm Selection

After testing 10 algorithms, **ExtraTrees Regressor** was selected for:
- Best MAE performance (36.63 days)
- Robust handling of non-linear relationships
- Excellent performance with mixed data types
- Fast training and prediction

### Features Used (18 total)

**Categorical**: unit, city, county, property_type, listing_type  
**Numeric**: zip_code, latitude, longitude, bedrooms, bathrooms, square_footage, lot_size, year_built, hoa_fee, price_per_sq_ft  
**Temporal**: listed_year, listed_month, listed_dayofweek

### Data Leakage Prevention

- Excluded post-listing information (status, removed_date)
- Excluded identifiers (listing_id, agent_id, office_id)
- Only used features available at listing time
- Proper train/test split (80/20)

## 🎓 Usage Examples

### Investment Analysis

1. Navigate to "Investment Analyzer"
2. Enter property details
3. Select expected listing date
4. Set investment amount
5. Click "Analyze Investment Risk"
6. Review predictions and risk assessment

### Market Dashboard

1. Navigate to "Market Dashboard"
2. Use sidebar filters to narrow data
3. Explore KPIs and visualizations
4. Hover on geographic heatmap for details
5. Review high-risk listings

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

[Add your license here]

## 👥 Authors

Data Science Team - AlloyTower

## 🙏 Acknowledgments

- Snowflake for data infrastructure
- scikit-learn community
- Streamlit team

## 📞 Support

For issues or questions:
- Check [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
- Review [Documentation](docs/)
- Open an issue on GitHub

---

**Version**: 1.0.0  
**Last Updated**: February 10, 2026
