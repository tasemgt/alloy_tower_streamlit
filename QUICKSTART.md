# Quick Start

## Run the App

### Option 1: Using Launcher Script (Recommended)

**PowerShell:**
```powershell
.\start_app.ps1
```

**Command Prompt:**
```cmd
start_app.bat
```

### Option 2: Manual Start

```bash
# Activate virtual environment first
.venv\Scripts\activate

# Run with Python module
python -m streamlit run run_app.py
```

**Important:** Always use the virtual environment to avoid "No module named 'sklearn'" errors.

## Two Main Views

### 1. Investment Analyzer
- Enter property details (location, features, etc.)
- **Select future listing date** to see timing impact
- Get risk assessment (🔴 High / 🟡 Medium / 🟢 Low)
- See predicted price and days on market
- View potential return on investment
- Get seasonal and timing insights

### 2. Market Dashboard
- Filter properties by location, type, price
- View market KPIs and trends
- Explore geographic risk heatmap
- Analyze high-risk listings

## Tips

- **Future Dates** - Select different listing dates to see how timing affects predictions
- **Seasonal Insights** - Spring/summer typically see faster sales
- **Filters apply everywhere** - Use sidebar filters to narrow down data
- **Hover on map** - See property details on the geographic heatmap
- **Risk colors** - 🟢 Green (fast selling), 🟡 Yellow (moderate), 🔴 Red (slow selling)
- **Pagination** - Use page selector to browse large datasets

## Troubleshooting

**"No module named 'sklearn'" error?**
1. Make sure virtual environment is activated: `.venv\Scripts\activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Use launcher scripts: `.\start_app.ps1` or `start_app.bat`
4. Run with: `python -m streamlit run run_app.py` (not just `streamlit run`)

**No data?**
- Ensure `ml/data/clean_sales_listings_*.csv` exists
- Run data ingestion: `python ingest_data.py`

**Models not found?**
- Train models: `python train_models.py`
- Check `models/` folder contains `.joblib` files

**Map not working?**
- Install: `pip install pydeck`
