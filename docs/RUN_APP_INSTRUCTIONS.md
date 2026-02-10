# How to Run the AlloyTower Real Estate Dashboard

## The Problem

You may encounter this error when running the app:
```
ModuleNotFoundError: No module named 'sklearn'
```

This happens because Streamlit might use a different Python installation than your virtual environment.

## The Solution

### ✅ Method 1: Use the Launcher Scripts (EASIEST)

We've created launcher scripts that ensure the correct Python environment is used.

**For PowerShell (Recommended):**
```powershell
.\start_app.ps1
```

**For Command Prompt:**
```cmd
start_app.bat
```

These scripts will:
1. Activate the virtual environment
2. Check if dependencies are installed
3. Start Streamlit with the correct Python

### ✅ Method 2: Manual Start (RELIABLE)

```powershell
# Step 1: Activate virtual environment
.venv\Scripts\activate

# Step 2: Verify sklearn is installed
python -c "import sklearn; print('sklearn OK')"

# Step 3: Run Streamlit using Python module
python -m streamlit run run_app.py
```

**Key Point:** Use `python -m streamlit` instead of just `streamlit` to ensure it uses the venv Python.

### ❌ Method 3: Direct Streamlit (MAY FAIL)

```powershell
streamlit run run_app.py
```

This might fail because `streamlit` command may use a different Python installation.

## Verification Steps

Before running the app, verify your setup:

### 1. Check Virtual Environment

```powershell
# Your prompt should show (.venv)
# If not, activate it:
.venv\Scripts\activate
```

### 2. Check Python Location

```powershell
where.exe python
# First line should be: C:\Development\Mwenya\alloy-tower-real-estate-ml\.venv\Scripts\python.exe
```

### 3. Check sklearn Installation

```powershell
python -c "import sklearn; print(sklearn.__version__)"
# Should print: 1.8.0
```

### 4. Check Models Exist

```powershell
dir models\*.joblib
# Should show: dom_model.joblib, price_model.joblib, feature_columns.joblib
```

## If You Still Get Errors

### Error: "No module named 'sklearn'"

**Solution 1:** Reinstall dependencies in venv
```powershell
.venv\Scripts\activate
pip install --force-reinstall scikit-learn
```

**Solution 2:** Recreate virtual environment
```powershell
# Deactivate current venv
deactivate

# Remove old venv
Remove-Item -Recurse -Force .venv

# Create new venv
python -m venv .venv

# Activate it
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Error: "Models not found"

**Solution:** Train the models
```powershell
python train_models.py
```

This will take ~5 minutes and create the model files.

### Error: "No data found"

**Solution:** Run data ingestion
```powershell
python ingest_data.py
```

## Quick Reference

| Task | Command |
|------|---------|
| **Start App (Easy)** | `.\start_app.ps1` |
| **Start App (Manual)** | `python -m streamlit run run_app.py` |
| **Train Models** | `python train_models.py` |
| **Get Data** | `python ingest_data.py` |
| **Install Deps** | `pip install -r requirements.txt` |
| **Check sklearn** | `python -c "import sklearn"` |

## Understanding the Issue

The error occurs because:

1. **Multiple Python Installations**: Windows may have Python installed in multiple locations:
   - Virtual environment: `.venv\Scripts\python.exe`
   - User installation: `C:\Users\...\AppData\Roaming\Python\Python314`
   - System installation: `C:\Python314\python.exe`

2. **Streamlit Command**: When you run `streamlit run`, it might use a different Python than your venv

3. **Module Location**: sklearn is installed in `.venv` but Streamlit is looking in a different Python

## The Fix

By using `python -m streamlit` instead of just `streamlit`, we explicitly tell Python to:
1. Use the current Python interpreter (from venv)
2. Run the streamlit module from that Python's packages
3. This ensures sklearn is found because it's in the same Python installation

## Success Indicators

When the app starts successfully, you'll see:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.3.161:8501
```

And the browser will open showing the dashboard without errors.

## Need More Help?

See `TROUBLESHOOTING.md` for additional solutions.
