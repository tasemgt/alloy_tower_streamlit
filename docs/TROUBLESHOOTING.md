# Troubleshooting Guide

## Issue: "No module named 'sklearn'" Error

### Problem
When running the Streamlit app, you see:
```
Error loading models: No module named 'sklearn'
Models not found: No module named 'sklearn'
```

### Root Cause
The joblib library needs scikit-learn to unpickle (load) the trained models. Even though sklearn is installed, it needs to be explicitly imported before loading the models.

### Solution Applied

Updated `app/models.py` to explicitly import sklearn modules:

```python
# Import sklearn to ensure it's available when unpickling models
try:
    import sklearn
    import sklearn.ensemble
    import sklearn.pipeline
    import sklearn.compose
    import sklearn.preprocessing
except ImportError as e:
    st.error(f"scikit-learn not found: {e}")
    st.info("Please install scikit-learn: pip install scikit-learn")
    raise
```

### Verification Steps

1. **Check sklearn is installed**:
   ```bash
   pip list | findstr scikit
   ```
   Should show: `scikit-learn 1.8.0`

2. **Test model loading**:
   ```bash
   python -c "from app.models import load_models; load_models(); print('Success!')"
   ```

3. **Run test app**:
   ```bash
   streamlit run test_model_loading.py
   ```

4. **Run main app**:
   ```bash
   streamlit run run_app.py
   ```

### If Issue Persists

1. **Reinstall scikit-learn**:
   ```bash
   pip uninstall scikit-learn
   pip install scikit-learn==1.8.0
   ```

2. **Verify virtual environment is activated**:
   ```bash
   # Should see (.venv) in prompt
   # If not, activate it:
   .venv\Scripts\activate
   ```

3. **Check Python version**:
   ```bash
   python --version
   ```
   Should be Python 3.14 or compatible

4. **Reinstall all requirements**:
   ```bash
   pip install -r requirements.txt
   ```

### Alternative: Retrain Models

If models are corrupted or incompatible:

```bash
python train_models.py
```

This will:
- Test 10 different algorithms
- Save new models to `models/` directory
- Take ~5 minutes to complete

## Other Common Issues

### Issue: Models Not Found

**Error**: `FileNotFoundError: DOM model not found at: models/dom_model.joblib`

**Solution**: Train the models first:
```bash
python train_models.py
```

### Issue: Data Not Found

**Error**: `No listings data found`

**Solution**: Run data ingestion:
```bash
python ingest_data.py
```

### Issue: Streamlit Won't Start

**Error**: Port already in use

**Solution**: Kill existing Streamlit process or use different port:
```bash
streamlit run run_app.py --server.port 8503
```

### Issue: Import Errors

**Error**: `ModuleNotFoundError: No module named 'app'`

**Solution**: Run from project root directory:
```bash
cd C:\Development\Mwenya\alloy-tower-real-estate-ml
streamlit run run_app.py
```

## Getting Help

If issues persist:
1. Check the error message carefully
2. Verify all files are in correct locations
3. Ensure virtual environment is activated
4. Try retraining models
5. Check Python and package versions match requirements
