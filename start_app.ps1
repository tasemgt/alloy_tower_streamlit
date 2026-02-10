# AlloyTower Real Estate Dashboard Launcher
Write-Host "Starting AlloyTower Real Estate Dashboard..." -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please create it first: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& .venv\Scripts\Activate.ps1

# Check if sklearn is installed
Write-Host "Checking dependencies..." -ForegroundColor Green
$sklearn = python -c "import sklearn; print('OK')" 2>&1
if ($sklearn -ne "OK") {
    Write-Host "ERROR: scikit-learn not found in virtual environment!" -ForegroundColor Red
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

# Start Streamlit
Write-Host ""
Write-Host "Starting Streamlit app..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""
python -m streamlit run run_app.py
