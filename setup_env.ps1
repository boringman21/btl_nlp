# PowerShell script to set up environment for water_leakage
Write-Host "====================================" -ForegroundColor Green
Write-Host "  Setting up environment for water_leakage" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Host "Using Python: $pythonVersion" -ForegroundColor Green
}
catch {
    Write-Host "Python not found. Please install Python and ensure it's added to PATH." -ForegroundColor Red
    exit 1
}

# Create and activate virtual environment
Write-Host "Starting virtual environment setup..." -ForegroundColor Cyan
Write-Host ""

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating new virtual environment..." -ForegroundColor Cyan
    python -m venv venv
}
else {
    Write-Host "Virtual environment already exists." -ForegroundColor Cyan
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

# Make sure pip is up to date
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Install dependencies
Write-Host ""
Write-Host "Installing required packages explicitly..." -ForegroundColor Cyan
pip install pandas numpy matplotlib seaborn psutil scikit-learn jupyterlab

# Also install from requirements.txt
Write-Host ""
Write-Host "Installing packages from requirements.txt..." -ForegroundColor Cyan
pip install -r requirements.txt

# Install water_leakage package in development mode
Write-Host ""
Write-Host "Installing water_leakage package..." -ForegroundColor Cyan
pip install -e .

# Verify installation
Write-Host ""
Write-Host "Checking installation..." -ForegroundColor Cyan
try {
    python -c "import water_leakage; print('water_leakage version:', water_leakage.__version__)"
    python -c "import matplotlib; print('matplotlib version:', matplotlib.__version__)"
    python -c "import pandas; print('pandas version:', pandas.__version__)"
    python -c "import numpy; print('numpy version:', numpy.__version__)"
    python -c "import seaborn; print('seaborn version:', seaborn.__version__)"
    Write-Host "All packages are installed correctly!" -ForegroundColor Green
}
catch {
    Write-Host "Error during verification: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "  - Virtual environment is activated" -ForegroundColor Green
Write-Host "  - water_leakage package is installed" -ForegroundColor Green
Write-Host ""
Write-Host "  To run sample analysis:" -ForegroundColor Yellow
Write-Host "  python run_sample.py" -ForegroundColor Yellow
Write-Host "===========================================" 