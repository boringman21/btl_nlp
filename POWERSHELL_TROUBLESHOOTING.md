# PowerShell Troubleshooting Guide for btl_nlp

This guide addresses common issues when setting up and running the btl_nlp package in PowerShell on Windows.

## ModuleNotFoundError: No module named 'matplotlib' (or other packages)

If you encounter errors about missing packages after activating your virtual environment, follow these steps:

### Solution 1: Manual package installation

```powershell
# Activate your virtual environment
.\venv\Scripts\Activate.ps1

# Install the missing packages
pip install matplotlib pandas numpy seaborn psutil scikit-learn
```

### Solution 2: Fix execution policy issues

If you encounter execution policy restrictions:

```powershell
# Run PowerShell as Administrator, then execute:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then try activating the environment again
.\venv\Scripts\Activate.ps1
```

### Solution 3: Use PowerShell script for setup

For a more comprehensive setup, run the provided PowerShell script:

```powershell
# Run with execution policy bypass
powershell.exe -ExecutionPolicy Bypass -File ./setup_env.ps1
```

## Running Python with 'python3' instead of 'python'

If your system uses `python3` instead of `python`:

```powershell
# Replace 'python' with 'python3' in commands
python3 -m venv venv
python3 run_sample.py
```

## Activation Issues in PowerShell

If you see "Activate.ps1 cannot be loaded because running scripts is disabled on this system":

```powershell
# Method 1: Bypass for just this script
powershell -ExecutionPolicy Bypass -File .\venv\Scripts\Activate.ps1

# Method 2: Change policy for current session only
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1
```

## Check if packages are installed correctly

To verify package installation:

```powershell
# Activate your virtual environment first
.\venv\Scripts\Activate.ps1

# Check installed packages
pip list

# Test specific imports
python -c "import matplotlib; print('matplotlib installed')"
python -c "import pandas; print('pandas installed')"
python -c "import btl_nlp; print('btl_nlp installed')"
```

## Use cmd.exe as an alternative

If PowerShell continues to give problems, you can use the standard Command Prompt:

```cmd
:: Open cmd.exe and run:
venv\Scripts\activate.bat
pip install -r requirements.txt
pip install -e .
python run_sample.py
```

## Full reinstallation

If all else fails, try removing and recreating the virtual environment:

```powershell
# Remove the existing virtual environment
Remove-Item -Recurse -Force venv

# Create a new one
python -m venv venv

# Activate and install
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
``` 