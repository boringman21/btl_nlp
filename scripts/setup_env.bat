@echo off
echo ====================================
echo   Setting up environment for water_leakage
echo ====================================
echo.

REM Check Python
python --version 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Python not found. Please install Python and ensure it's added to PATH.
    exit /b 1
)

echo Starting virtual environment setup...
echo.

REM Create virtual environment if it doesn't exist
if not exist venv\ (
    echo Creating new virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo.
echo Installing required packages...
pip install -r requirements.txt

REM Install water_leakage package in development mode
echo.
echo Installing water_leakage package...
pip install -e .

echo.
echo Checking installation...
python -c "import water_leakage; print('water_leakage version:', water_leakage.__version__)"

echo.
echo =========================================
echo   Setup complete! 
echo   - Virtual environment is activated
echo   - water_leakage package is installed
echo.
echo   To run sample analysis:
echo   python run_sample.py
echo ========================================= 