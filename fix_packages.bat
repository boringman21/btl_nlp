@echo off
echo ===================================
echo  Installing missing dependencies
echo ===================================

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing matplotlib and other required packages...
pip install matplotlib pandas numpy seaborn psutil scikit-learn jupyterlab

echo.
echo Ensuring all requirements are installed...
pip install -r requirements.txt

echo.
echo Checking installation...
python -c "import matplotlib; print('matplotlib is installed')"

echo.
echo ===================================
echo  Dependencies fixed!
echo  You can now run: python run_sample.py
echo =================================== 