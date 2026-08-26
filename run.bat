@echo off
setlocal enabledelayedexpansion

:: Automatically switch to script's directory
cd /d "%~dp0"

echo ============================================================
echo  SIF Precursor Detection Engine - Oil India Limited
echo ============================================================
echo.

:: Detect Python
set "PYTHON_EXE=python"
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not found in PATH.
    echo Please install Python 3.10+ and ensure "Add python.exe to PATH" is checked.
    pause
    exit /b 1
)

:: Add Windows App Python User Script paths to PATH
set "PATH=%LOCALAPPDATA%\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts;%APPDATA%\Python\Python313\Scripts;%LOCALAPPDATA%\Programs\Python\Python313\Scripts;%PATH%"

echo [1/3] Checking core dependencies...
python -m pip install pandas numpy scikit-learn sentence-transformers streamlit plotly openpyxl xlsxwriter --quiet --no-warn-script-location

echo [2/3] Generating demo dataset (500 synthetic OIL reports)...
python -c "import sys; sys.path.insert(0, '.'); from data.synthetic_reports import generate_dataset; df=generate_dataset(500); df.to_csv('data/sample_reports.csv', index=False); print(f'  Dataset ready: {len(df)} reports')"

echo [3/3] Launching Streamlit dashboard...
echo.
echo ============================================================
echo  Live Dashboard is opening at: http://localhost:8501
echo  Press Ctrl+C in this window to stop the server.
echo ============================================================
echo.

:: Open browser automatically
start http://localhost:8501

:: Run Streamlit via Python module to avoid PATH issues
python -m streamlit run app\dashboard.py --server.port 8501 --server.headless false --browser.gatherUsageStats false

pause
