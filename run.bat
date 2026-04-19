@echo off
echo ========================================================
echo   ML Reliability Platform - Startup Script
echo ========================================================
echo.

echo [1/3] Installing/verifying dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo ❌ ERROR: Failed to install Python dependencies. Please ensure Python is installed and accessible in your PATH.
    pause
    exit /b %errorlevel%
)
echo ✅ Dependencies installed successfully.
echo.

echo [2/3] Generating demo data...
python wipe.py
python scripts\generate_demo_data.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ ERROR: Failed to generate demo data.
    pause
    exit /b %errorlevel%
)
echo ✅ Demo data created.
echo.

echo [3/3] Starting the local production server...
echo 🌍 Once started, open your browser and navigate to: http://127.0.0.1:8000
echo.
echo Press CTRL+C to stop the server at any time.
echo --------------------------------------------------------

uvicorn serving.api:app --reload

pause
