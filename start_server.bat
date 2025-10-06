@echo off
echo 🚀 ExoML Backend Server Startup Script
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python first.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv\" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo 📚 Installing/Updating dependencies...
pip install -q -r requirements.txt

REM Check if model files exist
if not exist "model_files\exoplanet_bilstm.h5" (
    echo ❌ Error: model_files\exoplanet_bilstm.h5 not found!
    pause
    exit /b 1
)

if not exist "model_files\scaler.pkl" (
    echo ❌ Error: model_files\scaler.pkl not found!
    pause
    exit /b 1
)

if not exist "metadata.pkl" (
    echo ❌ Error: metadata.pkl not found!
    pause
    exit /b 1
)

echo ✅ All model files found!
echo.
echo 🎯 Starting Flask server on http://127.0.0.1:5001
echo 💡 Open exo.html in a browser and upload a CSV file to test!
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the Flask app
python app.py

