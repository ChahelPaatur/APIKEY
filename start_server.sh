#!/bin/bash

echo "🚀 ExoML Backend Server Startup Script"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📚 Installing/Updating dependencies..."
pip install -q -r requirements.txt

# Check if model files exist
if [ ! -f "model_files/exoplanet_bilstm.h5" ]; then
    echo "❌ Error: model_files/exoplanet_bilstm.h5 not found!"
    exit 1
fi

if [ ! -f "model_files/scaler.pkl" ]; then
    echo "❌ Error: model_files/scaler.pkl not found!"
    exit 1
fi

if [ ! -f "metadata.pkl" ]; then
    echo "❌ Error: metadata.pkl not found!"
    exit 1
fi

echo "✅ All model files found!"
echo ""
echo "🎯 Starting Flask server on http://127.0.0.1:5001"
echo "💡 Open http://127.0.0.1:5001 in a browser and upload a CSV file to test!"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the Flask app
python3 app.py

