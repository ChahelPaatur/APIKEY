#!/bin/bash

echo "🚀 ExoML Vercel Deployment Script"
echo "=================================="
echo ""

# Check if vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI is not installed."
    echo ""
    echo "Install it with:"
    echo "  npm install -g vercel"
    echo ""
    exit 1
fi

echo "✅ Vercel CLI found"
echo ""

# Check if model files exist
if [ ! -f "model_files/exoplanet_bilstm.h5" ]; then
    echo "❌ Error: model_files/exoplanet_bilstm.h5 not found!"
    exit 1
fi

if [ ! -f "model_files/scaler.pkl" ]; then
    echo "❌ Error: model_files/scaler.pkl not found!"
    exit 1
fi

echo "✅ All model files found"
echo ""

# Check model size
MODEL_SIZE=$(du -h model_files/exoplanet_bilstm.h5 | cut -f1)
echo "📊 Model size: $MODEL_SIZE"
echo ""
echo "⚠️  Note: Vercel has a 250MB limit for serverless functions."
echo "   If deployment fails, consider model optimization."
echo ""

# Ask for confirmation
read -p "Deploy to Vercel? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

echo ""
echo "📦 Deploying to Vercel..."
echo ""

# Deploy to Vercel
vercel --prod

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Next steps:"
echo "1. Test the health endpoint: https://your-app.vercel.app/api/health"
echo "2. Open your app: https://your-app.vercel.app"
echo "3. Upload a CSV file to test the ML model"
echo ""

