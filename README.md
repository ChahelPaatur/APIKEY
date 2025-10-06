# ExoML - NASA Exoplanet Detection Platform

A beautiful web application for detecting exoplanets using a BiLSTM neural network trained on NASA data. Features real-time ML predictions, interactive visualizations, and Firebase authentication.

![Model Size: 5.1MB](https://img.shields.io/badge/Model%20Size-5.1MB-green)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)

## ✨ Features

- 🔭 **ML-Powered Detection**: BiLSTM neural network trained on NASA exoplanet data
- 📊 **Interactive Visualizations**: Real-time charts with Chart.js
- 🔐 **Firebase Authentication**: Secure user login/signup
- 💾 **Cloud Storage**: Save analysis history to Firestore
- 🎨 **Beautiful UI**: Modern space-themed design with animations
- 🚀 **Production Ready**: Deploy to Vercel with one command
- 📱 **Responsive Design**: Works on desktop and mobile

---

## 🚀 Quick Start

### Deploy to Vercel (Production) ⭐

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
./deploy_to_vercel.sh
# Or: vercel --prod
```

Your app will be live at `https://your-app.vercel.app` in minutes!

### Run Locally (Development)

```bash
# Mac/Linux
./start_server.sh

# Windows
start_server.bat

# Or manually
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5001` and upload a CSV file.

---

## 📁 Project Structure

```
HTML-for-ExoML/
├── index.html              # Frontend web app
├── app.py                  # Flask backend (local dev)
├── api/
│   ├── predict.py          # Vercel serverless function
│   └── health.py           # Health check endpoint
├── model_files/
│   ├── exoplanet_bilstm.h5 # BiLSTM model (5.1MB)
│   ├── scaler.pkl          # Data preprocessor
│   └── metadata.pkl        # Model metadata
├── vercel.json             # Vercel config
├── requirements.txt        # Python dependencies
├── start_server.sh         # Local dev script (Mac/Linux)
├── start_server.bat        # Local dev script (Windows)
└── deploy_to_vercel.sh     # Deploy script
```

---

## 🔌 API Endpoints

### POST `/` or `/api/predict`
Accepts CSV data and returns exoplanet predictions.

**Request:**
```json
{
  "data": "FLUX.1,FLUX.2,FLUX.3...\n93.85,83.81,20.10..."
}
```

**Response:**
```json
{
  "isExoplanet": true,
  "confidence": 85.5,
  "probability": 0.855,
  "prediction": 1,
  "orbitalPeriod": 5.3,
  "temperature": 1200.5,
  "transitDepth": 0.001,
  "planetType": "Hot Jupiter",
  "modelType": "BiLSTM Neural Network",
  "numSamples": 1
}
```

### GET `/health` or `/api/health`
Health check endpoint.

---

## 🛠️ How It Works

1. **Upload CSV**: User uploads exoplanet transit data
2. **Preprocessing**: Data is scaled using saved scaler
3. **ML Prediction**: BiLSTM model processes the time-series data
4. **Results**: Confidence score, planet type, and orbital metrics displayed
5. **Save History**: Results stored in Firebase Firestore

---

## 📝 Deployment Notes

- **Model Size**: 5.1MB (well within Vercel's 250MB limit)
- **Auto-Detection**: Frontend automatically uses correct API endpoint (local vs Vercel)
- **CORS**: Enabled for all origins
- **Memory**: Configured for 3008MB on Vercel
- **Timeout**: 60 seconds max execution time

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)

---

## 🔧 Environment

The app automatically detects whether it's running locally or on Vercel:
- **Local**: Uses `http://127.0.0.1:5001`
- **Vercel**: Uses `/api/predict` endpoint

You can override this in Settings → API Endpoint.

---

## 📄 License

MIT License - feel free to use this project for your own exoplanet detection needs!
