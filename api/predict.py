import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
import io
import os
import json
from http.server import BaseHTTPRequestHandler

# Global variables for model caching
_model = None
_scaler = None
_metadata = None

def get_model_path(filename):
    """Get the correct path for model files"""
    # Try different path configurations for Vercel
    base_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'model_files'),
        os.path.join(os.getcwd(), 'model_files'),
        'model_files',
        '..'
    ]
    
    for base in base_paths:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    
    # If not found, return the default path
    return os.path.join('model_files', filename)

def load_models():
    """Load model and preprocessing objects (with caching)"""
    global _model, _scaler, _metadata
    
    if _model is None:
        print("Loading model and preprocessing objects...")
        try:
            model_path = get_model_path('exoplanet_bilstm.h5')
            scaler_path = get_model_path('scaler.pkl')
            
            print(f"Loading model from: {model_path}")
            _model = keras.models.load_model(model_path)
            
            print(f"Loading scaler from: {scaler_path}")
            with open(scaler_path, 'rb') as f:
                _scaler = pickle.load(f)
            
            # Try to load metadata if it exists
            try:
                metadata_path = get_model_path('metadata.pkl')
                with open(metadata_path, 'rb') as f:
                    _metadata = pickle.load(f)
            except:
                _metadata = {}
            
            print("Model loaded successfully!")
            print(f"Model input shape: {_model.input_shape}")
            print(f"Model output shape: {_model.output_shape}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    return _model, _scaler, _metadata

def preprocess_data(csv_data, scaler):
    """Preprocess CSV data for the BiLSTM model"""
    # Read CSV data
    df = pd.read_csv(io.StringIO(csv_data))
    
    # If the CSV has a target column, remove it for prediction
    if 'LABEL' in df.columns:
        df = df.drop('LABEL', axis=1)
    
    # Get numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Handle missing values
    numeric_df = numeric_df.fillna(numeric_df.mean())
    
    # Scale the data
    scaled_data = scaler.transform(numeric_df)
    
    # Reshape for BiLSTM (samples, timesteps, features)
    if len(scaled_data.shape) == 2:
        scaled_data = scaled_data.reshape((scaled_data.shape[0], scaled_data.shape[1], 1))
    
    return scaled_data, numeric_df

def estimate_planet_properties(features_df, confidence):
    """Estimate planet properties based on the features"""
    props = {}
    
    # Try to extract meaningful features if they exist
    if 'orbital_period' in features_df.columns:
        props['orbitalPeriod'] = float(features_df['orbital_period'].iloc[0])
    elif len(features_df.columns) > 0:
        props['orbitalPeriod'] = float(np.abs(features_df.iloc[0, 0]) * 10)
    else:
        props['orbitalPeriod'] = 5.3
    
    if 'temperature' in features_df.columns:
        props['temperature'] = float(features_df['temperature'].iloc[0])
    else:
        props['temperature'] = 1200 + (confidence * 10)
    
    if 'transit_depth' in features_df.columns:
        props['transitDepth'] = float(features_df['transit_depth'].iloc[0])
    else:
        props['transitDepth'] = 0.001
    
    # Determine planet type based on properties
    if props['temperature'] > 1500:
        props['planetType'] = 'Hot Jupiter'
    elif props['temperature'] > 800:
        props['planetType'] = 'Super Earth'
    elif props['orbitalPeriod'] < 10:
        props['planetType'] = 'Close-in Planet'
    else:
        props['planetType'] = 'Neptune-like'
    
    return props

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        """Handle POST requests"""
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            csv_data = data.get('data', '')
            
            if not csv_data:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = {'error': 'No data provided'}
                self.wfile.write(json.dumps(response).encode())
                return
            
            # Load models
            model, scaler, metadata = load_models()
            
            # Preprocess the data
            processed_data, features_df = preprocess_data(csv_data, scaler)
            
            # Make prediction
            predictions = model.predict(processed_data, verbose=0)
            
            # Calculate average prediction and confidence
            avg_prediction = float(np.mean(predictions))
            confidence = avg_prediction * 100
            
            # Determine if exoplanet is detected
            is_exoplanet = avg_prediction > 0.5
            
            # Estimate planet properties
            properties = estimate_planet_properties(features_df, confidence)
            
            # Prepare response
            response = {
                'isExoplanet': bool(is_exoplanet),
                'confidence': round(confidence, 2),
                'probability': float(avg_prediction),
                'prediction': 1 if is_exoplanet else 0,
                'orbitalPeriod': round(properties['orbitalPeriod'], 2),
                'temperature': round(properties['temperature'], 2),
                'transitDepth': properties['transitDepth'],
                'planetType': properties['planetType'],
                'modelType': 'BiLSTM Neural Network',
                'numSamples': len(processed_data)
            }
            
            print(f"Prediction made: {response}")
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {'error': str(e)}
            self.wfile.write(json.dumps(response).encode())
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

