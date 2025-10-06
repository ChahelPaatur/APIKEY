import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and preprocessing objects
print("Loading model and preprocessing objects...")
model = keras.models.load_model('model_files/exoplanet_bilstm.h5')
with open('model_files/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

print("Model loaded successfully!")
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")
print(f"Metadata: {metadata}")

def preprocess_data(csv_data):
    """
    Preprocess CSV data for the BiLSTM model
    """
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
    # Assuming each row is a sample, we need to reshape appropriately
    # For time series, we might need to reshape based on sequence length
    if len(scaled_data.shape) == 2:
        # If it's 2D (samples, features), reshape to (samples, timesteps, features)
        # We'll treat each feature as a timestep
        scaled_data = scaled_data.reshape((scaled_data.shape[0], scaled_data.shape[1], 1))
    
    return scaled_data, numeric_df

def estimate_planet_properties(features_df, confidence):
    """
    Estimate planet properties based on the features
    """
    # Get some basic statistics from the data
    props = {}
    
    # Try to extract meaningful features if they exist
    if 'orbital_period' in features_df.columns:
        props['orbitalPeriod'] = float(features_df['orbital_period'].iloc[0])
    elif len(features_df.columns) > 0:
        # Use first feature as proxy for orbital period
        props['orbitalPeriod'] = float(np.abs(features_df.iloc[0, 0]) * 10)
    else:
        props['orbitalPeriod'] = 5.3
    
    if 'temperature' in features_df.columns:
        props['temperature'] = float(features_df['temperature'].iloc[0])
    else:
        # Estimate temperature based on confidence
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

@app.route('/', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    """
    try:
        # Get CSV data from request
        data = request.json
        csv_data = data.get('data', '')
        
        if not csv_data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Preprocess the data
        processed_data, features_df = preprocess_data(csv_data)
        
        # Make prediction
        predictions = model.predict(processed_data, verbose=0)
        
        # Calculate average prediction and confidence
        avg_prediction = np.mean(predictions)
        confidence = float(avg_prediction * 100)
        
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
        return jsonify(response)
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'model': 'loaded',
        'modelType': 'BiLSTM'
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 ExoML Server Starting...")
    print("="*50)
    print("Model: BiLSTM Exoplanet Detection")
    print("Server: http://127.0.0.1:5001")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5001, debug=True)

