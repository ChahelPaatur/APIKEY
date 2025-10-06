from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import os
import secrets
import hashlib
from functools import wraps
from datetime import datetime

app = Flask(__name__)

# ===== API KEY MANAGEMENT =====
API_KEYS_FILE = 'api_keys.txt'

def load_api_keys():
    """Load API keys from file"""
    if os.path.exists(API_KEYS_FILE):
        with open(API_KEYS_FILE, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def save_api_key(api_key):
    """Save new API key to file"""
    with open(API_KEYS_FILE, 'a') as f:
        f.write(f"{api_key}\n")

def generate_api_key():
    """Generate a new secure API key"""
    return secrets.token_urlsafe(32)

# Load existing keys
VALID_API_KEYS = load_api_keys()

# Generate initial key if none exist
if not VALID_API_KEYS:
    initial_key = generate_api_key()
    save_api_key(initial_key)
    VALID_API_KEYS.add(initial_key)
    print(f"\n{'='*70}")
    print(f"INITIAL API KEY GENERATED:")
    print(f"{'='*70}")
    print(f"{initial_key}")
    print(f"{'='*70}")
    print(f"Save this key! Use it in your requests as 'X-API-Key' header")
    print(f"{'='*70}\n")

def require_api_key(f):
    """Decorator to require API key for endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            return jsonify({
                'error': 'Missing API key',
                'message': 'Include X-API-Key header in your request'
            }), 401
        
        if api_key not in VALID_API_KEYS:
            return jsonify({
                'error': 'Invalid API key',
                'message': 'Your API key is not valid'
            }), 403
        
        return f(*args, **kwargs)
    
    return decorated_function

# ===== LOAD MODEL =====
print("Loading model and preprocessing tools...")

MODEL_PATH = 'model_files/exoplanet_bilstm.h5'
SCALER_PATH = 'model_files/scaler.pkl'
METADATA_PATH = 'model_files/metadata.pkl'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training script first!")

model = keras.models.load_model(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
print(f"Scaler loaded from {SCALER_PATH}")

with open(METADATA_PATH, 'rb') as f:
    metadata = pickle.load(f)
print(f"Metadata loaded: {metadata}")

# ===== HELPER FUNCTIONS =====
def preprocess_input(data):
    """Preprocess input data for model prediction"""
    data = np.array(data, dtype=np.float32)
    
    if len(data.shape) == 1:
        data = data.reshape(1, -1, 1)
    elif len(data.shape) == 2:
        if data.shape[0] == 1:
            data = data.reshape(1, data.shape[1], 1)
        else:
            data = data.reshape(1, data.shape[0], data.shape[1])
    
    original_shape = data.shape
    data_flat = data.reshape(-1, data.shape[-1])
    data_scaled = scaler.transform(data_flat)
    data_scaled = data_scaled.reshape(original_shape)
    
    return data_scaled

# ===== PUBLIC ENDPOINTS (No API key required) =====
@app.route('/', methods=['GET'])
def home():
    """API info endpoint"""
    return jsonify({
        'service': 'Exoplanet Detection API',
        'model': 'BiLSTM Hybrid',
        'version': '2.0',
        'authentication': 'Required (X-API-Key header)',
        'accuracy': f"{metadata['test_accuracy']*100:.2f}%",
        'endpoints': {
            '/': 'GET - API information (public)',
            '/health': 'GET - Health check (public)',
            '/predict': 'POST - Predict exoplanet (requires API key)',
            '/predict_batch': 'POST - Batch predictions (requires API key)',
            '/generate_key': 'POST - Generate new API key (requires master key)'
        },
        'usage': 'Include "X-API-Key: your_key_here" in request headers'
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'timestamp': datetime.utcnow().isoformat()
    })

# ===== PROTECTED ENDPOINTS (API key required) =====
@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    """
    Main prediction endpoint (API key required)
    
    Headers:
        X-API-Key: your_api_key_here
    
    Body (JSON):
    {
        "data": [array of flux values],
        "return_probabilities": true/false (optional)
    }
    
    Returns:
    {
        "prediction": 0 or 1,
        "label": "No Exoplanet" or "Exoplanet Detected",
        "confidence": 0.0 to 1.0,
        "probabilities": {...} (if requested)
    }
    """
    try:
        json_data = request.get_json()
        
        if not json_data or 'data' not in json_data:
            return jsonify({
                'error': 'Missing required field: data',
                'example': {
                    'data': [1.0, 0.998, 0.995, '...']
                }
            }), 400
        
        input_data = json_data['data']
        return_probs = json_data.get('return_probabilities', False)
        
        if not isinstance(input_data, (list, np.ndarray)):
            return jsonify({'error': 'Data must be array or list'}), 400
        
        if len(input_data) == 0:
            return jsonify({'error': 'Data cannot be empty'}), 400
        
        processed_data = preprocess_input(input_data)
        prediction_probs = model.predict(processed_data, verbose=0)
        prediction_class = int(np.argmax(prediction_probs[0]))
        confidence = float(prediction_probs[0][prediction_class])
        
        response = {
            'prediction': prediction_class,
            'label': 'Exoplanet Detected' if prediction_class == 1 else 'No Exoplanet',
            'confidence': confidence,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if return_probs:
            response['probabilities'] = {
                'no_planet': float(prediction_probs[0][0]),
                'planet': float(prediction_probs[0][1])
            }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500

@app.route('/predict_batch', methods=['POST'])
@require_api_key
def predict_batch():
    """
    Batch prediction endpoint (API key required)
    
    Headers:
        X-API-Key: your_api_key_here
    
    Body (JSON):
    {
        "data": [[array1], [array2], ...],
        "return_probabilities": true/false (optional)
    }
    """
    try:
        json_data = request.get_json()
        
        if not json_data or 'data' not in json_data:
            return jsonify({'error': 'Missing required field: data'}), 400
        
        batch_data = json_data['data']
        return_probs = json_data.get('return_probabilities', False)
        
        if not isinstance(batch_data, list):
            return jsonify({'error': 'Data must be list of arrays'}), 400
        
        results = []
        for i, sample in enumerate(batch_data):
            try:
                processed = preprocess_input(sample)
                probs = model.predict(processed, verbose=0)
                pred_class = int(np.argmax(probs[0]))
                confidence = float(probs[0][pred_class])
                
                result = {
                    'index': i,
                    'prediction': pred_class,
                    'label': 'Exoplanet Detected' if pred_class == 1 else 'No Exoplanet',
                    'confidence': confidence
                }
                
                if return_probs:
                    result['probabilities'] = {
                        'no_planet': float(probs[0][0]),
                        'planet': float(probs[0][1])
                    }
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e)
                })
        
        return jsonify({
            'total': len(batch_data),
            'results': results,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500

@app.route('/generate_key', methods=['POST'])
def generate_key():
    """
    Generate new API key (requires master key)
    
    Body (JSON):
    {
        "master_key": "your_master_key"
    }
    """
    try:
        json_data = request.get_json()
        master_key = json_data.get('master_key')
        
        # Check if any key exists (first key is master)
        if not master_key or master_key not in VALID_API_KEYS:
            return jsonify({
                'error': 'Invalid or missing master key'
            }), 403
        
        # Generate new key
        new_key = generate_api_key()
        save_api_key(new_key)
        VALID_API_KEYS.add(new_key)
        
        return jsonify({
            'message': 'New API key generated',
            'api_key': new_key,
            'created_at': datetime.utcnow().isoformat()
        }), 201
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

# ===== ERROR HANDLERS =====
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# ===== RUN SERVER =====
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "="*70)
    print("EXOPLANET DETECTION API SERVER")
    print("="*70)
    print(f"Model: BiLSTM Hybrid")
    print(f"Accuracy: {metadata['test_accuracy']*100:.2f}%")
    print(f"Authentication: API Key Required (X-API-Key header)")
    print(f"\nAPI keys stored in: {API_KEYS_FILE}")
    print(f"Total active keys: {len(VALID_API_KEYS)}")
    print(f"\nStarting server on port {port}...")
    print("="*70 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=port)

