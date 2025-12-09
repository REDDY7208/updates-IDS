from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Load model artifacts
print("Loading model artifacts...")
model = joblib.load('models/ids_model.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
feature_names = joblib.load('models/feature_names.pkl')
print(f"✓ Model loaded with {len(label_encoder.classes_)} attack classes")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict-file', methods=['POST'])
def predict_file():
    """Upload CSV file and get predictions for all rows"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Read CSV
        df = pd.read_csv(file)
        original_df = df.copy()
        
        # Handle missing columns
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        
        # Select only required features
        df = df[feature_names]
        
        # Clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # Predict
        X_scaled = scaler.transform(df)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        # Decode labels
        predicted_labels = label_encoder.inverse_transform(predictions)
        
        # Get confidence scores
        confidence_scores = np.max(probabilities, axis=1)
        
        # Count attacks
        attack_counts = pd.Series(predicted_labels).value_counts().to_dict()
        
        # Prepare results
        results = []
        for i in range(min(100, len(df))):  # Return first 100 rows
            results.append({
                'row': i + 1,
                'prediction': predicted_labels[i],
                'confidence': float(confidence_scores[i] * 100)
            })
        
        return jsonify({
            'success': True,
            'total_rows': len(df),
            'attack_summary': attack_counts,
            'predictions': results,
            'attack_classes': list(label_encoder.classes_)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-live', methods=['POST'])
def predict_live():
    """Real-time prediction for hardware integration (single row JSON)"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Handle missing columns
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        
        # Select only required features
        df = df[feature_names]
        
        # Clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # Predict
        X_scaled = scaler.transform(df)
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        
        # Decode label
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        confidence = float(np.max(probabilities) * 100)
        
        # Get top 3 predictions
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        top_3_predictions = [
            {
                'attack': label_encoder.inverse_transform([idx])[0],
                'probability': float(probabilities[idx] * 100)
            }
            for idx in top_3_idx
        ]
        
        return jsonify({
            'success': True,
            'prediction': predicted_label,
            'confidence': confidence,
            'is_attack': predicted_label != 'Normal',
            'top_predictions': top_3_predictions,
            'timestamp': pd.Timestamp.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'attack_classes': list(label_encoder.classes_),
        'total_classes': len(label_encoder.classes_),
        'features_count': len(feature_names),
        'model_type': 'XGBoost Classifier'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
