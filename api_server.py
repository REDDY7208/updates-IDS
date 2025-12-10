from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
import uvicorn
from typing import Dict, List, Optional
import json
from datetime import datetime
import asyncio
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="IDS Real-Time Detection API",
    description="Real-time Intrusion Detection System API for IoT hardware integration",
    version="1.0.0"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model artifacts
model = None
scaler = None
label_encoder = None
feature_names = None

# Pydantic models for API requests/responses
class NetworkFeatures(BaseModel):
    """
    Network traffic features for real-time prediction.
    IoT devices should send these features extracted from network packets.
    """
    # Core flow features
    flow_duration: Optional[float] = 0.0
    total_fwd_packets: Optional[float] = 0.0
    total_backward_packets: Optional[float] = 0.0
    total_length_fwd_packets: Optional[float] = 0.0
    total_length_bwd_packets: Optional[float] = 0.0
    
    # Packet size features
    fwd_packet_length_max: Optional[float] = 0.0
    fwd_packet_length_min: Optional[float] = 0.0
    fwd_packet_length_mean: Optional[float] = 0.0
    fwd_packet_length_std: Optional[float] = 0.0
    bwd_packet_length_max: Optional[float] = 0.0
    bwd_packet_length_min: Optional[float] = 0.0
    bwd_packet_length_mean: Optional[float] = 0.0
    bwd_packet_length_std: Optional[float] = 0.0
    
    # Flow statistics
    flow_bytes_s: Optional[float] = 0.0
    flow_packets_s: Optional[float] = 0.0
    flow_iat_mean: Optional[float] = 0.0
    flow_iat_std: Optional[float] = 0.0
    flow_iat_max: Optional[float] = 0.0
    flow_iat_min: Optional[float] = 0.0
    
    # Forward/Backward timing
    fwd_iat_total: Optional[float] = 0.0
    fwd_iat_mean: Optional[float] = 0.0
    fwd_iat_std: Optional[float] = 0.0
    fwd_iat_max: Optional[float] = 0.0
    fwd_iat_min: Optional[float] = 0.0
    bwd_iat_total: Optional[float] = 0.0
    bwd_iat_mean: Optional[float] = 0.0
    bwd_iat_std: Optional[float] = 0.0
    bwd_iat_max: Optional[float] = 0.0
    bwd_iat_min: Optional[float] = 0.0
    
    # Protocol flags
    fwd_psh_flags: Optional[float] = 0.0
    bwd_psh_flags: Optional[float] = 0.0
    fwd_urg_flags: Optional[float] = 0.0
    bwd_urg_flags: Optional[float] = 0.0
    fwd_header_length: Optional[float] = 0.0
    bwd_header_length: Optional[float] = 0.0
    
    # Additional features (can be extended)
    min_packet_length: Optional[float] = 0.0
    max_packet_length: Optional[float] = 0.0
    packet_length_mean: Optional[float] = 0.0
    packet_length_std: Optional[float] = 0.0
    
    # Hardware/Device info (optional)
    device_id: Optional[str] = "unknown"
    timestamp: Optional[str] = None

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    success: bool
    prediction: str
    confidence: float
    is_attack: bool
    threat_level: str
    top_predictions: List[Dict[str, float]]
    timestamp: str
    device_id: str
    processing_time_ms: float

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    total_classes: int
    features_count: int
    uptime: str

# Store recent predictions for monitoring
recent_predictions = []
MAX_RECENT_PREDICTIONS = 1000

def load_model_artifacts():
    """Load all model artifacts"""
    global model, scaler, label_encoder, feature_names
    
    try:
        logger.info("Loading model artifacts...")
        
        # Load Keras model
        model = load_model('praveen/final_model.keras')
        logger.info("✓ Keras model loaded")
        
        # Load preprocessors
        scaler = joblib.load('praveen/scaler.pkl')
        label_encoder = joblib.load('praveen/label_encoder.pkl')
        feature_names = joblib.load('praveen/feature_names.pkl')
        
        logger.info(f"✓ Model artifacts loaded successfully")
        logger.info(f"✓ Classes: {len(label_encoder.classes_)}")
        logger.info(f"✓ Features: {len(feature_names)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model artifacts: {e}")
        return False

def preprocess_features(features_dict: dict) -> np.ndarray:
    """Preprocess features for prediction"""
    # Create DataFrame with all required features
    df = pd.DataFrame([features_dict])
    
    # Handle missing columns by adding them with default value 0
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    
    # Select only required features in correct order
    df = df[feature_names]
    
    # Clean data
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    # Scale features
    X_scaled = scaler.transform(df)
    
    # Reshape for CNN-LSTM model [samples, features, 1]
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
    
    return X_reshaped

def get_threat_level(prediction: str) -> str:
    """Determine threat level based on attack type"""
    critical_attacks = ['Backdoor', 'Botnet', 'DDoS', 'Exploit']
    high_attacks = ['DoS', 'Brute-Force', 'Web Attack – XSS', 'Web Attack – Brute Force']
    medium_attacks = ['PortScan', 'Probe', 'Fuzzers', 'Shellcode']
    
    if prediction == 'Normal':
        return 'INFO'
    elif prediction in critical_attacks:
        return 'CRITICAL'
    elif prediction in high_attacks:
        return 'HIGH'
    elif prediction in medium_attacks:
        return 'MEDIUM'
    else:
        return 'LOW'

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model_artifacts()
    if not success:
        logger.error("Failed to load model artifacts on startup!")
    else:
        logger.info("API server ready for real-time predictions!")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "IDS Real-Time Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/api/predict-live",
            "health": "/api/health",
            "stats": "/api/stats"
        }
    }

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        total_classes=len(label_encoder.classes_) if label_encoder else 0,
        features_count=len(feature_names) if feature_names else 0,
        uptime="running"
    )

@app.post("/api/predict-live", response_model=PredictionResponse)
async def predict_live(features: NetworkFeatures):
    """
    Real-time prediction endpoint for IoT hardware integration.
    
    This endpoint accepts network traffic features from IoT devices
    and returns instant attack predictions.
    
    Usage for IoT devices:
    1. Extract network features from packets
    2. Send POST request with JSON payload
    3. Receive prediction response instantly
    """
    start_time = datetime.now()
    
    try:
        # Check if model is loaded
        if model is None or scaler is None or label_encoder is None:
            raise HTTPException(
                status_code=503, 
                detail="Model not loaded. Please check server logs."
            )
        
        # Convert Pydantic model to dict
        features_dict = features.dict()
        
        # Remove non-feature fields
        device_id = features_dict.pop('device_id', 'unknown')
        timestamp = features_dict.pop('timestamp', None)
        
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Preprocess features
        X_processed = preprocess_features(features_dict)
        
        # Make prediction
        predictions_proba = model.predict(X_processed, verbose=0)
        prediction_idx = np.argmax(predictions_proba, axis=1)[0]
        
        # Decode prediction
        predicted_label = label_encoder.inverse_transform([prediction_idx])[0]
        confidence = float(np.max(predictions_proba) * 100)
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions_proba[0])[-3:][::-1]
        top_predictions = [
            {
                "attack_type": label_encoder.inverse_transform([idx])[0],
                "probability": float(predictions_proba[0][idx] * 100)
            }
            for idx in top_3_idx
        ]
        
        # Determine threat level
        threat_level = get_threat_level(predicted_label)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Create response
        response = PredictionResponse(
            success=True,
            prediction=predicted_label,
            confidence=round(confidence, 2),
            is_attack=predicted_label != 'Normal',
            threat_level=threat_level,
            top_predictions=top_predictions,
            timestamp=timestamp,
            device_id=device_id,
            processing_time_ms=round(processing_time, 2)
        )
        
        # Store for monitoring (keep last 1000 predictions)
        recent_predictions.append({
            "timestamp": timestamp,
            "device_id": device_id,
            "prediction": predicted_label,
            "confidence": confidence,
            "threat_level": threat_level,
            "processing_time_ms": processing_time
        })
        
        if len(recent_predictions) > MAX_RECENT_PREDICTIONS:
            recent_predictions.pop(0)
        
        # Log prediction
        logger.info(f"Prediction: {predicted_label} ({confidence:.1f}%) - Device: {device_id} - Time: {processing_time:.1f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/stats")
async def get_stats():
    """Get real-time statistics"""
    if not recent_predictions:
        return {
            "total_predictions": 0,
            "attack_rate": 0,
            "avg_confidence": 0,
            "avg_processing_time": 0,
            "threat_distribution": {},
            "recent_predictions": []
        }
    
    total = len(recent_predictions)
    attacks = sum(1 for p in recent_predictions if p['prediction'] != 'Normal')
    attack_rate = (attacks / total) * 100 if total > 0 else 0
    
    avg_confidence = sum(p['confidence'] for p in recent_predictions) / total
    avg_processing_time = sum(p['processing_time_ms'] for p in recent_predictions) / total
    
    # Threat level distribution
    threat_dist = {}
    for p in recent_predictions:
        threat = p['threat_level']
        threat_dist[threat] = threat_dist.get(threat, 0) + 1
    
    return {
        "total_predictions": total,
        "attack_rate": round(attack_rate, 2),
        "avg_confidence": round(avg_confidence, 2),
        "avg_processing_time": round(avg_processing_time, 2),
        "threat_distribution": threat_dist,
        "recent_predictions": recent_predictions[-10:]  # Last 10 predictions
    }

@app.get("/api/model-info")
async def get_model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "CNN-LSTM Deep Learning",
        "attack_classes": list(label_encoder.classes_),
        "total_classes": len(label_encoder.classes_),
        "features_count": len(feature_names),
        "feature_names": feature_names[:20],  # First 20 features
        "input_shape": f"({len(feature_names)}, 1)",
        "model_size": "~15MB"
    }

# Example endpoint for testing
@app.post("/api/test-prediction")
async def test_prediction():
    """Test endpoint with sample data"""
    sample_features = NetworkFeatures(
        flow_duration=120000,
        total_fwd_packets=10,
        total_backward_packets=8,
        total_length_fwd_packets=800,
        total_length_bwd_packets=600,
        flow_bytes_s=11.67,
        flow_packets_s=0.15,
        device_id="test_device"
    )
    
    return await predict_live(sample_features)

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )