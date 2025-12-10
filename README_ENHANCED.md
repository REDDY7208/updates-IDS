# 🛡️ Enhanced IDS Real-Time Detection System

A professional-grade Intrusion Detection System with real-time API capabilities and enhanced UI for network security analysis.

## 🌟 Key Features

### 🎯 **Professional Detection Dashboard**
- **Modern UI Design**: Clean, professional interface similar to enterprise security tools
- **Enhanced File Upload**: Drag-and-drop CSV analysis with instant validation
- **Real-time Metrics**: Beautiful gradient cards showing key statistics
- **Interactive Visualizations**: Professional pie charts and bar graphs
- **Advanced Filtering**: Multi-level filtering by threat level, confidence, and status

### 🚀 **Real-Time API Integration**
- **FastAPI Backend**: High-performance API for IoT hardware integration
- **Sub-100ms Response**: Ultra-fast predictions for real-time monitoring
- **Hardware Ready**: ESP32/ESP8266, Raspberry Pi, Arduino support
- **JSON API**: RESTful endpoints for seamless integration
- **Live Statistics**: Real-time monitoring and analytics

### 🧠 **Advanced ML Model**
- **CNN-LSTM Architecture**: Deep learning for complex pattern recognition
- **92% Accuracy**: High-performance multi-class classification
- **19 Attack Types**: Comprehensive threat detection
- **Multi-Dataset Training**: CIC-IDS, UNSW-NB15, WSN-DS datasets

## 📊 Detection Capabilities

### Attack Types Detected:
- **🔴 Critical**: Backdoor, Botnet, DDoS, Exploit
- **🟠 High**: DoS, Brute-Force, Web Attacks
- **🟡 Medium**: PortScan, Probe, Fuzzers
- **🟢 Low**: Flooding, TDMA, Shellcode
- **🔵 Normal**: Legitimate network traffic

## 🚀 Quick Start

### 1. **Install Dependencies**
```bash
# Install Python packages
pip install -r requirements_api.txt
pip install -r requirements_streamlit.txt

# Or install all at once
pip install fastapi uvicorn streamlit plotly pandas numpy tensorflow scikit-learn joblib
```

### 2. **Start the Complete System**
```bash
# Option 1: Use the startup script (Recommended)
python start_system.py

# Option 2: Start manually
# Terminal 1 - API Server
python api_server.py

# Terminal 2 - Streamlit Dashboard
streamlit run streamlit_app.py
```

### 3. **Access the System**
- **📊 Dashboard**: http://localhost:8501
- **🔌 API Docs**: http://localhost:8000/docs
- **🧪 API Health**: http://localhost:8000/api/health

## 🎯 Using the Detection Dashboard

### **Step 1: Upload Network Data**
1. Open the dashboard at http://localhost:8501
2. Go to "🎯 Detection Results" tab
3. Upload your CSV file with network traffic data
4. Click "🚀 Start Analysis"

### **Step 2: View Results**
- **Key Metrics**: Total packets, attacks detected, attack types, confidence
- **Visual Analysis**: Interactive pie charts and bar graphs
- **Detailed Table**: Filterable results with threat levels
- **Download**: Export results as CSV

### **Step 3: Filter & Analyze**
- Filter by attack status (All/Attacks Only/Normal Only)
- Filter by threat level (Critical/High/Medium/Low/Safe)
- Set minimum confidence threshold
- View detailed threat analysis

## 🌐 Real-Time API Integration

### **API Endpoints**

#### **POST /api/predict-live**
Real-time prediction for IoT hardware integration.

**Request Example:**
```json
{
    "flow_duration": 120000,
    "total_fwd_packets": 10,
    "total_backward_packets": 8,
    "total_length_fwd_packets": 800,
    "total_length_bwd_packets": 600,
    "flow_bytes_s": 11.67,
    "flow_packets_s": 0.15,
    "device_id": "ESP32_001"
}
```

**Response Example:**
```json
{
    "success": true,
    "prediction": "Normal",
    "confidence": 98.5,
    "is_attack": false,
    "threat_level": "SAFE",
    "top_predictions": [
        {"attack_type": "Normal", "probability": 98.5},
        {"attack_type": "DoS", "probability": 1.2}
    ],
    "timestamp": "2024-12-10T10:30:00Z",
    "device_id": "ESP32_001",
    "processing_time_ms": 45.2
}
```

#### **GET /api/stats**
Get real-time system statistics.

#### **GET /api/health**
Check API server health and model status.

## 🔧 Hardware Integration

### **ESP32/ESP8266 Example**
```cpp
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

void sendPredictionRequest() {
    HTTPClient http;
    http.begin("http://your-server:8000/api/predict-live");
    http.addHeader("Content-Type", "application/json");
    
    // Create JSON payload
    String payload = "{";
    payload += "\"flow_duration\":" + String(flow_duration) + ",";
    payload += "\"total_fwd_packets\":" + String(fwd_packets) + ",";
    payload += "\"device_id\":\"ESP32_001\"";
    payload += "}";
    
    int httpResponseCode = http.POST(payload);
    
    if (httpResponseCode == 200) {
        String response = http.getString();
        // Parse JSON response
        DynamicJsonDocument doc(1024);
        deserializeJson(doc, response);
        
        String prediction = doc["prediction"];
        float confidence = doc["confidence"];
        bool isAttack = doc["is_attack"];
        
        Serial.println("Prediction: " + prediction);
        Serial.println("Confidence: " + String(confidence) + "%");
    }
    
    http.end();
}
```

### **Raspberry Pi Example**
```python
import requests
import json
from scapy.all import sniff

API_URL = "http://localhost:8000/api/predict-live"

def extract_features(packet):
    """Extract network features from packet"""
    return {
        "flow_duration": packet.time,
        "total_fwd_packets": 1,
        "packet_length_mean": len(packet),
        "device_id": "RPI_001"
    }

def send_prediction(features):
    """Send prediction request to API"""
    response = requests.post(API_URL, json=features)
    return response.json()

def packet_handler(packet):
    """Handle captured packets"""
    features = extract_features(packet)
    result = send_prediction(features)
    
    if result['is_attack']:
        print(f"🚨 ATTACK: {result['prediction']} ({result['confidence']:.1f}%)")
    else:
        print(f"✅ Normal traffic ({result['confidence']:.1f}%)")

# Start packet capture
sniff(prn=packet_handler, count=100)
```

## 🧪 Testing the System

### **Test API Functionality**
```bash
# Run comprehensive API tests
python test_api.py

# Test with sample data
curl -X POST "http://localhost:8000/api/predict-live" \
     -H "Content-Type: application/json" \
     -d '{
       "flow_duration": 120000,
       "total_fwd_packets": 10,
       "device_id": "TEST_DEVICE"
     }'
```

### **Test Dashboard**
1. Upload the provided `demo_data.csv` file
2. Click "🚀 Start Analysis"
3. Explore the interactive visualizations
4. Test filtering and download functionality

## 📈 Model Performance

### **Overall Metrics**
- **Accuracy**: 92.0%
- **Precision**: 92.0% (weighted avg)
- **Recall**: 92.0% (weighted avg)
- **F1-Score**: 92.0% (weighted avg)

### **Per-Class Performance**
| Attack Class | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| Normal | 99% | 99% | 99% |
| DDoS | 100% | 98% | 99% |
| Generic | 100% | 97% | 98% |
| Brute-Force | 97% | 97% | 97% |
| Flooding | 99% | 99% | 99% |

## 🔒 Security Features

### **Threat Level Classification**
- **CRITICAL**: Immediate action required (Backdoor, Botnet, DDoS)
- **HIGH**: Monitor closely (DoS, Brute-Force, Web Attacks)
- **MEDIUM**: Investigate (PortScan, Probe, Fuzzers)
- **LOW**: Log and monitor (Flooding, TDMA)
- **SAFE**: Normal traffic

### **Real-Time Alerts**
- Instant threat detection (<100ms)
- Confidence scoring for reliability
- Device-specific monitoring
- Historical analysis and trends

## 📁 File Structure

```
├── streamlit_app.py          # Enhanced dashboard UI
├── api_server.py             # FastAPI backend
├── train_model.py            # Model training script
├── start_system.py           # System startup script
├── test_api.py               # API testing suite
├── models/                   # Trained model files
│   ├── final_model.keras
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── feature_names.pkl
├── demo_data.csv             # Sample test data
└── requirements_*.txt        # Dependencies
```

## 🚀 Production Deployment

### **Docker Deployment**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements_api.txt
RUN pip install -r requirements_streamlit.txt

EXPOSE 8000 8501

CMD ["python", "start_system.py"]
```

### **Cloud Deployment**
- **AWS**: Deploy on EC2 with Load Balancer
- **Azure**: Use Container Instances
- **GCP**: Deploy on Cloud Run
- **Heroku**: Use multi-buildpack setup

## 🔧 Configuration

### **Environment Variables**
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Streamlit Configuration  
STREAMLIT_HOST=0.0.0.0
STREAMLIT_PORT=8501

# Model Configuration
MODEL_PATH=models/
CONFIDENCE_THRESHOLD=0.5
```

## 📞 Support & Troubleshooting

### **Common Issues**

1. **Model files not found**
   ```bash
   # Train the model first
   python train_model.py
   ```

2. **API server not starting**
   ```bash
   # Check dependencies
   pip install fastapi uvicorn
   ```

3. **Memory issues**
   ```bash
   # Reduce batch size in model training
   # Use smaller CSV files for testing
   ```

### **Performance Optimization**
- Use GPU acceleration for faster predictions
- Implement caching for repeated requests
- Use load balancing for high traffic
- Monitor system resources

## 🎯 Next Steps

1. **Connect IoT Hardware**: Use the API endpoints for real-time monitoring
2. **Custom Training**: Train on your specific network data
3. **Integration**: Connect to SIEM systems and security tools
4. **Scaling**: Deploy in production with load balancing
5. **Monitoring**: Set up alerts and dashboards for 24/7 monitoring

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

**🛡️ Built for Enterprise Security • Real-Time Detection • IoT Ready**