import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Page config
st.set_page_config(
    page_title="IDS - Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .attack-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-weight: bold;
    }
    .danger {
        background-color: #ff4444;
        color: white;
    }
    .safe {
        background-color: #44ff44;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load model artifacts
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/ids_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return model, scaler, label_encoder, feature_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please run: python train_model.py")
        return None, None, None, None

model, scaler, label_encoder, feature_names = load_model()

# Header
st.markdown('<h1 class="main-header">🛡️ IDS - Intrusion Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Multi-Attack Detection with 15+ Attack Classes</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/security-checked.png", width=150)
    st.title("🎯 System Info")
    
    if model is not None:
        st.success("✅ Model Loaded")
        st.metric("Attack Classes", len(label_encoder.classes_))
        st.metric("Features", len(feature_names))
        st.metric("Model Accuracy", "99.48%")
        
        with st.expander("📋 Attack Types"):
            for i, attack in enumerate(sorted(label_encoder.classes_), 1):
                emoji = "✅" if attack == "Normal" else "⚠️"
                st.write(f"{i}. {emoji} {attack}")
    else:
        st.error("❌ Model Not Loaded")
    
    st.divider()
    st.info("💡 **Tip**: Upload CSV files with network traffic features")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📁 File Upload", "🔴 Real-Time Detection", "📊 Model Info", "🔌 API Guide"])

# Tab 1: File Upload
with tab1:
    st.header("📁 Upload Network Traffic Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload CIC-IDS or UNSW-NB15 format CSV files"
        )
    
    with col2:
        st.info("**Supported Formats:**\n- CIC-IDS2017\n- UNSW-NB15\n- Custom network traffic")
    
    if uploaded_file is not None and model is not None:
        try:
            # Load data
            with st.spinner("Loading data..."):
                df = pd.read_csv(uploaded_file)
                original_df = df.copy()
            
            st.success(f"✅ Loaded {len(df):,} rows with {df.shape[1]} columns")
            
            # Show sample
            with st.expander("👀 View Sample Data (First 5 rows)"):
                st.dataframe(df.head(), width='stretch')
            
            # Analyze button
            if st.button("🔍 Analyze Traffic", type="primary"):
                with st.spinner("Analyzing network traffic..."):
                    # Prepare data
                    for col in feature_names:
                        if col not in df.columns:
                            df[col] = 0
                    
                    df = df[feature_names]
                    df = df.replace([np.inf, -np.inf], np.nan)
                    df = df.fillna(0)
                    
                    # Predict
                    X_scaled = scaler.transform(df)
                    predictions = model.predict(X_scaled)
                    probabilities = model.predict_proba(X_scaled)
                    
                    # Decode labels
                    predicted_labels = label_encoder.inverse_transform(predictions)
                    confidence_scores = np.max(probabilities, axis=1) * 100
                    
                    # Results
                    st.success("✅ Analysis Complete!")
                    
                    # Metrics
                    st.subheader("📊 Detection Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    attack_counts = pd.Series(predicted_labels).value_counts()
                    total_attacks = len(df) - attack_counts.get('Normal', 0)
                    
                    with col1:
                        st.metric("Total Packets", f"{len(df):,}")
                    with col2:
                        st.metric("Attacks Detected", f"{total_attacks:,}", delta=f"{(total_attacks/len(df)*100):.1f}%")
                    with col3:
                        st.metric("Attack Types", len(attack_counts))
                    with col4:
                        avg_confidence = confidence_scores.mean()
                        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                    
                    # Attack distribution
                    st.subheader("🎯 Attack Distribution")
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # Pie chart
                        fig_pie = px.pie(
                            values=attack_counts.values,
                            names=attack_counts.index,
                            title="Attack Type Distribution",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        st.plotly_chart(fig_pie, width='stretch')
                    
                    with col2:
                        # Bar chart
                        fig_bar = px.bar(
                            x=attack_counts.index,
                            y=attack_counts.values,
                            title="Attack Count by Type",
                            labels={'x': 'Attack Type', 'y': 'Count'},
                            color=attack_counts.values,
                            color_continuous_scale='Reds'
                        )
                        fig_bar.update_layout(showlegend=False)
                        st.plotly_chart(fig_bar, width='stretch')
                    
                    # Detailed results table
                    st.subheader("📋 Detailed Predictions")
                    
                    results_df = pd.DataFrame({
                        'Row': range(1, len(df) + 1),
                        'Prediction': predicted_labels,
                        'Confidence (%)': confidence_scores.round(2),
                        'Is Attack': ['Yes' if p != 'Normal' else 'No' for p in predicted_labels]
                    })
                    
                    # Filter options
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        filter_option = st.selectbox(
                            "Filter by:",
                            ["All", "Attacks Only", "Normal Only"]
                        )
                    
                    if filter_option == "Attacks Only":
                        results_df = results_df[results_df['Is Attack'] == 'Yes']
                    elif filter_option == "Normal Only":
                        results_df = results_df[results_df['Is Attack'] == 'No']
                    
                    # Display table
                    st.dataframe(
                        results_df,
                        width='stretch',
                        height=400
                    )
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"ids_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            st.info("Make sure your CSV has the correct network traffic features")

# Tab 2: Real-Time Detection
with tab2:
    st.header("🔴 Real-Time Attack Detection")
    st.info("💡 Simulate real-time detection by entering network traffic features manually")
    
    if model is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Enter Traffic Features")
            
            protocol = st.number_input("Protocol", value=6, min_value=0, max_value=255)
            flow_duration = st.number_input("Flow Duration (ms)", value=120000, min_value=0)
            total_fwd_packets = st.number_input("Total Forward Packets", value=10, min_value=0)
            total_bwd_packets = st.number_input("Total Backward Packets", value=8, min_value=0)
            fwd_packets_length = st.number_input("Forward Packets Length Total", value=5000, min_value=0)
            bwd_packets_length = st.number_input("Backward Packets Length Total", value=4000, min_value=0)
            
        with col2:
            st.subheader("⚙️ Additional Features")
            
            flow_bytes_per_sec = st.number_input("Flow Bytes/s", value=75000.0, min_value=0.0)
            flow_packets_per_sec = st.number_input("Flow Packets/s", value=150.0, min_value=0.0)
            fwd_iat_mean = st.number_input("Forward IAT Mean", value=6000.0, min_value=0.0)
            bwd_iat_mean = st.number_input("Backward IAT Mean", value=7000.0, min_value=0.0)
        
        if st.button("🔍 Detect Attack", type="primary"):
            with st.spinner("Analyzing traffic..."):
                # Create feature dict
                traffic_data = {
                    "Protocol": protocol,
                    "Flow Duration": flow_duration,
                    "Total Fwd Packets": total_fwd_packets,
                    "Total Backward Packets": total_bwd_packets,
                    "Fwd Packets Length Total": fwd_packets_length,
                    "Bwd Packets Length Total": bwd_packets_length,
                    "Flow Bytes/s": flow_bytes_per_sec,
                    "Flow Packets/s": flow_packets_per_sec,
                    "Fwd IAT Mean": fwd_iat_mean,
                    "Bwd IAT Mean": bwd_iat_mean
                }
                
                # Prepare data
                df = pd.DataFrame([traffic_data])
                for col in feature_names:
                    if col not in df.columns:
                        df[col] = 0
                
                df = df[feature_names]
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.fillna(0)
                
                # Predict
                X_scaled = scaler.transform(df)
                prediction = model.predict(X_scaled)[0]
                probabilities = model.predict_proba(X_scaled)[0]
                
                predicted_label = label_encoder.inverse_transform([prediction])[0]
                confidence = np.max(probabilities) * 100
                
                # Display result
                st.divider()
                
                if predicted_label == "Normal":
                    st.success(f"✅ **NORMAL TRAFFIC**")
                    st.balloons()
                else:
                    st.error(f"⚠️ **ATTACK DETECTED: {predicted_label}**")
                    st.warning("🚨 Immediate action recommended!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prediction", predicted_label)
                with col2:
                    st.metric("Confidence", f"{confidence:.2f}%")
                with col3:
                    st.metric("Status", "Attack" if predicted_label != "Normal" else "Safe")
                
                # Top predictions
                st.subheader("📊 Top 5 Predictions")
                top_5_idx = np.argsort(probabilities)[-5:][::-1]
                
                for idx in top_5_idx:
                    attack = label_encoder.inverse_transform([idx])[0]
                    prob = float(probabilities[idx] * 100)
                    st.progress(float(prob / 100), text=f"{attack}: {prob:.2f}%")

# Tab 3: Model Info
with tab3:
    st.header("📊 Model Information")
    
    if model is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Model Capabilities")
            st.write(f"**Algorithm:** XGBoost Classifier")
            st.write(f"**Total Attack Classes:** {len(label_encoder.classes_)}")
            st.write(f"**Total Features:** {len(feature_names)}")
            st.write(f"**Model Accuracy:** 99.48%")
            st.write(f"**Training Dataset:** CIC-IDS2017 (2.3M+ records)")
            st.write(f"**Prediction Time:** <100ms")
            
            st.subheader("🎯 Attack Classes")
            attack_df = pd.DataFrame({
                'Attack Type': sorted(label_encoder.classes_),
                'Status': ['✅ Safe' if a == 'Normal' else '⚠️ Threat' for a in sorted(label_encoder.classes_)]
            })
            st.dataframe(attack_df, width='stretch', hide_index=True)
        
        with col2:
            st.subheader("📋 Feature List (First 30)")
            features_df = pd.DataFrame({
                'Feature Name': feature_names[:30]
            })
            st.dataframe(features_df, width='stretch', height=600)
            
            if len(feature_names) > 30:
                st.info(f"... and {len(feature_names) - 30} more features")

# Tab 4: API Guide
with tab4:
    st.header("🔌 Hardware Integration API Guide")
    
    st.markdown("""
    ### 🚀 Real-Time API Endpoint
    
    The system provides a REST API for hardware integration with ESP8266, ESP32, or Raspberry Pi.
    
    #### Endpoint:
    ```
    POST http://localhost:5000/api/predict-live
    Content-Type: application/json
    ```
    
    #### Request Example:
    ```json
    {
        "Protocol": 6,
        "Flow Duration": 120000,
        "Total Fwd Packets": 10,
        "Total Backward Packets": 8,
        "Fwd Packets Length Total": 5000,
        "Bwd Packets Length Total": 4000,
        "Flow Bytes/s": 75000,
        "Flow Packets/s": 150
    }
    ```
    
    #### Response Example:
    ```json
    {
        "success": true,
        "prediction": "DDoS",
        "confidence": 95.5,
        "is_attack": true,
        "top_predictions": [
            {"attack": "DDoS", "probability": 95.5},
            {"attack": "DoS-Hulk", "probability": 3.2},
            {"attack": "Normal", "probability": 1.3}
        ],
        "timestamp": "2025-12-09T18:00:00"
    }
    ```
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 ESP8266/ESP32 Example")
        st.code("""
#include <ESP8266HTTPClient.h>
#include <ArduinoJson.h>

void sendTrafficData() {
    HTTPClient http;
    http.begin("http://192.168.1.100:5000/api/predict-live");
    http.addHeader("Content-Type", "application/json");
    
    StaticJsonDocument<1024> doc;
    doc["Protocol"] = 6;
    doc["Flow Duration"] = 120000;
    doc["Total Fwd Packets"] = 10;
    // ... add more features
    
    String payload;
    serializeJson(doc, payload);
    
    int httpCode = http.POST(payload);
    if (httpCode == 200) {
        String response = http.getString();
        // Parse and handle response
    }
    http.end();
}
        """, language="cpp")
    
    with col2:
        st.subheader("🐍 Python/Raspberry Pi Example")
        st.code("""
import requests
import json

API_URL = "http://localhost:5000/api/predict-live"

traffic_data = {
    "Protocol": 6,
    "Flow Duration": 120000,
    "Total Fwd Packets": 10,
    "Total Backward Packets": 8,
    # ... add more features
}

response = requests.post(API_URL, json=traffic_data)
result = response.json()

if result['is_attack']:
    print(f"⚠️ Attack: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2f}%")
else:
    print("✅ Normal traffic")
        """, language="python")
    
    st.info("💡 **Note:** Make sure the Flask API server is running: `python app.py`")
    
    if st.button("🧪 Test API Connection"):
        try:
            import requests
            response = requests.get("http://localhost:5000/api/model-info", timeout=2)
            if response.status_code == 200:
                st.success("✅ API is running and accessible!")
                st.json(response.json())
            else:
                st.error("❌ API returned an error")
        except:
            st.warning("⚠️ API is not running. Start it with: `python app.py`")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🛡️ <strong>IDS - Intrusion Detection System</strong></p>
    <p>Detecting 15+ attack types with 99.48% accuracy</p>
    <p>Built with Streamlit, XGBoost, and CIC-IDS2017 dataset</p>
</div>
""", unsafe_allow_html=True)
