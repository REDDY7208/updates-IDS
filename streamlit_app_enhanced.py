import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page config
st.set_page_config(
    page_title="IDS - Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Main styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styling */
    .big-title {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #555;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 5px solid #667eea;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    /* Attack badge */
    .attack-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.3rem;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .badge-danger {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .badge-safe {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
    
    /* Success/Error boxes */
    .success-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #00d2ff;
        margin: 1rem 0;
    }
    
    .danger-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #ff6b6b;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Upload box */
    .uploadedFile {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Info box */
    .info-box {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/ids_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return model, scaler, label_encoder, feature_names
    except Exception as e:
        return None, None, None, None

model, scaler, label_encoder, feature_names = load_model()

# Header
st.markdown('<h1 class="big-title">🛡️ IDS - Intrusion Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Multi-Attack Detection with AI-Powered Analysis</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 System Dashboard")
    st.markdown("---")
    
    if model is not None:
        st.success("✅ **System Online**")
        
        # System metrics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(label_encoder.classes_)}</div>
                <div class="metric-label">Attack Classes</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(feature_names)}</div>
                <div class="metric-label">Features</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model info
        st.markdown("### 📊 Model Information")
        st.info(f"""
        **Algorithm:** XGBoost Classifier  
        **Accuracy:** 99.48%  
        **Training Data:** CIC-IDS2017  
        **Samples:** 2.3M+ records
        """)
        
        st.markdown("---")
        
        # Attack types
        with st.expander("🎯 Detected Attack Types", expanded=False):
            attacks = sorted(label_encoder.classes_)
            for attack in attacks:
                if attack == "Normal":
                    st.markdown(f"✅ {attack}")
                else:
                    st.markdown(f"⚠️ {attack}")
    else:
        st.error("❌ **System Offline**")
        st.warning("Please train the model first:\n```python train_model.py```")
    
    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    st.info("""
    • Upload CSV files with network traffic data
    • Supports CIC-IDS and UNSW-NB15 formats
    • Real-time detection available
    • Download results as CSV
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["📁 File Analysis", "🔴 Live Detection", "📚 Documentation"])

# TAB 1: File Analysis
with tab1:
    st.markdown("## 📁 Upload & Analyze Network Traffic")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload network traffic data in CSV format"
        )
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>📋 Supported Formats</h4>
            <ul>
                <li>CIC-IDS2017</li>
                <li>UNSW-NB15</li>
                <li>Custom traffic logs</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    if uploaded_file is not None and model is not None:
        try:
            # Load data
            with st.spinner("📥 Loading data..."):
                df = pd.read_csv(uploaded_file)
                original_df = df.copy()
            
            st.success(f"✅ Successfully loaded **{len(df):,}** rows with **{df.shape[1]}** columns")
            
            # Preview data
            with st.expander("👀 Preview Data (First 10 rows)", expanded=False):
                st.dataframe(df.head(10), width='stretch')
            
            st.markdown("---")
            
            # Analyze button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                analyze_btn = st.button("🔍 **Analyze Traffic**", use_container_width=True)
            
            if analyze_btn:
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Prepare data
                status_text.text("⚙️ Preparing data...")
                progress_bar.progress(25)
                time.sleep(0.3)
                
                for col in feature_names:
                    if col not in df.columns:
                        df[col] = 0
                
                df = df[feature_names]
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.fillna(0)
                
                # Step 2: Scale features
                status_text.text("📊 Scaling features...")
                progress_bar.progress(50)
                time.sleep(0.3)
                
                X_scaled = scaler.transform(df)
                
                # Step 3: Predict
                status_text.text("🤖 Running AI detection...")
                progress_bar.progress(75)
                time.sleep(0.3)
                
                predictions = model.predict(X_scaled)
                probabilities = model.predict_proba(X_scaled)
                
                # Step 4: Process results
                status_text.text("✨ Processing results...")
                progress_bar.progress(100)
                time.sleep(0.3)
                
                predicted_labels = label_encoder.inverse_transform(predictions)
                confidence_scores = np.max(probabilities, axis=1) * 100
                
                progress_bar.empty()
                status_text.empty()
                
                # Results
                st.markdown("---")
                st.markdown("## 🎯 Detection Results")
                
                # Metrics
                attack_counts = pd.Series(predicted_labels).value_counts()
                total_attacks = len(df) - attack_counts.get('Normal', 0)
                avg_confidence = confidence_scores.mean()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(df):,}</div>
                        <div class="metric-label">📦 Total Packets</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color: #ff6b6b;">{total_attacks:,}</div>
                        <div class="metric-label">⚠️ Attacks Detected</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color: #51cf66;">{len(attack_counts)}</div>
                        <div class="metric-label">🎯 Attack Types</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{avg_confidence:.1f}%</div>
                        <div class="metric-label">📊 Avg Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Attack distribution
                st.markdown("### 📊 Attack Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    colors = ['#51cf66' if x == 'Normal' else '#ff6b6b' for x in attack_counts.index]
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=attack_counts.index,
                        values=attack_counts.values,
                        hole=0.4,
                        marker=dict(colors=colors, line=dict(color='white', width=2))
                    )])
                    fig_pie.update_layout(
                        title="Attack Type Distribution",
                        height=400,
                        showlegend=True,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_pie, width='stretch')
                
                with col2:
                    # Bar chart
                    colors_bar = ['#51cf66' if x == 'Normal' else '#ff6b6b' for x in attack_counts.index]
                    fig_bar = go.Figure(data=[go.Bar(
                        x=attack_counts.index,
                        y=attack_counts.values,
                        marker=dict(
                            color=colors_bar,
                            line=dict(color='white', width=1)
                        )
                    )])
                    fig_bar.update_layout(
                        title="Attack Count by Type",
                        xaxis_title="Attack Type",
                        yaxis_title="Count",
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_bar, width='stretch')
                
                # Attack summary
                st.markdown("### 🎯 Attack Summary")
                
                summary_cols = st.columns(min(4, len(attack_counts)))
                for idx, (attack, count) in enumerate(attack_counts.items()):
                    with summary_cols[idx % 4]:
                        badge_class = "badge-safe" if attack == "Normal" else "badge-danger"
                        st.markdown(f"""
                        <div class="attack-badge {badge_class}">
                            {attack}: {count:,}
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Detailed results
                st.markdown("### 📋 Detailed Predictions")
                
                results_df = pd.DataFrame({
                    'Row': range(1, len(df) + 1),
                    'Prediction': predicted_labels,
                    'Confidence (%)': confidence_scores.round(2),
                    'Status': ['🔴 Attack' if p != 'Normal' else '🟢 Safe' for p in predicted_labels]
                })
                
                # Filter
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    filter_option = st.selectbox(
                        "Filter:",
                        ["All Traffic", "Attacks Only", "Normal Only"]
                    )
                
                with col2:
                    sort_option = st.selectbox(
                        "Sort by:",
                        ["Row Number", "Confidence (High to Low)", "Confidence (Low to High)"]
                    )
                
                # Apply filters
                if filter_option == "Attacks Only":
                    results_df = results_df[results_df['Status'] == '🔴 Attack']
                elif filter_option == "Normal Only":
                    results_df = results_df[results_df['Status'] == '🟢 Safe']
                
                # Apply sorting
                if sort_option == "Confidence (High to Low)":
                    results_df = results_df.sort_values('Confidence (%)', ascending=False)
                elif sort_option == "Confidence (Low to High)":
                    results_df = results_df.sort_values('Confidence (%)', ascending=True)
                
                # Display table
                st.dataframe(
                    results_df,
                    width='stretch',
                    height=400,
                    hide_index=True
                )
                
                # Download button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"ids_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
        except Exception as e:
            st.error(f"❌ **Error:** {str(e)}")
            st.info("Please ensure your CSV file has the correct network traffic features.")

# TAB 2: Live Detection
with tab2:
    st.markdown("## 🔴 Real-Time Attack Detection")
    st.info("💡 Enter network traffic features manually to simulate real-time detection")
    
    if model is not None:
        with st.form("live_detection_form"):
            st.markdown("### 📝 Network Traffic Features")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                protocol = st.number_input("Protocol", value=6, min_value=0, max_value=255, help="6=TCP, 17=UDP")
                flow_duration = st.number_input("Flow Duration (ms)", value=120000, min_value=0)
                total_fwd_packets = st.number_input("Forward Packets", value=10, min_value=0)
                total_bwd_packets = st.number_input("Backward Packets", value=8, min_value=0)
            
            with col2:
                fwd_packets_length = st.number_input("Fwd Packet Length", value=5000, min_value=0)
                bwd_packets_length = st.number_input("Bwd Packet Length", value=4000, min_value=0)
                flow_bytes_per_sec = st.number_input("Flow Bytes/s", value=75000.0, min_value=0.0)
                flow_packets_per_sec = st.number_input("Flow Packets/s", value=150.0, min_value=0.0)
            
            with col3:
                fwd_iat_mean = st.number_input("Fwd IAT Mean", value=6000.0, min_value=0.0)
                bwd_iat_mean = st.number_input("Bwd IAT Mean", value=7000.0, min_value=0.0)
                fwd_pkt_len_max = st.number_input("Fwd Pkt Len Max", value=1500, min_value=0)
                bwd_pkt_len_max = st.number_input("Bwd Pkt Len Max", value=1400, min_value=0)
            
            submitted = st.form_submit_button("🔍 **Detect Attack**", use_container_width=True)
            
            if submitted:
                with st.spinner("🤖 Analyzing traffic..."):
                    time.sleep(0.5)
                    
                    # Prepare data
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
                        "Bwd IAT Mean": bwd_iat_mean,
                        "Fwd Packet Length Max": fwd_pkt_len_max,
                        "Bwd Packet Length Max": bwd_pkt_len_max
                    }
                    
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
                    confidence = float(np.max(probabilities) * 100)
                    
                    # Display result
                    st.markdown("---")
                    
                    if predicted_label == "Normal":
                        st.markdown("""
                        <div class="success-box">
                            <h2>✅ NORMAL TRAFFIC DETECTED</h2>
                            <p style="font-size: 1.2rem;">No threats found. Traffic appears legitimate.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.markdown(f"""
                        <div class="danger-box">
                            <h2>⚠️ ATTACK DETECTED: {predicted_label}</h2>
                            <p style="font-size: 1.2rem;">Immediate action recommended!</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{predicted_label}</div>
                            <div class="metric-label">🎯 Prediction</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{confidence:.2f}%</div>
                            <div class="metric-label">📊 Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        status = "Attack" if predicted_label != "Normal" else "Safe"
                        status_color = "#ff6b6b" if predicted_label != "Normal" else "#51cf66"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value" style="color: {status_color};">{status}</div>
                            <div class="metric-label">🚦 Status</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Top predictions
                    st.markdown("### 📊 Top 5 Predictions")
                    top_5_idx = np.argsort(probabilities)[-5:][::-1]
                    
                    for idx in top_5_idx:
                        attack = label_encoder.inverse_transform([idx])[0]
                        prob = float(probabilities[idx] * 100)
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.progress(float(prob / 100))
                        with col2:
                            st.markdown(f"**{attack}**: {prob:.2f}%")

# TAB 3: Documentation
with tab3:
    st.markdown("## 📚 System Documentation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h3>🎯 Attack Classes (15 Types)</h3>
            <ul>
                <li>✅ <strong>Normal</strong> - Legitimate traffic</li>
                <li>⚠️ <strong>Botnet</strong> - C2 Command & Control</li>
                <li>⚠️ <strong>Brute-Force-FTP</strong> - FTP attacks</li>
                <li>⚠️ <strong>Brute-Force-SSH</strong> - SSH attacks</li>
                <li>⚠️ <strong>DDoS</strong> - Distributed DoS</li>
                <li>⚠️ <strong>DoS-GoldenEye</strong> - DoS variant</li>
                <li>⚠️ <strong>DoS-Hulk</strong> - DoS variant</li>
                <li>⚠️ <strong>DoS-SlowHTTP</strong> - Slow HTTP</li>
                <li>⚠️ <strong>DoS-Slowloris</strong> - Slowloris</li>
                <li>⚠️ <strong>Heartbleed</strong> - SSL vulnerability</li>
                <li>⚠️ <strong>Infiltration</strong> - Network infiltration</li>
                <li>⚠️ <strong>Port-Scan</strong> - Port scanning</li>
                <li>⚠️ <strong>SQL-Injection</strong> - SQL attacks</li>
                <li>⚠️ <strong>Web-Brute-Force</strong> - Web attacks</li>
                <li>⚠️ <strong>XSS</strong> - Cross-Site Scripting</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h3>📊 Model Performance</h3>
            <ul>
                <li><strong>Algorithm:</strong> XGBoost Classifier</li>
                <li><strong>Accuracy:</strong> 99.48%</li>
                <li><strong>Training Data:</strong> CIC-IDS2017</li>
                <li><strong>Total Samples:</strong> 2.3M+ records</li>
                <li><strong>Features:</strong> 77 network features</li>
                <li><strong>Prediction Time:</strong> <100ms</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="info-box">
        <h3>🔌 Hardware Integration API</h3>
        <p>The system provides REST API endpoints for hardware integration with ESP8266, ESP32, or Raspberry Pi.</p>
        
        <h4>Endpoint:</h4>
        <pre>POST http://localhost:5000/api/predict-live
Content-Type: application/json</pre>
        
        <h4>Example Request:</h4>
        <pre>{
    "Protocol": 6,
    "Flow Duration": 120000,
    "Total Fwd Packets": 10,
    "Total Backward Packets": 8,
    ...
}</pre>
        
        <h4>Example Response:</h4>
        <pre>{
    "success": true,
    "prediction": "DDoS",
    "confidence": 95.5,
    "is_attack": true,
    "timestamp": "2025-12-09T18:00:00"
}</pre>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p style="font-size: 1.2rem;"><strong>🛡️ IDS - Intrusion Detection System</strong></p>
    <p>Powered by XGBoost AI • Detecting 15+ Attack Types • 99.48% Accuracy</p>
    <p style="font-size: 0.9rem; color: #999;">Built with Streamlit, Plotly, and CIC-IDS2017 Dataset</p>
</div>
""", unsafe_allow_html=True)
