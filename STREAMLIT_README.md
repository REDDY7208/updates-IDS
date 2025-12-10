# 🛡️ IDS Attack Detection System - Streamlit App

A comprehensive Streamlit web application for Intrusion Detection System (IDS) that uses a trained CNN-LSTM deep learning model to detect various types of network attacks.

## 🚀 Features

### 📁 File Upload & Analysis
- Upload CSV files containing network traffic data
- Batch analysis of multiple samples
- Interactive visualizations:
  - **Pie Chart**: Attack type distribution
  - **Histogram**: Prediction confidence distribution
  - **Timeline**: Attack detection over samples
- Detailed results table with filtering options
- Download results as CSV

### 🔍 Single Prediction
- Manual input of network features
- Real-time prediction for individual samples
- Top 5 prediction probabilities
- Visual confidence display

### 📈 Model Statistics
- Model architecture information
- Attack class details
- Feature information
- Performance metrics

## 🎯 Supported Attack Types

The model can detect the following attack types:
- **Normal** - Legitimate traffic
- **DoS** - Denial of Service attacks
- **DDoS** - Distributed Denial of Service
- **Brute-Force** - Password/credential attacks
- **Botnet** - Bot network traffic
- **PortScan** - Port scanning activities
- **Exploit** - System exploitation attempts
- **Probe** - Network reconnaissance
- **Generic** - Generic attack patterns
- **Fuzzers** - Fuzzing attacks
- **Backdoor** - Backdoor communications
- **Web Attacks** - XSS, SQL Injection, etc.
- **Flooding** - Network flooding attacks
- **Blackhole/Grayhole** - Routing attacks (WSN)

## 🛠️ Installation & Setup

### Prerequisites
```bash
pip install -r requirements_streamlit.txt
```

### Required Files
Make sure you have these model files in the `models/` directory:
- `final_model.keras` - Trained CNN-LSTM model
- `scaler.pkl` - Feature scaler
- `label_encoder.pkl` - Label encoder
- `feature_names.pkl` - Feature names list

### Running the App
```bash
streamlit run streamlit_app.py
```

The app will be available at: `http://localhost:8501`

## 📊 How to Use

### 1. File Upload Analysis
1. Go to the "📁 File Upload & Analysis" tab
2. Upload a CSV file with network traffic features
3. Click "🔍 Analyze File" to process
4. View results in interactive charts and tables
5. Filter results by attack type or confidence level
6. Download results as CSV

### 2. Single Sample Prediction
1. Go to the "🔍 Single Prediction" tab
2. Enter values for the network features
3. Click "🔍 Predict" to get real-time results
4. View prediction confidence and top alternatives

### 3. Model Information
1. Go to the "📈 Model Statistics" tab
2. View model architecture details
3. See all supported attack classes
4. Check feature information

## 📋 CSV File Format

Your CSV file should contain network traffic features. The app will automatically:
- Handle missing columns (filled with 0)
- Clean infinite values and NaN
- Scale features using the trained scaler
- Reshape data for the CNN-LSTM model

### Sample Features
- Flow Duration
- Total Fwd/Bwd Packets
- Packet Length statistics
- Flow timing features
- Protocol flags
- And many more...

## 🎨 Visualizations

### Pie Chart
Shows the distribution of different attack types in your dataset.

### Confidence Histogram
Displays the distribution of prediction confidence scores.

### Timeline Chart
Shows attack detection patterns across samples (useful for temporal analysis).

### Bar Chart (Single Prediction)
Shows top 5 prediction probabilities for manual input.

## 🔧 Technical Details

### Model Architecture
- **Type**: CNN-LSTM Deep Learning
- **Input**: Network traffic features (137 features)
- **Preprocessing**: StandardScaler normalization
- **Output**: Multi-class classification (19 classes)

### Performance Features
- **Memory Optimized**: Efficient data processing
- **Real-time**: Fast predictions for single samples
- **Batch Processing**: Handle large CSV files
- **Interactive**: Plotly-based visualizations

## 📁 Demo Data

Use the included `demo_data.csv` file to test the application:
```bash
# The demo file contains 3 sample network flows
# Upload it in the File Upload tab to see the app in action
```

## 🚨 Security Notes

- The app runs locally on your machine
- No data is sent to external servers
- All processing happens on your local environment
- Model files are loaded locally

## 🐛 Troubleshooting

### Common Issues

1. **Model files not found**
   - Ensure all `.pkl` and `.keras` files are in the `models/` directory
   - Check file permissions

2. **CSV upload errors**
   - Verify CSV format and encoding
   - Check for special characters in column names
   - Ensure numeric data types

3. **Memory issues with large files**
   - The app processes files in batches
   - For very large files (>100MB), consider splitting them

4. **Prediction errors**
   - Check that input features match training data format
   - Verify all numeric values are valid

## 📞 Support

If you encounter any issues:
1. Check the Streamlit console for error messages
2. Verify all dependencies are installed
3. Ensure model files are present and accessible
4. Check CSV file format and data quality

## 🎯 Next Steps

- Add more visualization options
- Implement real-time monitoring
- Add model performance metrics
- Include feature importance analysis
- Add export options for reports