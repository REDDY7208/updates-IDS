#!/usr/bin/env python3
"""
Startup script for IDS Real-Time Detection System
Starts both the FastAPI backend and Streamlit frontend
"""

import subprocess
import sys
import time
import os
import signal
import threading
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'streamlit', 'pandas', 
        'numpy', 'tensorflow', 'plotly', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n💡 Install missing packages:")
        print("   pip install fastapi uvicorn streamlit plotly")
        return False
    
    return True

def check_model_files():
    """Check if model files exist"""
    model_files = [
        'praveen/final_model.keras',
        'praveen/scaler.pkl', 
        'praveen/label_encoder.pkl',
        'praveen/feature_names.pkl'
    ]
    
    missing_files = []
    
    for file_path in model_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing model files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 Please ensure model files are in praveen/ folder:")
        print("   Check if praveen/final_model.keras exists")
        return False
    
    return True

def start_api_server():
    """Start the FastAPI server"""
    print("🚀 Starting API Server...")
    try:
        # Start API server
        api_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "api_server:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if process is still running
        if api_process.poll() is None:
            print("✅ API Server started successfully!")
            print("   URL: http://localhost:8000")
            print("   Docs: http://localhost:8000/docs")
            return api_process
        else:
            stdout, stderr = api_process.communicate()
            print("❌ API Server failed to start:")
            print(f"   Error: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return None

def start_streamlit():
    """Start the Streamlit app"""
    print("\n🎨 Starting Streamlit App...")
    try:
        # Start Streamlit
        streamlit_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if process is still running
        if streamlit_process.poll() is None:
            print("✅ Streamlit App started successfully!")
            print("   URL: http://localhost:8501")
            return streamlit_process
        else:
            stdout, stderr = streamlit_process.communicate()
            print("❌ Streamlit failed to start:")
            print(f"   Error: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to start Streamlit: {e}")
        return None

def monitor_processes(api_process, streamlit_process):
    """Monitor both processes"""
    print("\n📊 System Status:")
    print("   API Server: ✅ Running")
    print("   Streamlit: ✅ Running")
    print("\n🌐 Access URLs:")
    print("   📊 Dashboard: http://localhost:8501")
    print("   🔌 API Docs: http://localhost:8000/docs")
    print("   🧪 API Health: http://localhost:8000/api/health")
    
    print("\n💡 Usage:")
    print("   1. Open http://localhost:8501 for the dashboard")
    print("   2. Use http://localhost:8000/api/predict-live for IoT integration")
    print("   3. Test with: python test_api.py")
    
    print("\n⚠️  Press Ctrl+C to stop both servers")
    
    try:
        while True:
            # Check if processes are still running
            if api_process.poll() is not None:
                print("❌ API Server stopped unexpectedly")
                break
            
            if streamlit_process.poll() is not None:
                print("❌ Streamlit stopped unexpectedly")
                break
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down system...")
        
        # Terminate processes
        if api_process.poll() is None:
            api_process.terminate()
            print("   ✅ API Server stopped")
        
        if streamlit_process.poll() is None:
            streamlit_process.terminate()
            print("   ✅ Streamlit stopped")
        
        print("👋 System shutdown complete!")

def main():
    """Main startup function"""
    print("=" * 60)
    print("🛡️  IDS REAL-TIME DETECTION SYSTEM")
    print("=" * 60)
    
    # Check requirements
    print("🔍 Checking requirements...")
    if not check_requirements():
        return 1
    
    print("✅ All packages installed")
    
    # Check model files
    print("🔍 Checking model files...")
    if not check_model_files():
        return 1
    
    print("✅ All model files found")
    
    # Start API server
    api_process = start_api_server()
    if not api_process:
        return 1
    
    # Start Streamlit
    streamlit_process = start_streamlit()
    if not streamlit_process:
        # Stop API if Streamlit fails
        api_process.terminate()
        return 1
    
    # Monitor both processes
    monitor_processes(api_process, streamlit_process)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())