#!/usr/bin/env python3
"""
Test script for IDS Real-Time API
Simulates IoT hardware sending network features to the API
"""

import requests
import json
import time
import random
from datetime import datetime

# API Configuration
API_BASE_URL = "http://localhost:8000"
PREDICT_ENDPOINT = f"{API_BASE_URL}/api/predict-live"
HEALTH_ENDPOINT = f"{API_BASE_URL}/api/health"
STATS_ENDPOINT = f"{API_BASE_URL}/api/stats"

def test_api_health():
    """Test API health endpoint"""
    print("🔍 Testing API Health...")
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ API Health Check Passed")
            print(f"   Status: {data['status']}")
            print(f"   Model Loaded: {data['model_loaded']}")
            print(f"   Classes: {data['total_classes']}")
            print(f"   Features: {data['features_count']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def generate_sample_features(device_id="TEST_DEVICE", attack_type="normal"):
    """Generate sample network features"""
    
    if attack_type == "normal":
        # Normal traffic patterns
        return {
            "flow_duration": random.uniform(1000, 300000),
            "total_fwd_packets": random.randint(1, 50),
            "total_backward_packets": random.randint(1, 30),
            "total_length_fwd_packets": random.uniform(100, 5000),
            "total_length_bwd_packets": random.uniform(100, 3000),
            "fwd_packet_length_mean": random.uniform(50, 200),
            "bwd_packet_length_mean": random.uniform(50, 150),
            "flow_bytes_s": random.uniform(1, 100),
            "flow_packets_s": random.uniform(0.01, 1.0),
            "fwd_psh_flags": random.randint(0, 3),
            "bwd_psh_flags": random.randint(0, 2),
            "device_id": device_id,
            "timestamp": datetime.now().isoformat()
        }
    
    elif attack_type == "dos":
        # DoS attack patterns (high packet rate, small packets)
        return {
            "flow_duration": random.uniform(100, 10000),
            "total_fwd_packets": random.randint(100, 1000),
            "total_backward_packets": random.randint(0, 10),
            "total_length_fwd_packets": random.uniform(5000, 50000),
            "total_length_bwd_packets": random.uniform(0, 500),
            "fwd_packet_length_mean": random.uniform(20, 80),
            "bwd_packet_length_mean": random.uniform(0, 50),
            "flow_bytes_s": random.uniform(1000, 10000),
            "flow_packets_s": random.uniform(10, 100),
            "fwd_psh_flags": random.randint(0, 1),
            "bwd_psh_flags": 0,
            "device_id": device_id,
            "timestamp": datetime.now().isoformat()
        }
    
    elif attack_type == "portscan":
        # Port scan patterns (many small connections)
        return {
            "flow_duration": random.uniform(10, 1000),
            "total_fwd_packets": random.randint(1, 5),
            "total_backward_packets": random.randint(0, 2),
            "total_length_fwd_packets": random.uniform(50, 200),
            "total_length_bwd_packets": random.uniform(0, 100),
            "fwd_packet_length_mean": random.uniform(40, 80),
            "bwd_packet_length_mean": random.uniform(0, 60),
            "flow_bytes_s": random.uniform(10, 100),
            "flow_packets_s": random.uniform(1, 10),
            "fwd_psh_flags": 0,
            "bwd_psh_flags": 0,
            "device_id": device_id,
            "timestamp": datetime.now().isoformat()
        }

def test_single_prediction(features):
    """Test single prediction"""
    try:
        response = requests.post(PREDICT_ENDPOINT, json=features, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Prediction Success:")
            print(f"   Device: {result['device_id']}")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']:.1f}%")
            print(f"   Threat Level: {result['threat_level']}")
            print(f"   Processing Time: {result['processing_time_ms']:.1f}ms")
            print(f"   Is Attack: {result['is_attack']}")
            
            return result
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None

def test_multiple_predictions(count=10):
    """Test multiple predictions with different patterns"""
    print(f"\n🚀 Testing {count} predictions...")
    
    attack_types = ["normal", "normal", "normal", "dos", "portscan"]  # Mostly normal
    results = []
    
    for i in range(count):
        attack_type = random.choice(attack_types)
        device_id = f"TEST_DEVICE_{i+1:03d}"
        
        print(f"\n--- Test {i+1}/{count} ---")
        print(f"Simulating: {attack_type.upper()} traffic from {device_id}")
        
        features = generate_sample_features(device_id, attack_type)
        result = test_single_prediction(features)
        
        if result:
            results.append(result)
        
        # Small delay between requests
        time.sleep(0.5)
    
    return results

def test_api_stats():
    """Test API statistics endpoint"""
    print("\n📊 Testing API Statistics...")
    try:
        response = requests.get(STATS_ENDPOINT, timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print("✅ Statistics Retrieved:")
            print(f"   Total Predictions: {stats['total_predictions']}")
            print(f"   Attack Rate: {stats['attack_rate']:.1f}%")
            print(f"   Avg Confidence: {stats['avg_confidence']:.1f}%")
            print(f"   Avg Processing Time: {stats['avg_processing_time']:.1f}ms")
            print(f"   Threat Distribution: {stats['threat_distribution']}")
            return stats
        else:
            print(f"❌ Stats failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Stats error: {e}")
        return None

def main():
    """Main test function"""
    print("=" * 60)
    print("🛡️  IDS REAL-TIME API TEST SUITE")
    print("=" * 60)
    
    # Test 1: Health Check
    if not test_api_health():
        print("\n❌ API is not healthy. Please start the API server first:")
        print("   python api_server.py")
        return
    
    # Test 2: Single Prediction
    print("\n" + "="*40)
    print("🧪 SINGLE PREDICTION TEST")
    print("="*40)
    
    sample_features = generate_sample_features("SINGLE_TEST", "normal")
    test_single_prediction(sample_features)
    
    # Test 3: Multiple Predictions
    print("\n" + "="*40)
    print("🔄 MULTIPLE PREDICTIONS TEST")
    print("="*40)
    
    results = test_multiple_predictions(5)
    
    # Test 4: Statistics
    print("\n" + "="*40)
    print("📈 STATISTICS TEST")
    print("="*40)
    
    stats = test_api_stats()
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    if results:
        attack_count = sum(1 for r in results if r['is_attack'])
        normal_count = len(results) - attack_count
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        avg_time = sum(r['processing_time_ms'] for r in results) / len(results)
        
        print(f"✅ Total Tests: {len(results)}")
        print(f"✅ Normal Traffic: {normal_count}")
        print(f"🚨 Attacks Detected: {attack_count}")
        print(f"📊 Average Confidence: {avg_confidence:.1f}%")
        print(f"⚡ Average Response Time: {avg_time:.1f}ms")
    
    print("\n🎉 API testing completed!")
    print("\n💡 Next Steps:")
    print("   1. Connect your IoT hardware to the API")
    print("   2. Use the /api/predict-live endpoint")
    print("   3. Monitor real-time predictions in Streamlit")

if __name__ == "__main__":
    main()