import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define the column headers
columns = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets',
    'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
    'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
    'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
    'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
    'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
    'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
    'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size',
    'Avg Bwd Segment Size', 'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
    'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]

def generate_normal_traffic():
    """Generate normal traffic patterns"""
    return {
        'Flow Duration': np.random.uniform(5000, 300000),
        'Total Fwd Packets': np.random.randint(5, 50),
        'Total Backward Packets': np.random.randint(3, 30),
        'Total Length of Fwd Packets': np.random.uniform(300, 3000),
        'Total Length of Bwd Packets': np.random.uniform(200, 2000),
        'Fwd Packet Length Max': np.random.uniform(60, 150),
        'Fwd Packet Length Min': np.random.uniform(40, 80),
        'Fwd Packet Length Mean': np.random.uniform(50, 100),
        'Fwd Packet Length Std': np.random.uniform(10, 30),
        'Bwd Packet Length Max': np.random.uniform(50, 120),
        'Bwd Packet Length Min': np.random.uniform(30, 70),
        'Bwd Packet Length Mean': np.random.uniform(40, 90),
        'Bwd Packet Length Std': np.random.uniform(8, 25),
        'Flow Bytes/s': np.random.uniform(10, 1000),
        'Flow Packets/s': np.random.uniform(0.1, 10),
        'Flow IAT Mean': np.random.uniform(1000, 10000),
        'Flow IAT Std': np.random.uniform(500, 5000),
        'Flow IAT Max': np.random.uniform(5000, 20000),
        'Flow IAT Min': np.random.uniform(100, 2000),
        'Fwd IAT Total': np.random.uniform(10000, 200000),
        'Fwd IAT Mean': np.random.uniform(1000, 15000),
        'Fwd IAT Std': np.random.uniform(500, 8000),
        'Fwd IAT Max': np.random.uniform(5000, 25000),
        'Fwd IAT Min': np.random.uniform(200, 3000),
        'Bwd IAT Total': np.random.uniform(8000, 150000),
        'Bwd IAT Mean': np.random.uniform(800, 12000),
        'Bwd IAT Std': np.random.uniform(400, 6000),
        'Bwd IAT Max': np.random.uniform(4000, 20000),
        'Bwd IAT Min': np.random.uniform(150, 2500),
        'Fwd PSH Flags': np.random.randint(0, 5),
        'Bwd PSH Flags': np.random.randint(0, 3),
        'Fwd URG Flags': 0,
        'Bwd URG Flags': 0,
        'Fwd Header Length': np.random.uniform(200, 1000),
        'Bwd Header Length': np.random.uniform(150, 800),
        'Fwd Packets/s': np.random.uniform(0.05, 5),
        'Bwd Packets/s': np.random.uniform(0.03, 3),
        'Min Packet Length': np.random.uniform(30, 60),
        'Max Packet Length': np.random.uniform(100, 200),
        'Packet Length Mean': np.random.uniform(50, 120),
        'Packet Length Std': np.random.uniform(15, 35),
        'Packet Length Variance': np.random.uniform(200, 1200),
        'FIN Flag Count': np.random.randint(0, 3),
        'SYN Flag Count': np.random.randint(0, 2),
        'RST Flag Count': 0,
        'PSH Flag Count': np.random.randint(1, 8),
        'ACK Flag Count': np.random.randint(5, 50),
        'URG Flag Count': 0,
        'CWE Flag Count': 0,
        'ECE Flag Count': 0,
        'Down/Up Ratio': np.random.uniform(0, 2),
        'Average Packet Size': np.random.uniform(50, 120),
        'Avg Fwd Segment Size': np.random.uniform(50, 100),
        'Avg Bwd Segment Size': np.random.uniform(40, 90),
        'Fwd Header Length.1': np.random.uniform(200, 1000),
        'Fwd Avg Bytes/Bulk': 0,
        'Fwd Avg Packets/Bulk': 0,
        'Fwd Avg Bulk Rate': 0,
        'Bwd Avg Bytes/Bulk': 0,
        'Bwd Avg Packets/Bulk': 0,
        'Bwd Avg Bulk Rate': 0,
        'Subflow Fwd Packets': np.random.randint(5, 50),
        'Subflow Fwd Bytes': np.random.uniform(300, 3000),
        'Subflow Bwd Packets': np.random.randint(3, 30),
        'Subflow Bwd Bytes': np.random.uniform(200, 2000),
        'Init_Win_bytes_forward': np.random.choice([4096, 8192, 16384]),
        'Init_Win_bytes_backward': np.random.choice([2048, 4096, 8192]),
        'act_data_pkt_fwd': np.random.randint(3, 48),
        'min_seg_size_forward': np.random.uniform(20, 60),
        'Active Mean': np.random.uniform(1000, 8000),
        'Active Std': np.random.uniform(500, 3000),
        'Active Max': np.random.uniform(5000, 15000),
        'Active Min': np.random.uniform(200, 2000),
        'Idle Mean': np.random.uniform(2000, 12000),
        'Idle Std': np.random.uniform(1000, 6000),
        'Idle Max': np.random.uniform(8000, 25000),
        'Idle Min': np.random.uniform(500, 4000)
    }

def generate_ddos_attack():
    """Generate DDoS attack patterns - high packet rate, small packets"""
    return {
        'Flow Duration': np.random.uniform(100, 5000),  # Short duration
        'Total Fwd Packets': np.random.randint(500, 2000),  # Very high packet count
        'Total Backward Packets': np.random.randint(0, 50),  # Low response
        'Total Length of Fwd Packets': np.random.uniform(5000, 50000),
        'Total Length of Bwd Packets': np.random.uniform(0, 1000),
        'Fwd Packet Length Max': np.random.uniform(30, 80),  # Small packets
        'Fwd Packet Length Min': np.random.uniform(20, 50),
        'Fwd Packet Length Mean': np.random.uniform(25, 60),
        'Fwd Packet Length Std': np.random.uniform(5, 15),
        'Bwd Packet Length Max': np.random.uniform(20, 60),
        'Bwd Packet Length Min': np.random.uniform(0, 30),
        'Bwd Packet Length Mean': np.random.uniform(10, 40),
        'Bwd Packet Length Std': np.random.uniform(0, 15),
        'Flow Bytes/s': np.random.uniform(10000, 100000),  # Very high bytes/s
        'Flow Packets/s': np.random.uniform(100, 1000),  # Very high packets/s
        'Flow IAT Mean': np.random.uniform(1, 50),  # Very low inter-arrival time
        'Flow IAT Std': np.random.uniform(1, 20),
        'Flow IAT Max': np.random.uniform(10, 200),
        'Flow IAT Min': np.random.uniform(0, 5),
        'Fwd IAT Total': np.random.uniform(50, 2000),
        'Fwd IAT Mean': np.random.uniform(0.1, 10),
        'Fwd IAT Std': np.random.uniform(0.1, 5),
        'Fwd IAT Max': np.random.uniform(5, 50),
        'Fwd IAT Min': np.random.uniform(0, 2),
        'Bwd IAT Total': np.random.uniform(0, 1000),
        'Bwd IAT Mean': np.random.uniform(0, 50),
        'Bwd IAT Std': np.random.uniform(0, 20),
        'Bwd IAT Max': np.random.uniform(0, 200),
        'Bwd IAT Min': np.random.uniform(0, 10),
        'Fwd PSH Flags': np.random.randint(0, 2),
        'Bwd PSH Flags': 0,
        'Fwd URG Flags': 0,
        'Bwd URG Flags': 0,
        'Fwd Header Length': np.random.uniform(10000, 40000),
        'Bwd Header Length': np.random.uniform(0, 1000),
        'Fwd Packets/s': np.random.uniform(100, 1000),
        'Bwd Packets/s': np.random.uniform(0, 20),
        'Min Packet Length': np.random.uniform(20, 40),
        'Max Packet Length': np.random.uniform(60, 100),
        'Packet Length Mean': np.random.uniform(25, 50),
        'Packet Length Std': np.random.uniform(5, 20),
        'Packet Length Variance': np.random.uniform(25, 400),
        'FIN Flag Count': 0,
        'SYN Flag Count': np.random.randint(1, 5),
        'RST Flag Count': 0,
        'PSH Flag Count': np.random.randint(0, 3),
        'ACK Flag Count': np.random.randint(0, 10),
        'URG Flag Count': 0,
        'CWE Flag Count': 0,
        'ECE Flag Count': 0,
        'Down/Up Ratio': np.random.uniform(0, 0.1),
        'Average Packet Size': np.random.uniform(25, 50),
        'Avg Fwd Segment Size': np.random.uniform(25, 60),
        'Avg Bwd Segment Size': np.random.uniform(0, 40),
        'Fwd Header Length.1': np.random.uniform(10000, 40000),
        'Fwd Avg Bytes/Bulk': 0,
        'Fwd Avg Packets/Bulk': 0,
        'Fwd Avg Bulk Rate': 0,
        'Bwd Avg Bytes/Bulk': 0,
        'Bwd Avg Packets/Bulk': 0,
        'Bwd Avg Bulk Rate': 0,
        'Subflow Fwd Packets': np.random.randint(500, 2000),
        'Subflow Fwd Bytes': np.random.uniform(5000, 50000),
        'Subflow Bwd Packets': np.random.randint(0, 50),
        'Subflow Bwd Bytes': np.random.uniform(0, 1000),
        'Init_Win_bytes_forward': np.random.choice([1024, 2048]),
        'Init_Win_bytes_backward': np.random.choice([0, 1024]),
        'act_data_pkt_fwd': np.random.randint(498, 1998),
        'min_seg_size_forward': np.random.uniform(20, 40),
        'Active Mean': np.random.uniform(10, 100),
        'Active Std': np.random.uniform(5, 50),
        'Active Max': np.random.uniform(50, 300),
        'Active Min': np.random.uniform(1, 20),
        'Idle Mean': np.random.uniform(0, 100),
        'Idle Std': np.random.uniform(0, 50),
        'Idle Max': np.random.uniform(0, 200),
        'Idle Min': np.random.uniform(0, 10)
    }

def generate_portscan_attack():
    """Generate PortScan attack patterns - many short connections"""
    return {
        'Flow Duration': np.random.uniform(10, 1000),  # Very short duration
        'Total Fwd Packets': np.random.randint(1, 10),  # Few packets
        'Total Backward Packets': np.random.randint(0, 5),  # Very few responses
        'Total Length of Fwd Packets': np.random.uniform(50, 500),
        'Total Length of Bwd Packets': np.random.uniform(0, 200),
        'Fwd Packet Length Max': np.random.uniform(40, 100),
        'Fwd Packet Length Min': np.random.uniform(40, 80),
        'Fwd Packet Length Mean': np.random.uniform(40, 80),
        'Fwd Packet Length Std': np.random.uniform(0, 10),
        'Bwd Packet Length Max': np.random.uniform(0, 80),
        'Bwd Packet Length Min': np.random.uniform(0, 60),
        'Bwd Packet Length Mean': np.random.uniform(0, 60),
        'Bwd Packet Length Std': np.random.uniform(0, 15),
        'Flow Bytes/s': np.random.uniform(50, 5000),
        'Flow Packets/s': np.random.uniform(1, 50),
        'Flow IAT Mean': np.random.uniform(10, 500),
        'Flow IAT Std': np.random.uniform(5, 200),
        'Flow IAT Max': np.random.uniform(50, 1000),
        'Flow IAT Min': np.random.uniform(1, 50),
        'Fwd IAT Total': np.random.uniform(5, 800),
        'Fwd IAT Mean': np.random.uniform(1, 200),
        'Fwd IAT Std': np.random.uniform(1, 100),
        'Fwd IAT Max': np.random.uniform(10, 500),
        'Fwd IAT Min': np.random.uniform(0, 20),
        'Bwd IAT Total': np.random.uniform(0, 500),
        'Bwd IAT Mean': np.random.uniform(0, 200),
        'Bwd IAT Std': np.random.uniform(0, 100),
        'Bwd IAT Max': np.random.uniform(0, 400),
        'Bwd IAT Min': np.random.uniform(0, 50),
        'Fwd PSH Flags': 0,
        'Bwd PSH Flags': 0,
        'Fwd URG Flags': 0,
        'Bwd URG Flags': 0,
        'Fwd Header Length': np.random.uniform(20, 200),
        'Bwd Header Length': np.random.uniform(0, 100),
        'Fwd Packets/s': np.random.uniform(1, 50),
        'Bwd Packets/s': np.random.uniform(0, 10),
        'Min Packet Length': np.random.uniform(40, 60),
        'Max Packet Length': np.random.uniform(60, 100),
        'Packet Length Mean': np.random.uniform(40, 80),
        'Packet Length Std': np.random.uniform(0, 20),
        'Packet Length Variance': np.random.uniform(0, 400),
        'FIN Flag Count': np.random.randint(0, 2),
        'SYN Flag Count': np.random.randint(1, 3),
        'RST Flag Count': np.random.randint(0, 2),
        'PSH Flag Count': 0,
        'ACK Flag Count': np.random.randint(0, 5),
        'URG Flag Count': 0,
        'CWE Flag Count': 0,
        'ECE Flag Count': 0,
        'Down/Up Ratio': np.random.uniform(0, 1),
        'Average Packet Size': np.random.uniform(40, 80),
        'Avg Fwd Segment Size': np.random.uniform(40, 80),
        'Avg Bwd Segment Size': np.random.uniform(0, 60),
        'Fwd Header Length.1': np.random.uniform(20, 200),
        'Fwd Avg Bytes/Bulk': 0,
        'Fwd Avg Packets/Bulk': 0,
        'Fwd Avg Bulk Rate': 0,
        'Bwd Avg Bytes/Bulk': 0,
        'Bwd Avg Packets/Bulk': 0,
        'Bwd Avg Bulk Rate': 0,
        'Subflow Fwd Packets': np.random.randint(1, 10),
        'Subflow Fwd Bytes': np.random.uniform(50, 500),
        'Subflow Bwd Packets': np.random.randint(0, 5),
        'Subflow Bwd Bytes': np.random.uniform(0, 200),
        'Init_Win_bytes_forward': np.random.choice([1024, 2048, 4096]),
        'Init_Win_bytes_backward': np.random.choice([0, 1024, 2048]),
        'act_data_pkt_fwd': np.random.randint(0, 8),
        'min_seg_size_forward': np.random.uniform(40, 60),
        'Active Mean': np.random.uniform(5, 200),
        'Active Std': np.random.uniform(2, 100),
        'Active Max': np.random.uniform(20, 500),
        'Active Min': np.random.uniform(1, 50),
        'Idle Mean': np.random.uniform(0, 500),
        'Idle Std': np.random.uniform(0, 200),
        'Idle Max': np.random.uniform(0, 1000),
        'Idle Min': np.random.uniform(0, 100)
    }

def generate_bruteforce_attack():
    """Generate Brute Force attack patterns - repeated login attempts"""
    return {
        'Flow Duration': np.random.uniform(1000, 10000),
        'Total Fwd Packets': np.random.randint(20, 100),
        'Total Backward Packets': np.random.randint(10, 50),
        'Total Length of Fwd Packets': np.random.uniform(1000, 8000),
        'Total Length of Bwd Packets': np.random.uniform(500, 3000),
        'Fwd Packet Length Max': np.random.uniform(80, 200),
        'Fwd Packet Length Min': np.random.uniform(60, 120),
        'Fwd Packet Length Mean': np.random.uniform(70, 150),
        'Fwd Packet Length Std': np.random.uniform(10, 30),
        'Bwd Packet Length Max': np.random.uniform(60, 150),
        'Bwd Packet Length Min': np.random.uniform(40, 100),
        'Bwd Packet Length Mean': np.random.uniform(50, 120),
        'Bwd Packet Length Std': np.random.uniform(8, 25),
        'Flow Bytes/s': np.random.uniform(500, 5000),
        'Flow Packets/s': np.random.uniform(5, 50),
        'Flow IAT Mean': np.random.uniform(100, 1000),
        'Flow IAT Std': np.random.uniform(50, 500),
        'Flow IAT Max': np.random.uniform(500, 3000),
        'Flow IAT Min': np.random.uniform(10, 200),
        'Fwd IAT Total': np.random.uniform(800, 8000),
        'Fwd IAT Mean': np.random.uniform(50, 500),
        'Fwd IAT Std': np.random.uniform(20, 200),
        'Fwd IAT Max': np.random.uniform(200, 1500),
        'Fwd IAT Min': np.random.uniform(5, 100),
        'Bwd IAT Total': np.random.uniform(600, 6000),
        'Bwd IAT Mean': np.random.uniform(60, 600),
        'Bwd IAT Std': np.random.uniform(30, 300),
        'Bwd IAT Max': np.random.uniform(300, 2000),
        'Bwd IAT Min': np.random.uniform(10, 150),
        'Fwd PSH Flags': np.random.randint(5, 20),
        'Bwd PSH Flags': np.random.randint(2, 10),
        'Fwd URG Flags': 0,
        'Bwd URG Flags': 0,
        'Fwd Header Length': np.random.uniform(400, 2000),
        'Bwd Header Length': np.random.uniform(200, 1000),
        'Fwd Packets/s': np.random.uniform(5, 50),
        'Bwd Packets/s': np.random.uniform(2, 25),
        'Min Packet Length': np.random.uniform(40, 80),
        'Max Packet Length': np.random.uniform(150, 250),
        'Packet Length Mean': np.random.uniform(60, 140),
        'Packet Length Std': np.random.uniform(20, 50),
        'Packet Length Variance': np.random.uniform(400, 2500),
        'FIN Flag Count': np.random.randint(1, 5),
        'SYN Flag Count': np.random.randint(1, 3),
        'RST Flag Count': np.random.randint(0, 3),
        'PSH Flag Count': np.random.randint(7, 30),
        'ACK Flag Count': np.random.randint(15, 80),
        'URG Flag Count': 0,
        'CWE Flag Count': 0,
        'ECE Flag Count': 0,
        'Down/Up Ratio': np.random.uniform(0.3, 0.8),
        'Average Packet Size': np.random.uniform(60, 140),
        'Avg Fwd Segment Size': np.random.uniform(70, 150),
        'Avg Bwd Segment Size': np.random.uniform(50, 120),
        'Fwd Header Length.1': np.random.uniform(400, 2000),
        'Fwd Avg Bytes/Bulk': 0,
        'Fwd Avg Packets/Bulk': 0,
        'Fwd Avg Bulk Rate': 0,
        'Bwd Avg Bytes/Bulk': 0,
        'Bwd Avg Packets/Bulk': 0,
        'Bwd Avg Bulk Rate': 0,
        'Subflow Fwd Packets': np.random.randint(20, 100),
        'Subflow Fwd Bytes': np.random.uniform(1000, 8000),
        'Subflow Bwd Packets': np.random.randint(10, 50),
        'Subflow Bwd Bytes': np.random.uniform(500, 3000),
        'Init_Win_bytes_forward': np.random.choice([4096, 8192]),
        'Init_Win_bytes_backward': np.random.choice([2048, 4096]),
        'act_data_pkt_fwd': np.random.randint(18, 98),
        'min_seg_size_forward': np.random.uniform(60, 100),
        'Active Mean': np.random.uniform(200, 2000),
        'Active Std': np.random.uniform(100, 800),
        'Active Max': np.random.uniform(800, 4000),
        'Active Min': np.random.uniform(50, 400),
        'Idle Mean': np.random.uniform(500, 3000),
        'Idle Std': np.random.uniform(200, 1500),
        'Idle Max': np.random.uniform(1500, 8000),
        'Idle Min': np.random.uniform(100, 1000)
    }

def generate_botnet_attack():
    """Generate Botnet attack patterns - coordinated malicious activity"""
    return {
        'Flow Duration': np.random.uniform(30000, 600000),  # Long duration
        'Total Fwd Packets': np.random.randint(100, 500),
        'Total Backward Packets': np.random.randint(80, 400),
        'Total Length of Fwd Packets': np.random.uniform(8000, 40000),
        'Total Length of Bwd Packets': np.random.uniform(6000, 30000),
        'Fwd Packet Length Max': np.random.uniform(200, 1500),
        'Fwd Packet Length Min': np.random.uniform(50, 150),
        'Fwd Packet Length Mean': np.random.uniform(80, 300),
        'Fwd Packet Length Std': np.random.uniform(30, 100),
        'Bwd Packet Length Max': np.random.uniform(150, 1200),
        'Bwd Packet Length Min': np.random.uniform(40, 120),
        'Bwd Packet Length Mean': np.random.uniform(70, 250),
        'Bwd Packet Length Std': np.random.uniform(25, 80),
        'Flow Bytes/s': np.random.uniform(100, 2000),
        'Flow Packets/s': np.random.uniform(1, 20),
        'Flow IAT Mean': np.random.uniform(500, 5000),
        'Flow IAT Std': np.random.uniform(200, 2000),
        'Flow IAT Max': np.random.uniform(2000, 15000),
        'Flow IAT Min': np.random.uniform(50, 1000),
        'Fwd IAT Total': np.random.uniform(25000, 500000),
        'Fwd IAT Mean': np.random.uniform(300, 3000),
        'Fwd IAT Std': np.random.uniform(150, 1500),
        'Fwd IAT Max': np.random.uniform(1500, 12000),
        'Fwd IAT Min': np.random.uniform(30, 800),
        'Bwd IAT Total': np.random.uniform(20000, 400000),
        'Bwd IAT Mean': np.random.uniform(250, 2500),
        'Bwd IAT Std': np.random.uniform(120, 1200),
        'Bwd IAT Max': np.random.uniform(1200, 10000),
        'Bwd IAT Min': np.random.uniform(25, 600),
        'Fwd PSH Flags': np.random.randint(10, 50),
        'Bwd PSH Flags': np.random.randint(8, 40),
        'Fwd URG Flags': 0,
        'Bwd URG Flags': 0,
        'Fwd Header Length': np.random.uniform(2000, 10000),
        'Bwd Header Length': np.random.uniform(1600, 8000),
        'Fwd Packets/s': np.random.uniform(1, 20),
        'Bwd Packets/s': np.random.uniform(0.8, 16),
        'Min Packet Length': np.random.uniform(40, 100),
        'Max Packet Length': np.random.uniform(200, 1500),
        'Packet Length Mean': np.random.uniform(75, 275),
        'Packet Length Std': np.random.uniform(40, 120),
        'Packet Length Variance': np.random.uniform(1600, 14400),
        'FIN Flag Count': np.random.randint(2, 10),
        'SYN Flag Count': np.random.randint(1, 5),
        'RST Flag Count': np.random.randint(0, 5),
        'PSH Flag Count': np.random.randint(18, 90),
        'ACK Flag Count': np.random.randint(80, 450),
        'URG Flag Count': 0,
        'CWE Flag Count': 0,
        'ECE Flag Count': 0,
        'Down/Up Ratio': np.random.uniform(0.6, 1.2),
        'Average Packet Size': np.random.uniform(75, 275),
        'Avg Fwd Segment Size': np.random.uniform(80, 300),
        'Avg Bwd Segment Size': np.random.uniform(70, 250),
        'Fwd Header Length.1': np.random.uniform(2000, 10000),
        'Fwd Avg Bytes/Bulk': 0,
        'Fwd Avg Packets/Bulk': 0,
        'Fwd Avg Bulk Rate': 0,
        'Bwd Avg Bytes/Bulk': 0,
        'Bwd Avg Packets/Bulk': 0,
        'Bwd Avg Bulk Rate': 0,
        'Subflow Fwd Packets': np.random.randint(100, 500),
        'Subflow Fwd Bytes': np.random.uniform(8000, 40000),
        'Subflow Bwd Packets': np.random.randint(80, 400),
        'Subflow Bwd Bytes': np.random.uniform(6000, 30000),
        'Init_Win_bytes_forward': np.random.choice([8192, 16384, 32768]),
        'Init_Win_bytes_backward': np.random.choice([4096, 8192, 16384]),
        'act_data_pkt_fwd': np.random.randint(98, 498),
        'min_seg_size_forward': np.random.uniform(50, 150),
        'Active Mean': np.random.uniform(2000, 15000),
        'Active Std': np.random.uniform(1000, 7500),
        'Active Max': np.random.uniform(8000, 30000),
        'Active Min': np.random.uniform(500, 3000),
        'Idle Mean': np.random.uniform(3000, 20000),
        'Idle Std': np.random.uniform(1500, 10000),
        'Idle Max': np.random.uniform(12000, 50000),
        'Idle Min': np.random.uniform(800, 5000)
    }

def generate_dos_attack():
    """Generate DoS attack patterns - resource exhaustion"""
    return {
        'Flow Duration': np.random.uniform(5000, 50000),
        'Total Fwd Packets': np.random.randint(200, 800),
        'Total Backward Packets': np.random.randint(0, 20),
        'Total Length of Fwd Packets': np.random.uniform(10000, 60000),
        'Total Length of Bwd Packets': np.random.uniform(0, 800),
        'Fwd Packet Length Max': np.random.uniform(50, 150),
        'Fwd Packet Length Min': np.random.uniform(30, 80),
        'Fwd Packet Length Mean': np.random.uniform(40, 100),
        'Fwd Packet Length Std': np.random.uniform(8, 25),
        'Bwd Packet Length Max': np.random.uniform(0, 100),
        'Bwd Packet Length Min': np.random.uniform(0, 50),
        'Bwd Packet Length Mean': np.random.uniform(0, 60),
        'Bwd Packet Length Std': np.random.uniform(0, 20),
        'Flow Bytes/s': np.random.uniform(2000, 20000),
        'Flow Packets/s': np.random.uniform(20, 200),
        'Flow IAT Mean': np.random.uniform(50, 500),
        'Flow IAT Std': np.random.uniform(20, 200),
        'Flow IAT Max': np.random.uniform(200, 2000),
        'Flow IAT Min': np.random.uniform(5, 100),
        'Fwd IAT Total': np.random.uniform(4000, 40000),
        'Fwd IAT Mean': np.random.uniform(20, 200),
        'Fwd IAT Std': np.random.uniform(10, 100),
        'Fwd IAT Max': np.random.uniform(100, 1000),
        'Fwd IAT Min': np.random.uniform(2, 50),
        'Bwd IAT Total': np.random.uniform(0, 2000),
        'Bwd IAT Mean': np.random.uniform(0, 200),
        'Bwd IAT Std': np.random.uniform(0, 100),
        'Bwd IAT Max': np.random.uniform(0, 800),
        'Bwd IAT Min': np.random.uniform(0, 50),
        'Fwd PSH Flags': np.random.randint(1, 5),
        'Bwd PSH Flags': 0,
        'Fwd URG Flags': 0,
        'Bwd URG Flags': 0,
        'Fwd Header Length': np.random.uniform(4000, 16000),
        'Bwd Header Length': np.random.uniform(0, 400),
        'Fwd Packets/s': np.random.uniform(20, 200),
        'Bwd Packets/s': np.random.uniform(0, 5),
        'Min Packet Length': np.random.uniform(30, 60),
        'Max Packet Length': np.random.uniform(100, 200),
        'Packet Length Mean': np.random.uniform(40, 90),
        'Packet Length Std': np.random.uniform(15, 40),
        'Packet Length Variance': np.random.uniform(225, 1600),
        'FIN Flag Count': 0,
        'SYN Flag Count': np.random.randint(1, 3),
        'RST Flag Count': 0,
        'PSH Flag Count': np.random.randint(1, 8),
        'ACK Flag Count': np.random.randint(5, 30),
        'URG Flag Count': 0,
        'CWE Flag Count': 0,
        'ECE Flag Count': 0,
        'Down/Up Ratio': np.random.uniform(0, 0.1),
        'Average Packet Size': np.random.uniform(40, 90),
        'Avg Fwd Segment Size': np.random.uniform(40, 100),
        'Avg Bwd Segment Size': np.random.uniform(0, 60),
        'Fwd Header Length.1': np.random.uniform(4000, 16000),
        'Fwd Avg Bytes/Bulk': 0,
        'Fwd Avg Packets/Bulk': 0,
        'Fwd Avg Bulk Rate': 0,
        'Bwd Avg Bytes/Bulk': 0,
        'Bwd Avg Packets/Bulk': 0,
        'Bwd Avg Bulk Rate': 0,
        'Subflow Fwd Packets': np.random.randint(200, 800),
        'Subflow Fwd Bytes': np.random.uniform(10000, 60000),
        'Subflow Bwd Packets': np.random.randint(0, 20),
        'Subflow Bwd Bytes': np.random.uniform(0, 800),
        'Init_Win_bytes_forward': np.random.choice([2048, 4096]),
        'Init_Win_bytes_backward': np.random.choice([0, 1024]),
        'act_data_pkt_fwd': np.random.randint(198, 798),
        'min_seg_size_forward': np.random.uniform(30, 80),
        'Active Mean': np.random.uniform(100, 1000),
        'Active Std': np.random.uniform(50, 500),
        'Active Max': np.random.uniform(500, 3000),
        'Active Min': np.random.uniform(20, 200),
        'Idle Mean': np.random.uniform(200, 2000),
        'Idle Std': np.random.uniform(100, 1000),
        'Idle Max': np.random.uniform(800, 6000),
        'Idle Min': np.random.uniform(50, 500)
    }

def generate_probe_attack():
    """Generate Probe/Reconnaissance attack patterns"""
    return {
        'Flow Duration': np.random.uniform(2000, 20000),
        'Total Fwd Packets': np.random.randint(10, 80),
        'Total Backward Packets': np.random.randint(5, 40),
        'Total Length of Fwd Packets': np.random.uniform(500, 5000),
        'Total Length of Bwd Packets': np.random.uniform(200, 2500),
        'Fwd Packet Length Max': np.random.uniform(60, 200),
        'Fwd Packet Length Min': np.random.uniform(40, 100),
        'Fwd Packet Length Mean': np.random.uniform(50, 150),
        'Fwd Packet Length Std': np.random.uniform(10, 40),
        'Bwd Packet Length Max': np.random.uniform(50, 180),
        'Bwd Packet Length Min': np.random.uniform(30, 90),
        'Bwd Packet Length Mean': np.random.uniform(40, 130),
        'Bwd Packet Length Std': np.random.uniform(8, 35),
        'Flow Bytes/s': np.random.uniform(200, 3000),
        'Flow Packets/s': np.random.uniform(2, 30),
        'Flow IAT Mean': np.random.uniform(200, 2000),
        'Flow IAT Std': np.random.uniform(100, 1000),
        'Flow IAT Max': np.random.uniform(1000, 8000),
        'Flow IAT Min': np.random.uniform(20, 400),
        'Fwd IAT Total': np.random.uniform(1500, 15000),
        'Fwd IAT Mean': np.random.uniform(100, 1000),
        'Fwd IAT Std': np.random.uniform(50, 500),
        'Fwd IAT Max': np.random.uniform(500, 4000),
        'Fwd IAT Min': np.random.uniform(10, 200),
        'Bwd IAT Total': np.random.uniform(1000, 12000),
        'Bwd IAT Mean': np.random.uniform(80, 800),
        'Bwd IAT Std': np.random.uniform(40, 400),
        'Bwd IAT Max': np.random.uniform(400, 3200),
        'Bwd IAT Min': np.random.uniform(8, 160),
        'Fwd PSH Flags': np.random.randint(2, 15),
        'Bwd PSH Flags': np.random.randint(1, 8),
        'Fwd URG Flags': 0,
        'Bwd URG Flags': 0,
        'Fwd Header Length': np.random.uniform(200, 1600),
        'Bwd Header Length': np.random.uniform(100, 800),
        'Fwd Packets/s': np.random.uniform(2, 30),
        'Bwd Packets/s': np.random.uniform(1, 15),
        'Min Packet Length': np.random.uniform(30, 70),
        'Max Packet Length': np.random.uniform(120, 300),
        'Packet Length Mean': np.random.uniform(45, 140),
        'Packet Length Std': np.random.uniform(20, 60),
        'Packet Length Variance': np.random.uniform(400, 3600),
        'FIN Flag Count': np.random.randint(1, 8),
        'SYN Flag Count': np.random.randint(1, 5),
        'RST Flag Count': np.random.randint(0, 3),
        'PSH Flag Count': np.random.randint(3, 23),
        'ACK Flag Count': np.random.randint(8, 120),
        'URG Flag Count': 0,
        'CWE Flag Count': 0,
        'ECE Flag Count': 0,
        'Down/Up Ratio': np.random.uniform(0.2, 1.5),
        'Average Packet Size': np.random.uniform(45, 140),
        'Avg Fwd Segment Size': np.random.uniform(50, 150),
        'Avg Bwd Segment Size': np.random.uniform(40, 130),
        'Fwd Header Length.1': np.random.uniform(200, 1600),
        'Fwd Avg Bytes/Bulk': 0,
        'Fwd Avg Packets/Bulk': 0,
        'Fwd Avg Bulk Rate': 0,
        'Bwd Avg Bytes/Bulk': 0,
        'Bwd Avg Packets/Bulk': 0,
        'Bwd Avg Bulk Rate': 0,
        'Subflow Fwd Packets': np.random.randint(10, 80),
        'Subflow Fwd Bytes': np.random.uniform(500, 5000),
        'Subflow Bwd Packets': np.random.randint(5, 40),
        'Subflow Bwd Bytes': np.random.uniform(200, 2500),
        'Init_Win_bytes_forward': np.random.choice([4096, 8192, 16384]),
        'Init_Win_bytes_backward': np.random.choice([2048, 4096, 8192]),
        'act_data_pkt_fwd': np.random.randint(8, 78),
        'min_seg_size_forward': np.random.uniform(40, 100),
        'Active Mean': np.random.uniform(500, 5000),
        'Active Std': np.random.uniform(250, 2500),
        'Active Max': np.random.uniform(2000, 15000),
        'Active Min': np.random.uniform(100, 1000),
        'Idle Mean': np.random.uniform(1000, 8000),
        'Idle Std': np.random.uniform(500, 4000),
        'Idle Max': np.random.uniform(4000, 20000),
        'Idle Min': np.random.uniform(200, 2000)
    }

def generate_dataset(attack_type, num_samples=100):
    """Generate a dataset for a specific attack type"""
    attack_generators = {
        'BENIGN': generate_normal_traffic,
        'DDoS': generate_ddos_attack,
        'PortScan': generate_portscan_attack,
        'Brute Force': generate_bruteforce_attack,
        'Botnet': generate_botnet_attack,
        'DoS': generate_dos_attack,
        'Probe': generate_probe_attack
    }
    
    if attack_type not in attack_generators:
        raise ValueError(f"Unknown attack type: {attack_type}")
    
    generator = attack_generators[attack_type]
    data = []
    
    for _ in range(num_samples):
        sample = generator()
        # Ensure all columns are present
        row = [sample.get(col, 0) for col in columns]
        data.append(row)
    
    df = pd.DataFrame(data, columns=columns)
    df['Label'] = attack_type
    
    return df

def generate_mixed_dataset(samples_per_type=50):
    """Generate a mixed dataset with all attack types"""
    attack_types = ['BENIGN', 'DDoS', 'PortScan', 'Brute Force', 'Botnet', 'DoS', 'Probe']
    
    all_data = []
    for attack_type in attack_types:
        print(f"Generating {samples_per_type} samples for {attack_type}...")
        df = generate_dataset(attack_type, samples_per_type)
        all_data.append(df)
    
    # Combine all datasets
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Shuffle the dataset
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    
    return combined_df

def save_attack_datasets():
    """Generate and save individual attack datasets"""
    attack_types = ['DDoS', 'PortScan', 'Brute Force', 'Botnet', 'DoS', 'Probe']
    
    for attack_type in attack_types:
        print(f"Generating {attack_type} dataset...")
        df = generate_dataset(attack_type, 200)  # 200 samples per attack type
        
        # Save to CSV
        filename = f"{attack_type.lower().replace(' ', '_')}_attacks.csv"
        df.to_csv(filename, index=False)
        print(f"Saved {filename} with {len(df)} samples")

def generate_test_data(num_samples=100):
    """Generate test data for API testing"""
    print("Generating mixed test dataset...")
    df = generate_mixed_dataset(num_samples // 7)  # Distribute samples across attack types
    
    # Save test data
    df.to_csv('test_attack_data.csv', index=False)
    print(f"Saved test_attack_data.csv with {len(df)} samples")
    
    return df

if __name__ == "__main__":
    print("Attack Data Generator")
    print("=" * 50)
    
    # Generate individual attack datasets
    save_attack_datasets()
    
    # Generate mixed test dataset
    generate_test_data(350)  # 50 samples per attack type
    
    # Generate a large mixed dataset
    print("\nGenerating large mixed dataset...")
    large_df = generate_mixed_dataset(100)  # 100 samples per type
    large_df.to_csv('large_attack_dataset.csv', index=False)
    print(f"Saved large_attack_dataset.csv with {len(large_df)} samples")
    
    print("\nDataset generation completed!")
    print(f"Generated datasets:")
    print("- Individual attack type files")
    print("- test_attack_data.csv (350 samples)")
    print("- large_attack_dataset.csv (700 samples)")