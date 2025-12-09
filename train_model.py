import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("MULTI-ATTACK IDS MODEL TRAINING")
print("=" * 80)

# Load CIC-IDS Parquet files (10-15 attack types)
print("\n[1/6] Loading CIC-IDS Dataset (Parquet)...")
parquet_path = 'Datasets/Datasets/cic-ids'
parquet_files = [
    'Benign-Monday-no-metadata.parquet',
    'Botnet-Friday-no-metadata.parquet',
    'Bruteforce-Tuesday-no-metadata.parquet',
    'DDoS-Friday-no-metadata.parquet',
    'DoS-Wednesday-no-metadata.parquet',
    'Infiltration-Thursday-no-metadata.parquet',
    'Portscan-Friday-no-metadata.parquet',
    'WebAttacks-Thursday-no-metadata.parquet'
]

dfs = []
for file in parquet_files:
    df = pd.read_parquet(os.path.join(parquet_path, file))
    dfs.append(df)
    print(f"  Loaded {file}: {len(df):,} records")

cic_df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal CIC-IDS records: {len(cic_df):,}")

# Load UNSW-NB15 Dataset
print("\n[2/6] Loading UNSW-NB15 Dataset...")
unsw_train = pd.read_csv('Datasets/Datasets/2nd/UNSW_NB15_training-set.csv')
unsw_test = pd.read_csv('Datasets/Datasets/2nd/UNSW_NB15_testing-set.csv')
unsw_df = pd.concat([unsw_train, unsw_test], ignore_index=True)
print(f"Total UNSW-NB15 records: {len(unsw_df):,}")

# Process CIC-IDS data
print("\n[3/6] Processing CIC-IDS Dataset...")
cic_df = cic_df.replace([np.inf, -np.inf], np.nan)
cic_df = cic_df.fillna(0)

# Map attack labels
label_mapping = {
    'Benign': 'Normal',
    'BENIGN': 'Normal',
    'Bot': 'Botnet',
    'FTP-Patator': 'Brute-Force-FTP',
    'SSH-Patator': 'Brute-Force-SSH',
    'DDoS': 'DDoS',
    'DoS slowloris': 'DoS-Slowloris',
    'DoS Slowhttptest': 'DoS-SlowHTTP',
    'DoS Hulk': 'DoS-Hulk',
    'DoS GoldenEye': 'DoS-GoldenEye',
    'Heartbleed': 'Heartbleed',
    'Infiltration': 'Infiltration',
    'PortScan': 'Port-Scan',
    'Web Attack � Brute Force': 'Web-Brute-Force',
    'Web Attack � XSS': 'XSS',
    'Web Attack � Sql Injection': 'SQL-Injection'
}

cic_df['Label'] = cic_df['Label'].map(label_mapping)
X_cic = cic_df.drop('Label', axis=1)
y_cic = cic_df['Label']

# Sample to balance dataset (take max 50k per class)
print("\n[4/6] Balancing dataset...")
sampled_dfs = []
for label in y_cic.unique():
    label_df = cic_df[cic_df['Label'] == label]
    if len(label_df) > 50000:
        label_df = label_df.sample(n=50000, random_state=42)
    sampled_dfs.append(label_df)

balanced_df = pd.concat(sampled_dfs, ignore_index=True)
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

X = balanced_df.drop('Label', axis=1)
y = balanced_df['Label']

print(f"\nBalanced dataset size: {len(balanced_df):,}")
print(f"\nAttack distribution:")
print(y.value_counts())

# Encode labels
print("\n[5/6] Training XGBoost Model...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost
model = XGBClassifier(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)

model.fit(X_train_scaled, y_train)

# Evaluate
print("\n[6/6] Evaluating Model...")
y_pred = model.predict(X_test_scaled)
accuracy = (y_pred == y_test).mean()
print(f"\nModel Accuracy: {accuracy*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save model and artifacts
print("\n[SAVING] Saving model artifacts...")
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/ids_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')
joblib.dump(list(X.columns), 'models/feature_names.pkl')

print("\n✓ Model saved successfully!")
print(f"✓ Total attack classes: {len(label_encoder.classes_)}")
print(f"✓ Attack types: {list(label_encoder.classes_)}")
print("=" * 80)
