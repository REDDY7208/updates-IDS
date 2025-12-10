# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler, LabelEncoder
# # from xgboost import XGBClassifier
# # from sklearn.metrics import classification_report, confusion_matrix
# # import joblib
# # import os
# # import warnings
# # warnings.filterwarnings('ignore')

# # print("=" * 80)
# # print("MULTI-ATTACK IDS MODEL TRAINING")
# # print("=" * 80)

# # # Load CIC-IDS Parquet files (10-15 attack types)
# # print("\n[1/6] Loading CIC-IDS Dataset (Parquet)...")
# # parquet_path = 'Datasets/Datasets/cic-ids'
# # parquet_files = [
# #     'Benign-Monday-no-metadata.parquet',
# #     'Botnet-Friday-no-metadata.parquet',
# #     'Bruteforce-Tuesday-no-metadata.parquet',
# #     'DDoS-Friday-no-metadata.parquet',
# #     'DoS-Wednesday-no-metadata.parquet',
# #     'Infiltration-Thursday-no-metadata.parquet',
# #     'Portscan-Friday-no-metadata.parquet',
# #     'WebAttacks-Thursday-no-metadata.parquet'
# # ]

# # dfs = []
# # for file in parquet_files:
# #     df = pd.read_parquet(os.path.join(parquet_path, file))
# #     dfs.append(df)
# #     print(f"  Loaded {file}: {len(df):,} records")

# # cic_df = pd.concat(dfs, ignore_index=True)
# # print(f"\nTotal CIC-IDS records: {len(cic_df):,}")

# # # Load UNSW-NB15 Dataset
# # print("\n[2/6] Loading UNSW-NB15 Dataset...")
# # unsw_train = pd.read_csv('Datasets/Datasets/2nd/UNSW_NB15_training-set.csv')
# # unsw_test = pd.read_csv('Datasets/Datasets/2nd/UNSW_NB15_testing-set.csv')
# # unsw_df = pd.concat([unsw_train, unsw_test], ignore_index=True)
# # print(f"Total UNSW-NB15 records: {len(unsw_df):,}")


# # print("\n[3/6] Loading WSN_DS Dataset...")
# # WSN_DS = pd.read_csv('Datasets/Datasets/1st/WSN-DS.csv')
# # print(f"Total UNSW-NB15 records: {len(WSN-DS):,}")

# # # Process CIC-IDS data
# # print("\n[3/6] Processing CIC-IDS Dataset...")
# # cic_df = cic_df.replace([np.inf, -np.inf], np.nan)
# # cic_df = cic_df.fillna(0)

# # # Map attack labels
# # label_mapping = {
# #     'Benign': 'Normal',
# #     'BENIGN': 'Normal',
# #     'Bot': 'Botnet',
# #     'FTP-Patator': 'Brute-Force-FTP',
# #     'SSH-Patator': 'Brute-Force-SSH',
# #     'DDoS': 'DDoS',
# #     'DoS slowloris': 'DoS-Slowloris',
# #     'DoS Slowhttptest': 'DoS-SlowHTTP',
# #     'DoS Hulk': 'DoS-Hulk',
# #     'DoS GoldenEye': 'DoS-GoldenEye',
# #     'Heartbleed': 'Heartbleed',
# #     'Infiltration': 'Infiltration',
# #     'PortScan': 'Port-Scan',
# #     'Web Attack � Brute Force': 'Web-Brute-Force',
# #     'Web Attack � XSS': 'XSS',
# #     'Web Attack � Sql Injection': 'SQL-Injection'
# # }

# # cic_df['Label'] = cic_df['Label'].map(label_mapping)
# # X_cic = cic_df.drop('Label', axis=1)
# # y_cic = cic_df['Label']

# # # Sample to balance dataset (take max 50k per class)
# # print("\n[4/6] Balancing dataset...")
# # sampled_dfs = []
# # for label in y_cic.unique():
# #     label_df = cic_df[cic_df['Label'] == label]
# #     if len(label_df) > 50000:
# #         label_df = label_df.sample(n=50000, random_state=42)
# #     sampled_dfs.append(label_df)

# # balanced_df = pd.concat(sampled_dfs, ignore_index=True)
# # balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# # X = balanced_df.drop('Label', axis=1)
# # y = balanced_df['Label']

# # print(f"\nBalanced dataset size: {len(balanced_df):,}")
# # print(f"\nAttack distribution:")
# # print(y.value_counts())

# # # Encode labels
# # print("\n[5/6] Training XGBoost Model...")
# # label_encoder = LabelEncoder()
# # y_encoded = label_encoder.fit_transform(y)

# # # Split data
# # X_train, X_test, y_train, y_test = train_test_split(
# #     X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
# # )

# # # Scale features
# # scaler = StandardScaler()
# # X_train_scaled = scaler.fit_transform(X_train)
# # X_test_scaled = scaler.transform(X_test)

# # # Train XGBoost
# # model = XGBClassifier(
# #     n_estimators=100,
# #     max_depth=10,
# #     learning_rate=0.1,
# #     random_state=42,
# #     n_jobs=-1,
# #     eval_metric='mlogloss'
# # )

# # model.fit(X_train_scaled, y_train)

# # # Evaluate
# # print("\n[6/6] Evaluating Model...")
# # y_pred = model.predict(X_test_scaled)
# # accuracy = (y_pred == y_test).mean()
# # print(f"\nModel Accuracy: {accuracy*100:.2f}%")

# # print("\nClassification Report:")
# # print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# # # Save model and artifacts
# # print("\n[SAVING] Saving model artifacts...")
# # os.makedirs('models', exist_ok=True)
# # joblib.dump(model, 'models/ids_model.pkl')
# # joblib.dump(scaler, 'models/scaler.pkl')
# # joblib.dump(label_encoder, 'models/label_encoder.pkl')
# # joblib.dump(list(X.columns), 'models/feature_names.pkl')

# # print("\n✓ Model saved successfully!")
# # print(f"✓ Total attack classes: {len(label_encoder.classes_)}")
# # print(f"✓ Attack types: {list(label_encoder.classes_)}")
# # print("=" * 80)
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import classification_report, confusion_matrix
# import joblib
# import os
# import warnings

# # Deep Learning Imports
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, Flatten, BatchNormalization
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# warnings.filterwarnings('ignore')

# print("=" * 80)
# print("MULTI-ATTACK IDS MODEL TRAINING (CNN-LSTM)")
# print("=" * 80)

# # ==========================================
# # 1. LOAD DATASETS
# # ==========================================

# # Load CIC-IDS Parquet files
# print("\n[1/6] Loading CIC-IDS Dataset (Parquet)...")
# parquet_path = 'Datasets/Datasets/cic-ids'
# # Ensure this path exists or adjust accordingly
# parquet_files = [
#     'Benign-Monday-no-metadata.parquet',
#     'Botnet-Friday-no-metadata.parquet',
#     'Bruteforce-Tuesday-no-metadata.parquet',
#     'DDoS-Friday-no-metadata.parquet',
#     'DoS-Wednesday-no-metadata.parquet',
#     'Infiltration-Thursday-no-metadata.parquet',
#     'Portscan-Friday-no-metadata.parquet',
#     'WebAttacks-Thursday-no-metadata.parquet'
# ]

# dfs = []
# try:
#     for file in parquet_files:
#         file_path = os.path.join(parquet_path, file)
#         if os.path.exists(file_path):
#             df = pd.read_parquet(file_path)
#             dfs.append(df)
#             print(f"  Loaded {file}: {len(df):,} records")
#         else:
#             print(f"  [WARNING] File not found: {file}")
# except Exception as e:
#     print(f"Error loading Parquet files: {e}")

# if dfs:
#     cic_df = pd.concat(dfs, ignore_index=True)
#     print(f"\nTotal CIC-IDS records: {len(cic_df):,}")
# else:
#     raise ValueError("No CIC-IDS data loaded. Check file paths.")

# # Load UNSW-NB15 Dataset (Loaded but currently not merged due to feature mismatch)
# print("\n[2/6] Loading UNSW-NB15 Dataset...")
# try:
#     unsw_train = pd.read_csv('Datasets/Datasets/2nd/UNSW_NB15_training-set.csv')
#     unsw_test = pd.read_csv('Datasets/Datasets/2nd/UNSW_NB15_testing-set.csv')
#     unsw_df = pd.concat([unsw_train, unsw_test], ignore_index=True)
#     print(f"Total UNSW-NB15 records: {len(unsw_df):,}")
# except Exception as e:
#     print(f"Warning: Could not load UNSW dataset: {e}")

# # Load WSN_DS Dataset
# print("\n[3/6] Loading WSN_DS Dataset...")
# try:
#     WSN_DS = pd.read_csv('Datasets/Datasets/1st/WSN-DS.csv')
#     # Fixed variable name from WSN-DS to WSN_DS
#     print(f"Total WSN_DS records: {len(WSN_DS):,}") 
# except Exception as e:
#     print(f"Warning: Could not load WSN-DS dataset: {e}")

# # ==========================================
# # 2. DATA PROCESSING
# # ==========================================
# print("\n[3/6] Processing CIC-IDS Dataset...")
# cic_df = cic_df.replace([np.inf, -np.inf], np.nan)
# cic_df = cic_df.fillna(0)

# # Map attack labels
# label_mapping = {
#     'Benign': 'Normal',
#     'BENIGN': 'Normal',
#     'Bot': 'Botnet',
#     'FTP-Patator': 'Brute-Force-FTP',
#     'SSH-Patator': 'Brute-Force-SSH',
#     'DDoS': 'DDoS',
#     'DoS slowloris': 'DoS-Slowloris',
#     'DoS Slowhttptest': 'DoS-SlowHTTP',
#     'DoS Hulk': 'DoS-Hulk',
#     'DoS GoldenEye': 'DoS-GoldenEye',
#     'Heartbleed': 'Heartbleed',
#     'Infiltration': 'Infiltration',
#     'PortScan': 'Port-Scan',
#     'Web Attack  Brute Force': 'Web-Brute-Force',
#     'Web Attack  XSS': 'XSS',
#     'Web Attack  Sql Injection': 'SQL-Injection'
# }

# # Ensure Label column exists
# if 'Label' in cic_df.columns:
#     cic_df['Label'] = cic_df['Label'].map(label_mapping).fillna('Other')
# else:
#     raise ValueError("Column 'Label' not found in dataset.")

# # Balancing dataset (take max 50k per class to save memory for LSTM)
# print("\n[4/6] Balancing dataset...")
# sampled_dfs = []
# for label in cic_df['Label'].unique():
#     label_df = cic_df[cic_df['Label'] == label]
#     if len(label_df) > 50000:
#         label_df = label_df.sample(n=50000, random_state=42)
#     sampled_dfs.append(label_df)

# balanced_df = pd.concat(sampled_dfs, ignore_index=True)
# balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# X = balanced_df.drop('Label', axis=1)
# y = balanced_df['Label']

# print(f"\nBalanced dataset size: {len(balanced_df):,}")
# print(f"Attack distribution:\n{y.value_counts()}")

# # ==========================================
# # 3. PREPARATION FOR DEEP LEARNING
# # ==========================================
# print("\n[5/6] Preparing Data for CNN-LSTM...")

# # 1. Encode Labels
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)
# # Convert to One-Hot Encoding for Keras
# y_categorical = to_categorical(y_encoded)

# # 2. Split Data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
# )

# # 3. Scale Features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # 4. Reshape for CNN/LSTM [samples, time_steps, features]
# # For tabular data, we treat 'features' as 'time_steps' or 1 channel
# # Shape: (N, Features, 1)
# X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
# X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# print(f"Input Shape: {X_train_reshaped.shape}")
# print(f"Num Classes: {y_categorical.shape[1]}")

# # ==========================================
# # 4. BUILD CNN-LSTM MODEL
# # ==========================================
# model = Sequential()

# # CNN Layers (Feature Extraction)
# model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)))
# model.add(BatchNormalization())
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.2))

# model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.2))

# # LSTM Layers (Sequence Learning)
# model.add(LSTM(64, return_sequences=False)) # False because we feed into Dense next
# model.add(Dropout(0.2))

# # Dense Layers (Classification)
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(y_categorical.shape[1], activation='softmax')) # Output Layer

# # Compile
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()

# # ==========================================
# # 5. TRAIN MODEL
# # ==========================================
# print("\nStarting Training...")
# os.makedirs('models', exist_ok=True)

# # Callbacks
# early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# checkpoint = ModelCheckpoint('models/best_cnn_lstm.keras', monitor='val_accuracy', save_best_only=True)

# history = model.fit(
#     X_train_reshaped, y_train,
#     epochs=20,               # Adjust based on hardware
#     batch_size=128,          # 64, 128, or 256 depending on RAM
#     validation_split=0.1,
#     callbacks=[early_stop, checkpoint],
#     verbose=1
# )

# # ==========================================
# # 6. EVALUATION
# # ==========================================
# print("\n[6/6] Evaluating Model...")

# # Predict
# y_pred_probs = model.predict(X_test_reshaped)
# y_pred_classes = np.argmax(y_pred_probs, axis=1)
# y_true_classes = np.argmax(y_test, axis=1)

# # Accuracy
# accuracy = np.mean(y_pred_classes == y_true_classes)
# print(f"\nModel Accuracy: {accuracy*100:.2f}%")

# # Classification Report
# print("\nClassification Report:")
# print(classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_))

# # Confusion Matrix
# cm = confusion_matrix(y_true_classes, y_pred_classes)
# print("\nConfusion Matrix:\n", cm)

# # ==========================================
# # 7. SAVE ARTIFACTS
# # ==========================================
# print("\n[SAVING] Saving model artifacts...")

# # Save Model (Keras format)
# model.save('models/ids_cnn_lstm_model.h5') # Legacy H5 format
# model.save('models/ids_cnn_lstm_model.keras') # New Keras format

# # Save Preprocessors
# joblib.dump(scaler, 'models/scaler.pkl')
# joblib.dump(label_encoder, 'models/label_encoder.pkl')
# joblib.dump(list(X.columns), 'models/feature_names.pkl')

# print("\n✓ Model and preprocessors saved successfully!")
# print(f"✓ Total attack classes: {len(label_encoder.classes_)}")
# print("=" * 80)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import warnings
import gc

# Deep Learning Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Nadam

warnings.filterwarnings('ignore')

print("=" * 80)
print("MULTI-DATASET IDS MODEL (Memory Optimized)")
print("=" * 80)

# ==========================================
# HELPER: OPTIMIZE & SAMPLE FUNCTION
# ==========================================
def load_and_optimize(df, label_col_name, dataset_name):
    """
    1. Renames label to 'Label'
    2. Drops non-numeric columns to save space
    3. Samples max 25,000 rows per class (Prevents Memory Crash)
    4. Converts to float32 (Reduces RAM by 50%)
    """
    if df.empty:
        return df

    print(f"  > Processing {dataset_name}...")

    # 1. Rename Label
    if label_col_name in df.columns:
        df = df.rename(columns={label_col_name: 'Label'})
    else:
        print(f"    [!] Label column '{label_col_name}' not found. Skipping.")
        return pd.DataFrame()

    # 2. Drop Non-Numeric (Keep Label)
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Ensure Label is kept even if it's string
    cols_to_keep = numeric_cols + ['Label']
    # Filter to only existing columns
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    # Drop duplicates to be safe
    cols_to_keep = list(set(cols_to_keep))
    df = df[cols_to_keep]

    # 3. Early Sampling (Crucial for Memory)
    # We take max 25,000 samples per class. 
    # If we don't do this here, the merge will crash your RAM.
    sampled_dfs = []
    for label in df['Label'].unique():
        class_subset = df[df['Label'] == label]
        if len(class_subset) > 25000:
            class_subset = class_subset.sample(n=25000, random_state=42)
        sampled_dfs.append(class_subset)
    
    if not sampled_dfs:
        return pd.DataFrame()
        
    df_optimized = pd.concat(sampled_dfs, ignore_index=True)

    # 4. Downcast to float32
    # Identify float/int columns (excluding Label)
    num_cols = df_optimized.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        df_optimized[c] = pd.to_numeric(df_optimized[c], downcast='float')
    
    print(f"    -> Reduced {dataset_name} to {len(df_optimized):,} rows (float32)")
    return df_optimized

# ==========================================
# 1. LOAD & OPTIMIZE DATASETS
# ==========================================

processed_dfs = []

# --- 1. CIC-IDS ---
print("\n[1/5] Loading CIC-IDS...")
parquet_path = 'Datasets/Datasets/cic-ids'
parquet_files = [
    'Benign-Monday-no-metadata.parquet', 'Botnet-Friday-no-metadata.parquet',
    'Bruteforce-Tuesday-no-metadata.parquet', 'DDoS-Friday-no-metadata.parquet',
    'DoS-Wednesday-no-metadata.parquet', 'Infiltration-Thursday-no-metadata.parquet',
    'Portscan-Friday-no-metadata.parquet', 'WebAttacks-Thursday-no-metadata.parquet'
]

try:
    for file in parquet_files:
        p_path = os.path.join(parquet_path, file)
        if os.path.exists(p_path):
            temp_df = pd.read_parquet(p_path)
            # Optimize IMMEDIATELY per file to save RAM
            opt_df = load_and_optimize(temp_df, 'Label', f"CIC-{file}")
            processed_dfs.append(opt_df)
            del temp_df # Free memory
            gc.collect()
except Exception as e:
    print(f"Error loading CIC: {e}")

# --- 2. UNSW-NB15 ---
print("\n[2/5] Loading UNSW-NB15...")
try:
    unsw_train = pd.read_csv('Datasets/Datasets/2nd/UNSW_NB15_training-set.csv')
    unsw_test = pd.read_csv('Datasets/Datasets/2nd/UNSW_NB15_testing-set.csv')
    unsw_full = pd.concat([unsw_train, unsw_test])
    
    # UNSW Label is 'attack_cat'
    opt_unsw = load_and_optimize(unsw_full, 'attack_cat', "UNSW-NB15")
    processed_dfs.append(opt_unsw)
    
    del unsw_train, unsw_test, unsw_full
    gc.collect()
except Exception as e:
    print(f"Error loading UNSW: {e}")

# --- 3. WSN-DS ---
print("\n[3/5] Loading WSN-DS...")
try:
    wsn_df = pd.read_csv('Datasets/Datasets/1st/WSN-DS.csv')
    # Determine label column
    wsn_label = 'Class' if 'Class' in wsn_df.columns else 'Attack type'
    
    opt_wsn = load_and_optimize(wsn_df, wsn_label, "WSN-DS")
    processed_dfs.append(opt_wsn)
    
    del wsn_df
    gc.collect()
except Exception as e:
    print(f"Error loading WSN: {e}")

# ==========================================
# 2. MERGE DATASETS
# ==========================================
print("\n[4/5] Merging All Datasets...")
# Remove empty dfs
processed_dfs = [df for df in processed_dfs if not df.empty]

if not processed_dfs:
    raise ValueError("No data loaded!")

# Concatenate (Union of features)
# Fill NaN with 0 (missing features are 0)
full_df = pd.concat(processed_dfs, ignore_index=True).fillna(0)

# Replace Inf
full_df = full_df.replace([np.inf, -np.inf], 0)

print(f"  > Combined Size: {len(full_df):,} rows, {len(full_df.columns)} columns")
gc.collect()

# ==========================================
# 3. LABEL MAPPING
# ==========================================
print("  > Normalizing Labels...")
label_mapping = {
    # CIC
    'BENIGN': 'Normal', 'Benign': 'Normal',
    'Bot': 'Botnet', 'DDoS': 'DDoS',
    'DoS slowloris': 'DoS', 'DoS Slowhttptest': 'DoS', 'DoS Hulk': 'DoS', 'DoS GoldenEye': 'DoS',
    'FTP-Patator': 'Brute-Force', 'SSH-Patator': 'Brute-Force',
    'PortScan': 'PortScan',
    # UNSW
    'Normal': 'Normal', 'Generic': 'Generic', 'Exploits': 'Exploit', 'Fuzzers': 'Fuzzers',
    'DoS': 'DoS', 'Reconnaissance': 'Probe', 'Analysis': 'Probe', 'Backdoor': 'Backdoor',
    # WSN
    'Blackhole': 'Blackhole', 'Grayhole': 'Grayhole', 'Flooding': 'Flooding', 
    'Scheduling': 'Scheduling', 'TDMA': 'TDMA'
}

full_df['Label'] = full_df['Label'].map(label_mapping).fillna(full_df['Label'])

# Filter rare classes (< 100 samples)
class_counts = full_df['Label'].value_counts()
valid_classes = class_counts[class_counts > 100].index
full_df = full_df[full_df['Label'].isin(valid_classes)]

X = full_df.drop('Label', axis=1)
y = full_df['Label']

print(f"  > Final Class Distribution:\n{y.value_counts()}")

# ==========================================
# 4. TRAINING (CNN-LSTM)
# ==========================================
print("\n[5/5] Training Model...")

# Encode
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, stratify=y_encoded, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for CNN [samples, features, 1]
n_features = X_train_scaled.shape[1]
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], n_features, 1)
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], n_features, 1)

# Model
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', padding='same', input_shape=(n_features, 1)))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Dropout(0.3))

model.add(LSTM(64))
model.add(Dropout(0.3))

model.add(Dense(y_categorical.shape[1], activation='softmax'))

#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

opt = Nadam(learning_rate=0.001)


model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# Train
history = model.fit(
    X_train_reshaped, y_train,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=2)]
)

# Evaluate
print("\nEvaluating...")
y_pred = np.argmax(model.predict(X_test_reshaped), axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# Save
os.makedirs('praveen', exist_ok=True)
model.save('praveen/final_model.keras')
joblib.dump(scaler, 'praveen/scaler.pkl')
joblib.dump(label_encoder, 'praveen/label_encoder.pkl')
joblib.dump(list(X.columns), 'praveen/feature_names.pkl')
print("\n✓ Saved successfully.")