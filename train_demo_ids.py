import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import joblib
import os
import warnings
import time
from datetime import datetime

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

class DemoIDSTrainer:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        
    def log_step(self, step, message, start_time=None):
        """Enhanced logging with timestamps"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if start_time:
            elapsed = time.time() - start_time
            print(f"[{timestamp}] {step} {message} (⏱️ {elapsed:.2f}s)")
        else:
            print(f"[{timestamp}] {step} {message}")
    
    def load_cic_ids_data(self):
        """Load CIC-IDS dataset for demonstration"""
        print("🚀 ADVANCED CNN-LSTM IDS DEMONSTRATION")
        print("=" * 60)
        
        start_time = time.time()
        self.log_step("📊 [LOADING]", "Loading CIC-IDS Dataset...")
        
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
        total_records = 0
        
        for file in parquet_files:
            file_path = os.path.join(parquet_path, file)
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                # Take sample to manage memory
                sample_size = min(15000, len(df))
                df_sample = df.sample(n=sample_size, random_state=42)
                dfs.append(df_sample)
                total_records += len(df_sample)
                print(f"    ✓ {file}: {len(df_sample):,} samples from {len(df):,}")
        
        if not dfs:
            print("❌ No CIC-IDS files found!")
            return None
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        self.log_step("✅", f"Dataset loaded: {len(combined_df):,} records", start_time)
        return combined_df
    
    def preprocess_data(self, df):
        """Preprocess the dataset"""
        start_time = time.time()
        self.log_step("🔧 [PREPROCESSING]", "Advanced Data Preprocessing...")
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # Label mapping
        label_mapping = {
            'Benign': 'Normal',
            'BENIGN': 'Normal',
            'Bot': 'Botnet',
            'FTP-Patator': 'Brute-Force',
            'SSH-Patator': 'Brute-Force',
            'DDoS': 'DDoS',
            'DoS slowloris': 'DoS',
            'DoS Slowhttptest': 'DoS',
            'DoS Hulk': 'DoS',
            'DoS GoldenEye': 'DoS',
            'Heartbleed': 'Heartbleed',
            'Infiltration': 'Infiltration',
            'PortScan': 'Port-Scan',
            'Web Attack – Brute Force': 'Web-Attack',
            'Web Attack – XSS': 'Web-Attack',
            'Web Attack – Sql Injection': 'Web-Attack'
        }
        
        df['Label'] = df['Label'].map(label_mapping).fillna(df['Label'])
        
        # Balance dataset
        balanced_dfs = []
        for label in df['Label'].unique():
            label_df = df[df['Label'] == label]
            if len(label_df) > 10000:
                label_df = label_df.sample(n=10000, random_state=42)
            balanced_dfs.append(label_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\n    📊 Processed Dataset:")
        print(f"    📈 Total Records: {len(balanced_df):,}")
        print(f"    🔢 Features: {len(balanced_df.columns)-1}")
        print(f"    🎯 Classes: {balanced_df['Label'].nunique()}")
        
        print(f"\n    🏷️ Class Distribution:")
        class_dist = balanced_df['Label'].value_counts()
        for label, count in class_dist.items():
            percentage = (count / len(balanced_df)) * 100
            print(f"      {label}: {count:,} ({percentage:.1f}%)")
        
        self.log_step("✅", f"Data preprocessed: {len(balanced_df):,} records", start_time)
        return balanced_df
    
    def create_cnn_lstm_model(self, input_shape, num_classes):
        """Advanced CNN-LSTM Hybrid Architecture"""
        print("\n🧠 Building Advanced CNN-LSTM Hybrid Model...")
        
        model = Sequential([
            # Reshape for CNN
            tf.keras.layers.Reshape((input_shape[0], 1)),
            
            # CNN Feature Extraction Layers
            Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(0.3),
            
            # LSTM Temporal Pattern Recognition
            LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            LSTM(64, dropout=0.3, recurrent_dropout=0.3),
            
            # Dense Classification Layers
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(128, activation='relu'),
            Dropout(0.3),
            
            # Output Layer
            Dense(num_classes, activation='softmax')
        ])
        
        # Advanced Optimizer Configuration
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Build the model to count parameters
        model.build(input_shape=(None, input_shape[0]))
        
        print(f"    ✓ CNN-LSTM Architecture Summary:")
        print(f"      🔹 Input Shape: {input_shape}")
        print(f"      🔹 CNN Layers: 3 blocks (64→128→256 filters)")
        print(f"      🔹 LSTM Layers: 2 layers (128→64 units)")
        print(f"      🔹 Dense Layers: 3 layers (512→256→128 units)")
        print(f"      🔹 Output Classes: {num_classes}")
        print(f"      🔹 Total Parameters: {model.count_params():,}")
        
        return model
    
    def train_models(self, df):
        """Train CNN-LSTM and XGBoost models"""
        print("\n🚀 ADVANCED MODEL TRAINING PIPELINE")
        print("=" * 50)
        
        # Prepare data
        X = df.drop('Label', axis=1)
        y = df['Label']
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        num_classes = len(np.unique(y_encoded))
        self.feature_names = list(X.columns)
        
        print(f"\n📊 Training Configuration:")
        print(f"   🔢 Training Samples: {len(X_train):,}")
        print(f"   🔢 Testing Samples: {len(X_test):,}")
        print(f"   🔢 Features: {X_train.shape[1]}")
        print(f"   🎯 Classes: {num_classes}")
        
        # 1. Train CNN-LSTM Model
        start_time = time.time()
        self.log_step("🧠 [1/3]", "Training CNN-LSTM Hybrid Model...")
        
        cnn_lstm_model = self.create_cnn_lstm_model((X_train_scaled.shape[1],), num_classes)
        
        # Advanced Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy', 
                patience=10, 
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Train with validation monitoring
        history = cnn_lstm_model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=100,
            batch_size=256,
            callbacks=callbacks,
            verbose=1
        )
        
        self.models['cnn_lstm'] = cnn_lstm_model
        
        # Evaluate CNN-LSTM
        cnn_lstm_pred = np.argmax(cnn_lstm_model.predict(X_test_scaled, verbose=0), axis=1)
        cnn_lstm_accuracy = accuracy_score(y_test, cnn_lstm_pred)
        
        self.log_step("✅", f"CNN-LSTM trained - Accuracy: {cnn_lstm_accuracy*100:.2f}%", start_time)
        
        # 2. Train XGBoost Model
        start_time = time.time()
        self.log_step("🌳 [2/3]", "Training Advanced XGBoost Model...")
        
        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        xgb_model.fit(X_train_scaled, y_train)
        self.models['xgboost'] = xgb_model
        
        # Evaluate XGBoost
        xgb_pred = xgb_model.predict(X_test_scaled)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        
        self.log_step("✅", f"XGBoost trained - Accuracy: {xgb_accuracy*100:.2f}%", start_time)
        
        # 3. Create Advanced Ensemble
        start_time = time.time()
        self.log_step("🎯 [3/3]", "Creating Advanced Ensemble Model...")
        
        # Get prediction probabilities
        cnn_lstm_proba = cnn_lstm_model.predict(X_test_scaled, verbose=0)
        xgb_proba = xgb_model.predict_proba(X_test_scaled)
        
        # Weighted ensemble (CNN-LSTM gets higher weight)
        ensemble_proba = 0.65 * cnn_lstm_proba + 0.35 * xgb_proba
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        
        self.log_step("✅", f"Ensemble created - Accuracy: {ensemble_accuracy*100:.2f}%", start_time)
        
        # Display comprehensive results
        print(f"\n🎯 FINAL MODEL PERFORMANCE COMPARISON:")
        print(f"   🧠 CNN-LSTM Model:    {cnn_lstm_accuracy*100:.2f}%")
        print(f"   🌳 XGBoost Model:     {xgb_accuracy*100:.2f}%")
        print(f"   🎯 Ensemble Model:    {ensemble_accuracy*100:.2f}%")
        
        # Determine best model
        best_accuracy = max(cnn_lstm_accuracy, xgb_accuracy, ensemble_accuracy)
        if best_accuracy == ensemble_accuracy:
            best_pred = ensemble_pred
            best_name = "Ensemble"
        elif best_accuracy == cnn_lstm_accuracy:
            best_pred = cnn_lstm_pred
            best_name = "CNN-LSTM"
        else:
            best_pred = xgb_pred
            best_name = "XGBoost"
        
        print(f"\n📊 DETAILED CLASSIFICATION REPORT ({best_name} - Best Performing):")
        print(classification_report(y_test, best_pred, target_names=self.label_encoder.classes_))
        
        # Training history summary
        if history:
            final_train_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            print(f"\n📈 CNN-LSTM Training Summary:")
            print(f"   🔹 Final Training Accuracy: {final_train_acc*100:.2f}%")
            print(f"   🔹 Final Validation Accuracy: {final_val_acc*100:.2f}%")
            print(f"   🔹 Epochs Completed: {len(history.history['accuracy'])}")
        
        return X_test_scaled, y_test
    
    def save_models(self):
        """Save all trained models and artifacts"""
        start_time = time.time()
        self.log_step("💾 [SAVING]", "Saving Advanced Model Artifacts...")
        
        os.makedirs('models', exist_ok=True)
        
        # Save CNN-LSTM model
        if 'cnn_lstm' in self.models:
            self.models['cnn_lstm'].save('models/cnn_lstm_demo.h5')
            print("    ✓ CNN-LSTM model saved (cnn_lstm_demo.h5)")
        
        # Save XGBoost model
        if 'xgboost' in self.models:
            joblib.dump(self.models['xgboost'], 'models/xgboost_demo.pkl')
            print("    ✓ XGBoost model saved (xgboost_demo.pkl)")
        
        # Save preprocessing artifacts
        joblib.dump(self.scaler, 'models/scaler_demo.pkl')
        joblib.dump(self.label_encoder, 'models/label_encoder_demo.pkl')
        joblib.dump(self.feature_names, 'models/feature_names_demo.pkl')
        
        print("    ✓ Scaler saved (scaler_demo.pkl)")
        print("    ✓ Label encoder saved (label_encoder_demo.pkl)")
        print("    ✓ Feature names saved (feature_names_demo.pkl)")
        
        self.log_step("✅", "All artifacts saved successfully", start_time)
        
        print(f"\n🎉 ADVANCED IDS TRAINING COMPLETED!")
        print(f"   📁 Models Directory: ./models/")
        print(f"   🏷️ Attack Classes: {len(self.label_encoder.classes_)}")
        print(f"   🎯 Detected Attacks: {list(self.label_encoder.classes_)}")

def main():
    """Main training pipeline"""
    trainer = DemoIDSTrainer()
    
    # Load data
    df = trainer.load_cic_ids_data()
    if df is None:
        return
    
    # Preprocess data
    processed_df = trainer.preprocess_data(df)
    
    # Train models
    trainer.train_models(processed_df)
    
    # Save models
    trainer.save_models()
    
    print("\n" + "=" * 60)
    print("🚀 CNN-LSTM IDS DEMONSTRATION COMPLETED SUCCESSFULLY! 🚀")
    print("=" * 60)

if __name__ == "__main__":
    main()