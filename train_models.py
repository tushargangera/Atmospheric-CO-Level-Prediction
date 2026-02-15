import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)
import joblib
import os
import json

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Load dataset
print("Loading dataset...")
# Using Wine Quality dataset from UCI
# You can download it from: https://archive.ics.uci.edu/dataset/360/air+quality
# For this implementation, we'll use a publicly available version

try:
    # Try to load from local file first (Wine Quality uses semicolon separator)
    df = pd.read_csv('AirQualityUCI.csv', sep=';')
except FileNotFoundError:
    # If not found, try to download it
    print("Dataset file not found. Attempting to download...")
    try:
        import urllib.request
        url = "https://archive.ics.uci.edu/dataset/360/air+quality/AirQualityUCI.csv"
        urllib.request.urlretrieve(url, "AirQualityUCI.csv")
        df = pd.read_csv('AirQualityUCI.csv', sep=';')
        print("✅ Dataset downloaded successfully!")
    except Exception as e:
        print(f"❌ Error downloading dataset: {str(e)}")
        print("Please download manually from: https://archive.ics.uci.edu/dataset/360/air+quality")
        df = None

if df is not None:
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")
    
    # Prepare data
    # Assuming the last column is the target, or we need to specify it
    # For wine quality, typically 'quality' is the target
    if 'quality' in df.columns:
        X = df.drop('quality', axis=1)
        y = df['quality']
        
        # Convert to binary classification (quality >= 6 is good, < 6 is bad)
        # Or keep as multi-class
        # For this assignment, let's keep it as multi-class but ensure we have enough classes
        y = y.astype(int)
        
        # Encode labels to start from 0 (required for XGBoost)
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Save label encoder for inverse transform
        joblib.dump(label_encoder, 'model/label_encoder.pkl')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Keep original y_test for display purposes
        y_test_original = y_test.copy()
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler
        joblib.dump(scaler, 'model/scaler.pkl')
        
        # Initialize models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='mlogloss', verbosity=0)
        }
        
        # Dictionary to store all metrics
        all_metrics = {}
        
        # Train and evaluate each model
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            
            # Use scaled data for models that need it
            if model_name in ['Logistic Regression', 'KNN', 'Naive Bayes']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics (using encoded labels)
            accuracy = accuracy_score(y_test, y_pred)
            
            # For multi-class, use average='weighted' or 'macro'
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            # AUC for multi-class (one-vs-rest)
            try:
                if len(np.unique(y_test)) > 2:
                    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                else:
                    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            except:
                auc = 0.0
            
            # Store metrics
            all_metrics[model_name] = {
                'accuracy': float(accuracy),
                'auc': float(auc),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'mcc': float(mcc),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            # Save model with proper filename mapping
            filename_map = {
                'Logistic Regression': 'logistic_regression.pkl',
                'Decision Tree': 'decision_tree.pkl',
                'KNN': 'knn.pkl',
                'Naive Bayes': 'naive_bayes.pkl',
                'Random Forest': 'random_forest.pkl',
                'XGBoost': 'xgboost.pkl'
            }
            model_filename = f"model/{filename_map[model_name]}"
            joblib.dump(model, model_filename)
            
            print(f"{model_name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
        
        # Save all metrics to JSON
        with open('model/metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=4)
        
        # Save test data for Streamlit app (using original indices)
        X_test_df = pd.DataFrame(X_test, columns=X.columns)
        X_test_df.to_csv('model/test_data.csv', index=False)
        # Save both encoded and original labels
        pd.Series(y_test).to_csv('model/test_labels.csv', index=False)
        pd.Series(label_encoder.inverse_transform(y_test)).to_csv('model/test_labels_original.csv', index=False)
        
        print("\n" + "="*50)
        print("All models trained and saved successfully!")
        print("="*50)
        
        # Print summary table
        print("\nModel Performance Summary:")
        print("-" * 100)
        print(f"{'Model':<25} {'Accuracy':<12} {'AUC':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'MCC':<12}")
        print("-" * 100)
        for model_name, metrics in all_metrics.items():
            print(f"{model_name:<25} {metrics['accuracy']:<12.4f} {metrics['auc']:<12.4f} "
                  f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                  f"{metrics['f1']:<12.4f} {metrics['mcc']:<12.4f}")
        print("-" * 100)
