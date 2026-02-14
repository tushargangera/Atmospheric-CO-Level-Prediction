import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Atmospheric CO Level Prediction",
    page_icon="💨",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">☁️ Atmospheric CO Level Prediction App</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📋 Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["📊 Model Comparison", "🔮 Predict CO Level", "📈 Dataset Info"]
)

# Load models and metrics
@st.cache_data
def load_models():
    """Load all trained models"""
    models = {}
    model_files = {
        'Logistic Regression': 'model/logistic_regression.pkl',
        'Decision Tree': 'model/decision_tree.pkl',
        'KNN': 'model/knn.pkl',
        'Naive Bayes': 'model/naive_bayes.pkl',
        'Random Forest': 'model/random_forest.pkl',
        'XGBoost': 'model/xgboost.pkl'
    }
    
    scaler = None
    scaler_path = 'model/scaler.pkl'
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            st.warning(f"Could not load scaler: {str(e)}")
    
    for model_name, filepath in model_files.items():
        if os.path.exists(filepath):
            try:
                models[model_name] = joblib.load(filepath)
            except Exception as e:
                st.warning(f"Could not load {model_name}: {str(e)}")
    
    return models, scaler

@st.cache_data
def load_metrics():
    """Load saved metrics"""
    metrics_path = 'model/metrics.json'
    if os.path.exists(metrics_path):
        try:
            import json
            with open(metrics_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Could not load metrics: {str(e)}")
            return None
    return None

# Load data
try:
    models, scaler = load_models()
    metrics = load_metrics()
except Exception as e:
    st.error(f"Error loading models/metrics: {str(e)}")
    models, scaler = {}, None
    metrics = None

# Define feature names (order matters for prediction)
feature_names = [
    'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)',
    'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH',
    'hour', 'day_of_week', 'day_of_year', 'month'
]


# Page 1: Model Comparison
if page == "📊 Model Comparison":
    st.header("Model Performance Comparison")
    
    # Check if models/metrics exist
    if not metrics and not models:
        st.error("""
        ⚠️ **Models and metrics not found!**
        
        Please ensure you have:
        1. Trained the models by running: `python train_models.py`
        2. The `model/` directory contains:
           - `metrics.json`
           - `*.pkl` files for all 6 models
           - `scaler.pkl`
        
        If you're running this on Streamlit Cloud, make sure the model files are committed to GitHub.
        """)
    elif metrics:
        # Display metrics table
        st.subheader("Evaluation Metrics Table")
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, model_metrics in metrics.items():
            comparison_data.append({
                'ML Model Name': model_name,
                'Accuracy': f"{model_metrics['accuracy']:.4f}",
                'AUC': f"{model_metrics['auc']:.4f}",
                'Precision': f"{model_metrics['precision']:.4f}",
                'Recall': f"{model_metrics['recall']:.4f}",
                'F1': f"{model_metrics['f1']:.4f}",
                'MCC': f"{model_metrics['mcc']:.4f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, width='stretch')
        
        # Model selection for detailed view
        st.subheader("Select Model for Detailed Analysis")
        selected_model = st.selectbox(
            "Choose a model",
            list(metrics.keys())
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Metrics")
            selected_metrics = metrics[selected_model]
            st.metric("Accuracy", f"{selected_metrics['accuracy']:.4f}")
            st.metric("AUC Score", f"{selected_metrics['auc']:.4f}")
            st.metric("Precision", f"{selected_metrics['precision']:.4f}")
            st.metric("Recall", f"{selected_metrics['recall']:.4f}")
            st.metric("F1 Score", f"{selected_metrics['f1']:.4f}")
            st.metric("MCC Score", f"{selected_metrics['mcc']:.4f}")
        
        with col2:
            st.markdown("### 📈 Confusion Matrix")
            cm = np.array(selected_metrics['confusion_matrix'])
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create heatmap with better formatting
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                       cbar_kws={'label': 'Count'}, linewidths=0.5)
            ax.set_xlabel('Predicted Quality', fontsize=12, fontweight='bold')
            ax.set_ylabel('Actual Quality', fontsize=12, fontweight='bold')
            ax.set_title(f'Confusion Matrix - {selected_model}', fontsize=14, fontweight='bold')
            
            # Set class labels if available
            if cm.shape[0] <= 10:  # Only if reasonable number of classes
                class_labels = [f'Q{i}' for i in range(cm.shape[0])]
                ax.set_xticklabels(class_labels, rotation=0)
                ax.set_yticklabels(class_labels, rotation=0)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Calculate and display accuracy from confusion matrix
            cm_accuracy = np.trace(cm) / np.sum(cm)
            st.caption(f"Accuracy from Confusion Matrix: {cm_accuracy:.4f}")
        
        # Classification Report
        st.markdown("### 📋 Classification Report")
        report = selected_metrics['classification_report']
        if isinstance(report, dict):
            # Convert to DataFrame for better display
            report_df = pd.DataFrame(report).transpose()
            # Format numeric columns
            numeric_cols = report_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                report_df[col] = report_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
            st.dataframe(report_df, width='stretch')
            
            # Show summary metrics
            if 'weighted avg' in report_df.index:
                st.markdown("**Weighted Average Metrics:**")
                weighted_avg = report_df.loc['weighted avg']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Precision (Weighted)", weighted_avg.get('precision', 'N/A'))
                with col2:
                    st.metric("Recall (Weighted)", weighted_avg.get('recall', 'N/A'))
                with col3:
                    st.metric("F1-Score (Weighted)", weighted_avg.get('f1-score', 'N/A'))
    else:
        st.warning("⚠️ No metrics found. Please train the models first using train_models.py")
