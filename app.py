
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
# --- Conditional installation of joblib ---
try:
    import joblib
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
    import joblib

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
@st.cache_resource # Use st.cache_resource for models and scaler
def load_artifacts():
    """Load all trained models, scaler, and metrics"""
    models = {}
    model_files = {
        'Logistic Regression': 'model/logistic_regression.pkl',
        'Decision Tree': 'model/decision_tree.pkl',
        'K-Nearest Neighbor': 'model/knn.pkl',
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

    metrics_data = None
    metrics_path = 'model/metrics.json'
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
        except Exception as e:
            st.warning(f"Could not load metrics: {str(e)}")

    # Load X_train_scaled descriptive statistics
    x_train_scaled_desc = None
    x_train_scaled_desc_path = 'model/X_train_scaled_description.csv'
    if os.path.exists(x_train_scaled_desc_path):
        try:
            x_train_scaled_desc = pd.read_csv(x_train_scaled_desc_path, index_col=0)
        except Exception as e:
            st.warning(f"Could not load X_train_scaled_description.csv: {str(e)}")

    return models, scaler, metrics_data, x_train_scaled_desc

# Load data
try:
    models, scaler, all_model_results, x_train_scaled_description = load_artifacts()
    # In your previous step, label_mapping was explicitly NOT added to metrics.json.
    # So, we need to define it here or load it from another source if needed for display.
    # For this demonstration, we'll define a default. If you saved a label_encoder.pkl,
    # you could load it here and reconstruct label_mapping_reverse.
    label_mapping_reverse = {0: 'Low', 1: 'Moderate', 2: 'High'}

    metrics = {}
    confusion_matrices = {}
    classification_reports = {}

    if all_model_results:
        for model_name, result in all_model_results.items():
            metrics[model_name] = {
                'accuracy': result.get('accuracy'),
                'auc': result.get('auc'),
                'precision': result.get('precision'),
                'recall': result.get('recall'),
                'f1': result.get('f1'),
                'mcc': result.get('mcc')
            }
            confusion_matrices[model_name] = result.get('confusion_matrix')
            classification_reports[model_name] = result.get('classification_report')

except Exception as e:
    st.error(f"Error loading models/metrics: {str(e)}")
    models, scaler = {}, None
    metrics, confusion_matrices, classification_reports = None, None, None
    x_train_scaled_description = None
    label_mapping_reverse = {0: 'Low', 1: 'Moderate', 2: 'High'} # Defaulting if error

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
    if not metrics:
        st.error("""
        ⚠️ **Models and metrics not found!**

        Please ensure you have:
        1. Trained the models and saved them to the 'model/' directory.
        2. The `model/` directory contains:
           - `metrics.json`
           - `*.pkl` files for all 6 models
           - `scaler.pkl`
        """)
    else:
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
            if confusion_matrices and selected_model in confusion_matrices:
                st.markdown("### 📈 Confusion Matrix")
                cm = np.array(confusion_matrices[selected_model])
                fig, ax = plt.subplots(figsize=(10, 8))

                class_labels = [label_mapping_reverse[i] for i in sorted(label_mapping_reverse.keys())]
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           cbar_kws={'label': 'Count'}, linewidths=0.5,
                           xticklabels=class_labels, yticklabels=class_labels)
                ax.set_xlabel('Predicted Category', fontsize=12, fontweight='bold')
                ax.set_ylabel('Actual Category', fontsize=12, fontweight='bold')
                ax.set_title(f'Confusion Matrix - {selected_model}', fontsize=14, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # Calculate and display accuracy from confusion matrix
                cm_accuracy = np.trace(cm) / np.sum(cm)
                st.caption(f"Accuracy from Confusion Matrix: {cm_accuracy:.4f}")
            else:
                st.warning("Confusion matrix data not available.")

        # Classification Report
        if classification_reports and selected_model in classification_reports:
            st.markdown("### 📋 Classification Report")
            report = classification_reports[selected_model]
            if isinstance(report, dict):
                # Convert to DataFrame for better display
                report_df = pd.DataFrame(report).transpose()
                # Format numeric columns
                numeric_cols = report_df.select_dtypes(include=np.number).columns
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
            st.warning("Classification report data not available.")

# Page 2: Predict CO Level
elif page == "🔮 Predict CO Level":
    st.header("Predict Atmospheric CO Level")

    st.markdown("Enter the sensor readings and meteorological data below to predict the CO level.")

    # Input fields for features
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            pt08_s1_co = st.number_input('PT08.S1(CO) (Tin oxide sensor)', min_value=0.0, value=1000.0, format="%.2f")
            c6h6_gt = st.number_input('C6H6(GT) (Benzene)', min_value=0.0, value=10.0, format="%.2f")
            pt08_s2_nmhc = st.number_input('PT08.S2(NMHC) (Titania sensor)', min_value=0.0, value=900.0, format="%.2f")
            nox_gt = st.number_input('NOx(GT) (NOx concentration)', min_value=0.0, value=150.0, format="%.2f")
            pt08_s3_nox = st.number_input('PT08.S3(NOx) (Tungsten oxide sensor NOx)', min_value=0.0, value=1000.0, format="%.2f")

        with col2:
            no2_gt = st.number_input('NO2(GT) (NO2 concentration)', min_value=0.0, value=100.0, format="%.2f")
            pt08_s4_no2 = st.number_input('PT08.S4(NO2) (Tungsten oxide sensor NO2)', min_value=0.0, value=1500.0, format="%.2f")
            pt08_s5_o3 = st.number_input('PT08.S5(O3) (Indium oxide sensor O3)', min_value=0.0, value=1000.0, format="%.2f")
            temp = st.number_input('T (Temperature °C)', min_value=-50.0, max_value=50.0, value=15.0, format="%.2f")
            rh = st.number_input('RH (Relative Humidity %)', min_value=0.0, max_value=100.0, value=50.0, format="%.2f")

        with col3:
            ah = st.number_input('AH (Absolute Humidity)', min_value=0.0, value=1.0, format="%.4f")
            hour = st.slider('Hour of Day', min_value=0, max_value=23, value=12)
            day_of_week = st.slider('Day of Week', min_value=0, max_value=6, value=2, help='0=Monday, 6=Sunday')
            day_of_year = st.slider('Day of Year', min_value=1, max_value=366, value=180)
            month = st.slider('Month of Year', min_value=1, max_value=12, value=6)

        submit_button = st.form_submit_button("Predict CO Level")

    if submit_button:
        if models and scaler:
            input_data = pd.DataFrame([[pt08_s1_co, c6h6_gt, pt08_s2_nmhc, nox_gt, pt08_s3_nox,
                                        no2_gt, pt08_s4_no2, pt08_s5_o3, temp, rh, ah,
                                        hour, day_of_week, day_of_year, month]],
                                      columns=feature_names)

            # Scale the input data
            input_data_scaled = scaler.transform(input_data)

            # Select model for prediction (e.g., the best performing one or a default)
            # For now, let's use XGBoost as it was the best performer
            if 'XGBoost' in models:
                model_to_use = models['XGBoost']
                prediction_raw = model_to_use.predict(input_data_scaled)

                # Robustly extract the scalar prediction
                if isinstance(prediction_raw, np.ndarray):
                    prediction_encoded = prediction_raw.item() # Extracts scalar from 0-d or 1-d array
                else:
                    prediction_encoded = int(prediction_raw) # Assume it's already a scalar integer

                prediction_proba = model_to_use.predict_proba(input_data_scaled)[0]

                predicted_category = label_mapping_reverse.get(prediction_encoded, 'Unknown')
                confidence = np.max(prediction_proba)

                st.subheader("Prediction Result")
                st.success(f"The predicted Atmospheric CO Level is: **{predicted_category}**")
                st.info(f"Confidence: **{confidence:.2f}**")

                # Optional: Show probabilities for all classes
                st.write("Class Probabilities:")
                proba_df = pd.DataFrame({
                    'Category': [label_mapping_reverse[i] for i in sorted(label_mapping_reverse.keys())],
                    'Probability': prediction_proba
                })
                st.dataframe(proba_df.set_index('Category'))
            else:
                st.warning("XGBoost model not loaded. Please ensure models are trained and saved correctly.")
        else:
            st.warning("Models or scaler not loaded. Cannot make predictions.")

# Page 3: Dataset Info
elif page == "📈 Dataset Info":
    st.header("Dataset Information")

    # Dataset Overview
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Air Quality Dataset

        **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/air+quality)

        **Description:**
        The dataset contains hourly averaged responses from an array of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device. The device was located on the field in a polluted area within an Italian city, at road level.
        The target variable, `CO_Level`, is a categorical variable derived from `CO(GT)` and classified into 'low', 'moderate', and 'high' based on quantiles.

        **Citation:**
        De Vito, S., Massera, G., Piga, M., Martinelli, P., & Di Francia, G. (2008). On field calibration of an electronic nose for the monitoring of pollutant gasses, Sensors and Actuators B: Chemical, Volume 129, Issue 2, 2008, Pages 750-757, ISSN 0925-4005.
        """
        )

    with col2:
        st.info("""
        **Quick Stats:**
        - 📊 Instances: ~9357 (original, after preprocessing ~7674)
        - 🔢 Features: 15
        - 🎯 Classes: 3 ('Low', 'Moderate', 'High')
        - 📈 Type: Multi-class Classification
        """
        )

    # Feature Information
    st.subheader("🔍 Feature Information")

    features_df = pd.DataFrame({
        'Feature': [
            'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)',
            'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH',
            'hour', 'day_of_week', 'day_of_year', 'month'
        ],
        'Unit': [
            'sensor response', 'microg/m^3', 'sensor response', 'microg/m^3', 'sensor response',
            'microg/m^3', 'sensor response', 'sensor response', '°C', '%', 'AH',
            'hour (0-23)', 'day of week (0-6)', 'day of year (1-366)', 'month (1-12)'
        ],
        'Description': [
            'Tin oxide sensor response (CO)',
            'True hourly averaged Benzene concentration',
            'Titania sensor response (NMHC)',
            'True hourly averaged NOx concentration',
            'Tungsten oxide sensor response (NOx)',
            'True hourly averaged NO2 concentration',
            'Tungsten oxide sensor response (NO2)',
            'Indium oxide sensor response (O3)',
            'Temperature',
            'Relative Humidity',
            'Absolute Humidity',
            'Hour of the day',
            'Day of the week',
            'Day of the year',
            'Month of the year'
        ]
    })

    st.dataframe(features_df, width='stretch', hide_index=True)

    # Dataset Exploration (Simplified for Streamlit deployment without full original data)
    st.subheader("📊 Dataset Exploration")
    st.info("To explore the dataset interactively, please refer to the original notebook where EDA was performed.")
    st.markdown("**Key statistics from training data (after preprocessing and scaling):**")
    if x_train_scaled_description is not None: # Check if descriptive statistics were loaded
        st.dataframe(x_train_scaled_description)
    else:
        st.warning("Training data statistics not directly available in the Streamlit app's context.")

    # Models Information
    st.subheader("🤖 Implemented Models")

    models_info = pd.DataFrame({
        'Model': [
            'Logistic Regression',
            'Decision Tree',
            'K-Nearest Neighbors (KNN)',
            'Naive Bayes (Gaussian)',
            'Random Forest',
            'XGBoost'
        ],
        'Type': [
            'Linear',
            'Tree-based',
            'Instance-based',
            'Probabilistic',
            'Ensemble',
            'Ensemble (Boosting)'
        ],
        'Key Characteristics': [
            'Fast, interpretable, works well with linear relationships',
            'Non-linear, interpretable, prone to overfitting',
            'Non-parametric, sensitive to feature scaling',
            'Fast, works well with independent features',
            'Reduces overfitting, handles non-linear relationships',
            'High performance, handles complex patterns'
        ]
    })

    st.dataframe(models_info, width='stretch', hide_index=True)

    # Evaluation Metrics
    st.subheader("📏 Evaluation Metrics")

    metrics_info = pd.DataFrame({
        'Metric': [
            'Accuracy',
            'AUC Score',
            'Precision',
            'Recall',
            'F1 Score',
            'MCC'
        ],
        'Description': [
            'Overall correctness of predictions',
            'Area under ROC curve - model discrimination ability',
            'Accuracy of positive predictions',
            'Coverage of actual positive cases',
            'Harmonic mean of precision and recall',
            'Matthews Correlation Coefficient - balanced measure'
        ],
        'Range': [
            '0 to 1 (higher is better)',
            '0 to 1 (higher is better)',
            '0 to 1 (higher is better)',
            '0 to 1 (higher is better)',
            '0 to 1 (higher is better)',
            '-1 to 1 (higher is better)'
        ]
    })

    st.dataframe(metrics_info, width='stretch', hide_index=True)

# Footer
st.markdown("___")
st.markdown(
    "<div style='text-align: center; color: #666;'>ML Assignment 2 - Atmospheric CO Level Prediction</div>",
    unsafe_allow_html=True
)

