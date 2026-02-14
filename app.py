
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Define the directory where models are saved
models_dir = 'model'

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

# Load scaler and the best model
@st.cache_resource
def load_artifacts():
    scaler = None
    best_model = None
    label_mapping_inv = {0: 'Low', 1: 'Moderate', 2: 'High'}

    try:
        scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        best_model = joblib.load('best_model.pkl')
        st.sidebar.success("Models and scaler loaded successfully!")
    except FileNotFoundError as e:
        st.sidebar.error(f"Error loading model or scaler: {e}. Make sure 'scaler.pkl' and 'best_model.pkl' are in the root directory (or 'model' subdirectory).")
        st.stop()
    except Exception as e:
        st.sidebar.error(f"An unexpected error occurred while loading artifacts: {e}")
        st.stop()

    return scaler, best_model, label_mapping_inv

scaler, best_model, label_mapping_inv = load_artifacts()

# Load metrics dataframe
@st.cache_data
def load_metrics_df():
    metrics_path = 'model_metrics.csv'
    if os.path.exists(metrics_path):
        try:
            return pd.read_csv(metrics_path)
        except Exception as e:
            st.warning(f"Could not load model comparison metrics: {e}")
            return None
    st.warning("model_metrics.csv not found. Model comparison data will not be available.")
    return None

metrics_df = load_metrics_df()

# Define feature names (order matters for prediction)
feature_names = [
    'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)',
    'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH',
    'hour', 'day_of_week', 'day_of_year', 'month'
]


# Page 1: Model Comparison
if page == "📊 Model Comparison":
    st.header("Overall Model Performance Comparison")

    if metrics_df is not None and not metrics_df.empty:
        st.dataframe(metrics_df.set_index('Model'), use_container_width=True)

        st.markdown("---")
        st.subheader("Visualizing Model Performance")

        metrics_to_plot = [col for col in metrics_df.columns if col != 'Model']
        n_rows = (len(metrics_to_plot) + 2) // 3
        n_cols = 3

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
        axes = axes.flatten()

        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            sns.barplot(x='Model', y=metric, data=metrics_df, palette='viridis', ax=ax, hue='Model', legend=False)
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=12)
            ax.set_xlabel('Model', fontsize=10)
            ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10)
            ax.tick_params(axis='x', rotation=45, labelsize=9)
            ax.tick_params(axis='y', labelsize=9)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Model comparison data not available. Please ensure 'model_metrics.csv' is in the project directory.")

# Page 2: Predict CO Level
elif page == "🔮 Predict CO Level":
    st.header("Predict Atmospheric CO Level")

    st.info("Adjust the parameters in the sidebar to get a prediction for the CO level.")

    st.subheader("Input Parameters")
    user_input_display = {}
    user_input = {}
    
    cols = st.columns(3)
    
    for i, feature in enumerate(feature_names):
        with cols[i % 3]:
            if feature == 'hour':
                user_input[feature] = st.slider(f'Hour ({feature})', 0, 23, 12)
            elif feature == 'day_of_week':
                user_input[feature] = st.slider(f'Day of Week ({feature}) (0=Mon, 6=Sun)', 0, 6, 2)
            elif feature == 'month':
                user_input[feature] = st.slider(f'Month ({feature})', 1, 12, 6)
            elif feature == 'day_of_year':
                user_input[feature] = st.slider(f'Day of Year ({feature})', 1, 366, 180)
            elif feature == 'T':
                user_input[feature] = st.number_input(f'Temperature (°C) ({feature})', value=15.0, min_value=-50.0, max_value=50.0, step=0.1)
            elif feature == 'RH':
                user_input[feature] = st.number_input(f'Relative Humidity (%) ({feature})', value=50.0, min_value=0.0, max_value=100.0, step=0.1)
            elif feature == 'AH':
                user_input[feature] = st.number_input(f'Absolute Humidity ({feature})', value=1.0, min_value=0.0, max_value=10.0, step=0.01)
            elif feature == 'C6H6(GT)':
                 user_input[feature] = st.number_input(f'Benzene Concentration (microg/m^3) ({feature})', value=8.0, min_value=0.0, max_value=100.0, step=0.1)
            else:
                user_input[feature] = st.number_input(f'{feature}', value=1000.0, min_value=0.0, max_value=3000.0, step=10.0)
        user_input_display[feature] = user_input[feature]

    input_df = pd.DataFrame([user_input])
    st.write(input_df)


    if st.button('Predict CO Level'):
        if scaler is None or best_model is None:
            st.error("Model artifacts not loaded. Please ensure they are available in the 'model' directory.")
        else:
            try:
                scaled_input = scaler.transform(input_df)
                scaled_input_df = pd.DataFrame(scaled_input, columns=feature_names)

                prediction_numeric = best_model.predict(scaled_input_df)
                predicted_co_level = label_mapping_inv[prediction_numeric[0]]

                st.subheader('Prediction Result')
                st.success(f'The Predicted CO Level is: **{predicted_co_level}**')
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.exception(e)


elif page == "📈 Dataset Info":
    st.header("Dataset Information: Air Quality (UCI)")

    st.markdown("""
    ### Dataset Overview

    The dataset used is the "Air Quality (UCI)" dataset, sourced from the UCI Machine Learning Repository. It contains measurements of various air pollutants and meteorological parameters recorded hourly from March 2004 to February 2005 in an Italian city.

    The target variable, `CO_Level`, is a categorical variable derived from `CO(GT)` (True hourly averaged CO concentration) and classified into 'low', 'moderate', and 'high' based on quantiles.

    **Key characteristics:**
    - Time-series data, hourly measurements.
    - Focus on atmospheric pollutants and related environmental factors.
    """)

    st.subheader("🔍 Feature Information")

    features_df = pd.DataFrame({
        'Feature': [
            '`CO(GT)`', '`PT08.S1(CO)`', '`C6H6(GT)`', '`PT08.S2(NMHC)`',
            '`NOx(GT)`', '`PT08.S3(NOx)`', '`NO2(GT)`', '`PT08.S4(NO2)`',
            '`PT08.S5(O3)`', '`T`', '`RH`', '`AH`',
            '`hour`', '`day_of_week`', '`day_of_year`', '`month`'
        ],
        'Description': [
            'True hourly averaged CO concentration (mg/m^3)',
            'Tin oxide sensor response (CO)',
            'True hourly averaged Benzene concentration (microg/m^3)',
            'Titania sensor response (NMHC)',
            'True hourly averaged NOx concentration (microg/m^3)',
            'Tungsten oxide sensor response (NOx)',
            'True hourly averaged NO2 concentration (microg/m^3)',
            'Tungsten oxide sensor response (NO2)',
            'Indium oxide sensor response (O3)',
            'Temperature (°C)',
            'Relative Humidity (%)',
            'Absolute Humidity (AH)',
            'Hour of the day',
            'Day of the week (0=Monday, 6=Sunday)',
            'Day of the year',
            'Month of the year'
        ]
    })
    st.dataframe(features_df, use_container_width=True)

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
    st.dataframe(models_info, use_container_width=True)

    st.subheader("📏 Evaluation Metrics")

    metrics_info = pd.DataFrame({
        'Metric': [
            'Accuracy',
            'Precision (Weighted)',
            'Recall (Weighted)',
            'F1 Score (Weighted)',
            'ROC AUC Score (Weighted OVr)',
            'MCC (Matthews Correlation Coefficient)'
        ],
        'Description': [
            'Overall correctness of predictions',
            'Weighted average of precision across classes',
            'Weighted average of recall across classes',
            'Weighted average of F1-score across classes',
            'Weighted average of Area Under the Receiver Operating Characteristic Curve (One-vs-Rest)',
            'A balanced measure that considers true and false positives and negatives'
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
    st.dataframe(metrics_info, use_container_width=True)


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>ML Assignment - Atmospheric CO Level Prediction</div>",
    unsafe_allow_html=True
)
