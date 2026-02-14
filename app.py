import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Define the directory where models are saved
models_dir = 'model'

# Ensure the models directory exists (if running locally without prior execution)
if not os.path.exists(models_dir):
    st.error(f"Model directory '{models_dir}' not found. Please ensure models are saved in the correct location.")
    st.stop()

# Load the scaler and the best model
try:
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    best_model = joblib.load('best_model.pkl')
    # The original labels were 'Low', 'Moderate', 'High' mapped to 0, 1, 2 respectively.
    # Reconstruct the inverse label mapping directly.
    label_mapping_inv = {0: 'Low', 1: 'Moderate', 2: 'High'}

except FileNotFoundError as e:
    st.error(f"Error loading model or scaler: {e}. Make sure 'scaler.pkl' and 'best_model.pkl' are in the root directory and 'y_test_encoded.pkl' in the 'model' directory.")
    st.stop()

# Define feature names (order matters for prediction)
feature_names = [
    'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)',
    'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH',
    'hour', 'day_of_week', 'day_of_year', 'month'
]

# Streamlit App Title
st.title('Atmospheric CO Level Prediction')
st.markdown('Predict the Carbon Monoxide (CO) level based on various atmospheric parameters.')

# Input fields for features
st.sidebar.header('Input Parameters')

user_input = {}
for feature in feature_names:
    # Provide appropriate default values and ranges for input fields
    # These are illustrative defaults; actual ranges might need to be derived from your dataset's min/max
    if feature == 'hour':
        user_input[feature] = st.sidebar.slider(f'Hour ({feature})', 0, 23, 12)
    elif feature == 'day_of_week':
        user_input[feature] = st.sidebar.slider(f'Day of Week ({feature}) (0=Mon, 6=Sun)', 0, 6, 2)
    elif feature == 'month':
        user_input[feature] = st.sidebar.slider(f'Month ({feature})', 1, 12, 6)
    elif feature == 'day_of_year':
        user_input[feature] = st.sidebar.slider(f'Day of Year ({feature})', 1, 366, 180)
    elif feature == 'T':
        user_input[feature] = st.sidebar.number_input(f'Temperature (°C) ({feature})', value=15.0, min_value=-50.0, max_value=50.0)
    elif feature == 'RH':
        user_input[feature] = st.sidebar.number_input(f'Relative Humidity (%) ({feature})', value=50.0, min_value=0.0, max_value=100.0)
    elif feature == 'AH':
        user_input[feature] = st.sidebar.number_input(f'Absolute Humidity ({feature})', value=1.0, min_value=0.0, max_value=10.0)
    elif feature == 'C6H6(GT)':
         user_input[feature] = st.sidebar.number_input(f'Benzene Concentration (microg/m^3) ({feature})', value=8.0, min_value=0.0, max_value=100.0)
    else:
        # Generic number input for other sensor readings and concentrations
        user_input[feature] = st.sidebar.number_input(f'{feature}', value=1000.0, min_value=0.0, max_value=3000.0)

# Convert user input to DataFrame
input_df = pd.DataFrame([user_input])

# Scale the input features
scaled_input = scaler.transform(input_df)
scaled_input_df = pd.DataFrame(scaled_input, columns=feature_names)

st.subheader('Input Parameters Overview')
st.write(input_df)

# Make prediction
if st.button('Predict CO Level'):
    prediction_numeric = best_model.predict(scaled_input_df)
    # Convert numeric prediction back to original labels
    predicted_co_level = label_mapping_inv[prediction_numeric[0]]

    st.subheader('Prediction Result')
    st.success(f'The Predicted CO Level is: **{predicted_co_level}**')

st.markdown("""
---
### How to Run This App:
1.  Save the code above as `app.py`.
2.  Open your terminal or command prompt.
3.  Navigate to the directory where you saved `app.py`.
4.  Run the command: `streamlit run app.py`
""")


# Write the content to app.py
with open('app.py', 'w') as f:
    f.write('''
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Define the directory where models are saved
models_dir = 'model'

# Ensure the models directory exists (if running locally without prior execution)
if not os.path.exists(models_dir):
    st.error(f"Model directory '{models_dir}' not found. Please ensure models are saved in the correct location.")
    st.stop()

# Load the scaler and the best model
try:
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    best_model = joblib.load('best_model.pkl')
    # The original labels were 'Low', 'Moderate', 'High' mapped to 0, 1, 2 respectively.
    # Reconstruct the inverse label mapping directly.
    label_mapping_inv = {0: 'Low', 1: 'Moderate', 2: 'High'}

except FileNotFoundError as e:
    st.error(f"Error loading model or scaler: {e}. Make sure 'scaler.pkl' and 'best_model.pkl' are in the root directory and 'y_test_encoded.pkl' in the 'model' directory.")
    st.stop()

# Define feature names (order matters for prediction)
feature_names = [
    'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)',
    'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH',
    'hour', 'day_of_week', 'day_of_year', 'month'
]

# Streamlit App Title
st.title('Atmospheric CO Level Prediction')
st.markdown('Predict the Carbon Monoxide (CO) level based on various atmospheric parameters.')

# Input fields for features
st.sidebar.header('Input Parameters')

user_input = {}
for feature in feature_names:
    # Provide appropriate default values and ranges for input fields
    # These are illustrative defaults; actual ranges might need to be derived from your dataset's min/max
    if feature == 'hour':
        user_input[feature] = st.sidebar.slider(f'Hour ({feature})', 0, 23, 12)
    elif feature == 'day_of_week':
        user_input[feature] = st.sidebar.slider(f'Day of Week ({feature}) (0=Mon, 6=Sun)', 0, 6, 2)
    elif feature == 'month':
        user_input[feature] = st.sidebar.slider(f'Month ({feature})', 1, 12, 6)
    elif feature == 'day_of_year':
        user_input[feature] = st.sidebar.slider(f'Day of Year ({feature})', 1, 366, 180)
    elif feature == 'T':
        user_input[feature] = st.sidebar.number_input(f'Temperature (°C) ({feature})', value=15.0, min_value=-50.0, max_value=50.0)
    elif feature == 'RH':
        user_input[feature] = st.sidebar.number_input(f'Relative Humidity (%) ({feature})', value=50.0, min_value=0.0, max_value=100.0)
    elif feature == 'AH':
        user_input[feature] = st.sidebar.number_input(f'Absolute Humidity ({feature})', value=1.0, min_value=0.0, max_value=10.0)
    elif feature == 'C6H6(GT)':
         user_input[feature] = st.sidebar.number_input(f'Benzene Concentration (microg/m^3) ({feature})', value=8.0, min_value=0.0, max_value=100.0)
    else:
        # Generic number input for other sensor readings and concentrations
        user_input[feature] = st.sidebar.number_input(f'{feature}', value=1000.0, min_value=0.0, max_value=3000.0)

# Convert user input to DataFrame
input_df = pd.DataFrame([user_input])

# Scale the input features
scaled_input = scaler.transform(input_df)
scaled_input_df = pd.DataFrame(scaled_input, columns=feature_names)

st.subheader('Input Parameters Overview')
st.write(input_df)

# Make prediction
if st.button('Predict CO Level'):
    prediction_numeric = best_model.predict(scaled_input_df)
    # Convert numeric prediction back to original labels
    predicted_co_level = label_mapping_inv[prediction_numeric[0]]

    st.subheader('Prediction Result')
    st.success(f'The Predicted CO Level is: **{predicted_co_level}**')
''')
print("Streamlit app.py created successfully!")