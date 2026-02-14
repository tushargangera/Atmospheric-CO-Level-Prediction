import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the best model and scaler
best_model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app title and description
st.title('Atmospheric CO Level Prediction')
st.write('Enter the atmospheric parameters to predict the Carbon Monoxide (CO) level (Low, Moderate, or High).')

# Input features from the user
st.header('Input Atmospheric Parameters')

# Define min/max/default for each feature based on your dataset's scaled values
# (These are approximate ranges, you might want to fine-tune based on X_train.describe() or specific domain knowledge)

# Input for numerical features
PT08S1_CO = st.slider('PT08.S1 (CO) Sensor Response', float(scaler.data_min_[0]), float(scaler.data_max_[0]), float(scaler.mean_[0]))
C6H6_GT = st.slider('Benzene (C6H6(GT))', float(scaler.data_min_[1]), float(scaler.data_max_[1]), float(scaler.mean_[1]))
PT08S2_NMHC = st.slider('PT08.S2 (NMHC) Sensor Response', float(scaler.data_min_[2]), float(scaler.data_max_[2]), float(scaler.mean_[2]))
NOx_GT = st.slider('NOx (NOx(GT))', float(scaler.data_min_[3]), float(scaler.data_max_[3]), float(scaler.mean_[3]))
PT08S3_NOx = st.slider('PT08.S3 (NOx) Sensor Response', float(scaler.data_min_[4]), float(scaler.data_max_[4]), float(scaler.mean_[4]))
NO2_GT = st.slider('NO2 (NO2(GT))', float(scaler.data_min_[5]), float(scaler.data_max_[5]), float(scaler.mean_[5]))
PT08S4_NO2 = st.slider('PT08.S4 (NO2) Sensor Response', float(scaler.data_min_[6]), float(scaler.data_max_[6]), float(scaler.mean_[6]))
PT08S5_O3 = st.slider('PT08.S5 (O3) Sensor Response', float(scaler.data_min_[7]), float(scaler.data_max_[7]), float(scaler.mean_[7]))
T = st.slider('Temperature (T)', float(scaler.data_min_[8]), float(scaler.data_max_[8]), float(scaler.mean_[8]))
RH = st.slider('Relative Humidity (RH)', float(scaler.data_min_[9]), float(scaler.data_max_[9]), float(scaler.mean_[9]))
AH = st.slider('Absolute Humidity (AH)', float(scaler.data_min_[10]), float(scaler.data_max_[10]), float(scaler.mean_[10]))
hour = st.slider('Hour of Day', 0, 23, 12)
day_of_week = st.slider('Day of Week (0=Monday, 6=Sunday)', 0, 6, 2)
day_of_year = st.slider('Day of Year', 1, 365, 182)
month = st.slider('Month', 1, 12, 6)

# Create a DataFrame from user inputs
input_data = pd.DataFrame([[PT08S1_CO, C6H6_GT, PT08S2_NMHC, NOx_GT, PT08S3_NOx,
                            NO2_GT, PT08S4_NO2, PT08S5_O3, T, RH, AH,
                            hour, day_of_week, day_of_year, month]],
                            columns=['PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)',
                                     'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH',
                                     'hour', 'day_of_week', 'day_of_year', 'month'])

# Scale the input data
# Note: The scaler was fit on X_train which included 'CO(GT)' and 'air_quality_category' originally. 
# However, the input_data for prediction should only contain the features. 
# Ensure the columns in input_data match the columns X_train_scaled was trained on.

# For simplicity, let's assume the order of columns in input_data matches X_train_scaled after preprocessing.
# A more robust solution would involve storing feature names used during training.

# It's important that input_data has the same column order as X_train when scaler was fitted.
# If the scaler was fitted on X_train, and X_train has 15 columns as seen from the shapes, 
# then input_data should also have 15 columns in the exact same order.

# The current slider values are *unscaled*. We need to scale them for prediction.
# To correctly scale individual inputs, you would usually create a dummy DataFrame 
# with one row for the input and then apply the scaler.
# Let's adjust the sliders to take raw input values, and then scale the single row.

# Correcting the slider ranges to more realistic raw values based on the original dataset's describe() output.
# You can find these ranges from df.describe() for the original (unscaled) features.
# For example, if T ranged from -10 to 40, RH from 0 to 100, etc.

# Re-defining input sliders to take *raw* values, not scaled values.
# The values below are educated guesses based on typical ranges for these sensors/measurements.
PT08S1_CO = st.slider('PT08.S1 (CO) Sensor Response (Raw)', 700.0, 2100.0, 1100.0) 
C6H6_GT = st.slider('Benzene (C6H6(GT)) (microg/m^3)', 0.0, 60.0, 10.0)
PT08S2_NMHC = st.slider('PT08.S2 (NMHC) Sensor Response (Raw)', 300.0, 2500.0, 900.0)
NOx_GT = st.slider('NOx (NOx(GT)) (microg/m^3)', 0.0, 1500.0, 200.0)
PT08S3_NOx = st.slider('PT08.S3 (NOx) Sensor Response (Raw)', 300.0, 2000.0, 900.0)
NO2_GT = st.slider('NO2 (NO2(GT)) (microg/m^3)', 0.0, 500.0, 100.0)
PT08S4_NO2 = st.slider('PT08.S4 (NO2) Sensor Response (Raw)', 200.0, 2500.0, 1500.0)
PT08S5_O3 = st.slider('PT08.S5 (O3) Sensor Response (Raw)', 200.0, 2500.0, 1000.0)
T = st.slider('Temperature (T) (°C)', -10.0, 40.0, 20.0)
RH = st.slider('Relative Humidity (RH) (%)', 0.0, 100.0, 50.0)
AH = st.slider('Absolute Humidity (AH)', 0.0, 3.0, 1.0)
hour = st.slider('Hour of Day', 0, 23, 12)
day_of_week = st.slider('Day of Week (0=Monday, 6=Sunday)', 0, 6, 2)
day_of_year = st.slider('Day of Year', 1, 366, 182) # Max 366 for leap years
month = st.slider('Month', 1, 12, 6)

# Create a DataFrame from user inputs (raw values)
input_data = pd.DataFrame([[PT08S1_CO, C6H6_GT, PT08S2_NMHC, NOx_GT, PT08S3_NOx,
                            NO2_GT, PT08S4_NO2, PT08S5_O3, T, RH, AH,
                            hour, day_of_week, day_of_year, month]],
                            columns=['PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)',
                                     'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH',
                                     'hour', 'day_of_week', 'day_of_year', 'month'])

# Scale the input data
scaled_input_data = scaler.transform(input_data)

# Make prediction
if st.button('Predict CO Level'):
    prediction_encoded = best_model.predict(scaled_input_data)
    
    # Map the encoded prediction back to original labels
    # The label mapping was {'Low': 0, 'Moderate': 1, 'High': 2}
    label_map_reverse = {0: 'Low', 1: 'Moderate', 2: 'High'}
    predicted_label = label_map_reverse[prediction_encoded[0]]
    
    st.success(f'The predicted CO level is: **{predicted_label}**')

st.write("--- Developed by Tushar Gangera --- ")
