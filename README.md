# Atmospheric CO Level Prediction Project

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Dataset Description](#dataset-description)
3.  [Data Preprocessing and Feature Engineering](#data-preprocessing-and-feature-engineering)
4.  [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5.  [Model Training and Evaluation](#model-training-and-evaluation)
6.  [Model Comparison](#model-comparison)
7.  [Deployment (Streamlit App)](#deployment-streamlit-app)
8.  [Environment Setup](#environment-setup)

## 1. Project Overview

This project aimed to develop and evaluate several machine learning classification models to predict the Carbon Monoxide (CO) level in the atmosphere. The CO level was categorized into 'low', 'medium', and 'high' based on quantiles of the CO(GT) concentration from the Air Quality dataset. The ultimate goal was to build an interactive Streamlit web application to demonstrate these models and their performance, allowing users to input atmospheric parameters and get real-time CO level predictions.

## 2. Dataset Description

The dataset used was the "Air Quality (UCI)" dataset, sourced from the UCI Machine Learning Repository. It contains measurements of various air pollutants and meteorological parameters recorded hourly from March 2004 to February 2005 in an Italian city. Key features included:

*   `CO(GT)`: True hourly averaged CO concentration (mg/m^3)
*   `PT08.S1(CO)`: Tin oxide sensor response (CO)
*   `C6H6(GT)`: True hourly averaged Benzene concentration (microg/m^3)
*   `PT08.S2(NMHC)`: Titania sensor response (NMHC)
*   `NOx(GT)`: True hourly averaged NOx concentration (microg/m^3)
*   `PT08.S3(NOx)`: Tungsten oxide sensor response (NOx)
*   `NO2(GT)`: True hourly averaged NO2 concentration (microg/m^3)
*   `PT08.S4(NO2)`: Tungsten oxide sensor response (NO2)
*   `PT08.S5(O3)`: Indium oxide sensor response (O3)
*   `T`: Temperature (°C)
*   `RH`: Relative Humidity (%)
*   `AH`: Absolute Humidity (AH)

The target variable, `CO_Level`, is a categorical variable derived from `CO(GT)` and classified into 'low', 'medium', and 'high' based on quantiles.

## 3. Data Preprocessing and Feature Engineering

The `AirQualityUCI.csv` dataset underwent extensive cleaning and preprocessing:

*   **Missing Values**: The placeholder value '-200' was replaced with `np.nan`.
*   **Column Dropping**: Irrelevant columns such as `Unnamed: 15`, `Unnamed: 16`, and `NMHC(GT)` (due to high missingness) were dropped.
*   **Data Type Conversion**: Object columns with comma decimal separators (e.g., `CO(GT)`, `C6H6(GT)`) were converted to numeric (float).
*   **Temporal Feature Creation**: 'Date' and 'Time' columns were combined to create a `DateTime` object, from which `hour`, `day_of_week`, `day_of_year`, and `month` were extracted.
*   **Imputation**: Remaining missing values were filled using forward fill (`ffill()`) followed by backfill (`bfill()`).
*   **Target Variable**: `CO(GT)` was categorized into 'Low', 'Moderate', and 'High' based on quantiles, creating the `air_quality_category` as the target variable.
*   **Data Split and Scaling**: The dataset was split into training (80%) and testing (20%) sets with stratification. Features were then scaled using `StandardScaler` to prepare them for model training.

## 4. Exploratory Data Analysis (EDA)

EDA was performed to understand the distribution, relationships, and patterns within the data. This included:

*   **Descriptive Statistics**: Generated for `X_train_scaled` to understand central tendency, dispersion, and shape.
*   **Histograms**: Visualized the distribution of each scaled feature.
*   **Box Plots**: Showcased the spread and potential outliers for each scaled feature.
*   **Correlation Matrix**: A heatmap was generated to visualize linear relationships between features, identifying highly correlated pairs.
*   **Feature vs. Target Analysis**: Box plots were used to visualize how the distribution of each scaled feature varied across the 'Low', 'Moderate', and 'High' air quality categories.

## 5. Model Training and Evaluation

Six different machine learning classification models were implemented and evaluated on the preprocessed and scaled data:

*   **Logistic Regression**
*   **Decision Tree Classifier**
*   **K-Nearest Neighbor (KNN) Classifier**
*   **Naive Bayes (GaussianNB) Classifier**
*   **Random Forest Classifier**
*   **XGBoost Classifier**

Each model was trained on `X_train_scaled` and `y_train`, and evaluated on `X_test_scaled` and `y_test` using metrics such as Accuracy, Precision, Recall, F1 Score, ROC AUC Score, and Matthews Correlation Coefficient (MCC).

## 6. Model Comparison

After training, all models were compared based on their performance metrics:

| Model               | Accuracy   | Precision   | Recall   | F1 Score   | ROC AUC Score   | MCC Score   |
|:--------------------|:-----------|:------------|:---------|:-----------|:----------------|:------------|
| Logistic Regression | 0.869707   | 0.87031     | 0.869707 | 0.869947   | 0.957197        | 0.798613    |
| Decision Tree       | 0.859283   | 0.859405    | 0.859283 | 0.859281   | 0.891596        | 0.78306     |
| K-Nearest Neighbor  | 0.872964   | 0.875308    | 0.872964 | 0.873656   | 0.964922        | 0.804761    |
| Naive Bayes         | 0.841042   | 0.843611    | 0.841042 | 0.841902   | 0.938832        | 0.755219    |
| Random Forest       | 0.898371   | 0.899933    | 0.898371 | 0.898835   | 0.979700        | 0.84375     |
| XGBoost             | 0.912052   | 0.912813    | 0.912052 | 0.912285   | 0.984400        | 0.864600    |

**Key Findings:**

*   **XGBoost Classifier** emerged as the **best-performing model**, achieving the highest scores across most metrics, notably an Accuracy of 0.9121 and an ROC AUC Score of 0.9844.
*   **Random Forest Classifier** was a close second, demonstrating strong predictive capabilities.
*   **Decision Tree Classifier** generally showed the lowest performance among the ensemble and advanced models.

## 7. Deployment (Streamlit App)

A Streamlit web application (`app.py`) was created to provide an interactive interface for users to predict CO levels. This app:

*   Loads the `best_model.pkl` (XGBoost) and `scaler.pkl` artifacts.
*   Allows users to input various atmospheric parameters via sliders.
*   Scales the user input using the loaded `StandardScaler`.
*   Makes a prediction using the best model.
*   Displays the predicted CO level as 'Low', 'Moderate', or 'High'.


## How to Run Locally

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Streamlit App:**
    ```bash
    streamlit run app.py
    ```

## How to Deploy on Streamlit Cloud

1.  Push this repository to GitHub.
2.  Log in to [Streamlit Community Cloud](https://streamlit.io/cloud).
3.  Click "New app" and select this repository.
4.  Set the **Main file path** to `app.py`.
5.  Click **Deploy**.

Streamlit App Link: https://atmospheric-co-level-prediction.streamlit.app/
