# Atmospheric CO Level Prediction using Machine Learning

## a. Problem Statement
This project aims to develop and evaluate several machine learning classification models to predict the Carbon Monoxide (CO) level in the atmosphere. The CO level is categorized into 'low', 'medium', and 'high' based on quantiles of the CO(GT) concentration from the Air Quality dataset. The ultimate goal is to build an interactive Streamlit web application to demonstrate these models and their performance.

## b. Dataset Description
The dataset used is the "Air Quality (UCI)" dataset, sourced from the [Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality). It contains measurements of various air pollutants and meteorological parameters recorded hourly from March 2004 to February 2005 in an Italian city. Key features include:

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

The target variable, `CO_Level`, is a categorical variable derived from `CO(GT)` and classified into 'low', 'moderate', and 'high' based on quantiles.

## c. Models Used
Six machine learning classification models were implemented and evaluated on scaled data to predict atmospheric CO levels. The models are Logistic Regression, Decision Tree, K-Nearest Neighbor (KNN), Naive Bayes (GaussianNB), Random Forest, and XGBoost.

### Comparison Table
| ML Model Name       | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
|:--------------------|:---------|:----------|:----------|:-------|:---------|:----------|
| Logistic Regression | 0.8697   | 0.9572    | 0.8703    | 0.8697 | 0.8699   | 0.7986    |
| Decision Tree       | 0.8593   | 0.8916    | 0.8594    | 0.8593 | 0.8593   | 0.7831    |
| K-Nearest Neighbor  | 0.8730   | 0.9649    | 0.8753    | 0.8730 | 0.8737   | 0.8048    |
| Naive Bayes         | 0.8410   | 0.9388    | 0.8436    | 0.8410 | 0.8419   | 0.7552    |
| Random Forest       | 0.8984   | 0.9797    | 0.8999    | 0.8984 | 0.8988   | 0.8437    |
| XGBoost             | 0.9068   | 0.9831    | 0.9076    | 0.9068 | 0.9071   | 0.8565    |

### Observations on Model Performance
| ML Model Name       | Observation about Model Performance                                                                                                                                                                                                                              |
|:--------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression | Showed competitive performance with strong predictive capabilities across all metrics, benefiting significantly from feature scaling. Achieved a high AUC score, indicating good class separability.                                                       |
| Decision Tree       | Exhibited the lowest performance among the ensemble models. While interpretable, it might be prone to overfitting or less capable of capturing complex relationships compared to more sophisticated models.                                             |
| K-Nearest Neighbor  | Performed well, particularly after feature scaling. Its effectiveness underscores the importance of proper data normalization for distance-based algorithms. Achieved high AUC, precision, and recall.                                                    |
| Naive Bayes         | Provided a decent baseline performance. Its assumption of feature independence might not always hold true for atmospheric data, potentially limiting its overall accuracy compared to more complex models.                                                 |
| Random Forest       | Demonstrated strong performance, closely trailing XGBoost. Its ensemble nature effectively reduces overfitting and handles non-linear relationships well, making it a robust choice for this dataset.                                                         |
| XGBoost             | Achieved the highest overall performance across all evaluation metrics. Its boosting approach, which sequentially builds trees to correct errors of previous trees, proved highly effective in capturing the underlying patterns in the air quality data. |

## Step 6: Deploy on Streamlit Community Cloud
To deploy this application on Streamlit Community Cloud, follow these steps:

1.  Go to https://streamlit.io/cloud
2.  Sign in using your GitHub account.
3.  Click “New App”.
4.  Select your repository containing the `app.py` file, `requirements.txt`, and the `model/` directory with all saved artifacts.
5.  Choose the branch (usually `main` or `master`).
6.  Select `app.py` as the main file path.
7.  Click Deploy.
