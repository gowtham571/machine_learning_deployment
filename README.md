# Diabetes Prediction using Machine Learning

A Flask-based machine learning application that predicts the likelihood of diabetes using a trained Logistic Regression model.

## Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Joblib

## Features

- Accepts patient health information
- Preprocesses numerical and categorical features
- Uses a trained Logistic Regression model
- Returns diabetes prediction probability
- Provides a REST API using Flask

## Input Features

- Gender
- Age
- Hypertension
- Heart Disease
- Smoking History
- BMI
- HbA1c Level
- Blood Glucose Level

## Project Files

- `app.py` – Flask application
- `logistic_regression_diabetes_model.joblib` – Trained ML model
- `scaler.joblib` – Numerical feature scaler
- `encoder.joblib` – Categorical feature encoder
- `requirements.txt` – Required Python packages

## API Endpoint

### POST `/predict`

Send patient information as JSON to receive a prediction.

## Deployment

The application is deployed using Render.
