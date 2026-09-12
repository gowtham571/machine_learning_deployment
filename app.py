import flask
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Create a Flask app instance
app = Flask(__name__)

# Load the trained model and preprocessing objects
try:
    model = joblib.load('logistic_regression_diabetes_model.joblib')
    scaler = joblib.load('scaler.joblib')
    encoder = joblib.load('encoder.joblib')
    print("Model, scaler, and encoder loaded successfully.")
except Exception as e:
    print(f"Error loading model or preprocessing objects: {e}")
    # Exit or handle the error appropriately
    exit()

# Define the expected categorical and numerical columns based on training data
categorical_cols = ['gender', 'smoking_history'] # Make sure these match the training phase
numerical_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level'] # Make sure these match the training phase

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()

        # Convert input data to a Pandas DataFrame
        input_df = pd.DataFrame([data])
        
        # Ensure all expected columns are present, fill missing with 0 or appropriate default if needed
        # This step is crucial if the input data might not contain all features or in the correct order
        # For simplicity, assuming input_df has all columns as they appeared in training X
        
        # Separate categorical and numerical data from the input
        input_categorical = input_df[categorical_cols]
        input_numerical = input_df[numerical_cols]

        # Preprocess the input data
        input_encoded = encoder.transform(input_categorical)
        input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(categorical_cols))
        
        input_scaled = scaler.transform(input_numerical)
        input_scaled_df = pd.DataFrame(input_scaled, columns=numerical_cols)
        
        # Concatenate processed features
        input_processed = pd.concat([input_scaled_df, input_encoded_df], axis=1)

        # Make prediction
        prediction = model.predict(input_processed)
        prediction_proba = model.predict_proba(input_processed)

        # Return the prediction as a JSON response
        result = {
            'prediction': int(prediction[0]), # Convert numpy int to Python int
            'probability_no_diabetes': prediction_proba[0][0],
            'probability_diabetes': prediction_proba[0][1]
        }
        return jsonify(result)

# To run the app:
if __name__ == '__main__':
 app.run(debug=True, port=5000)

# Note: In a Colab environment, running app.run() directly might block the notebook.
# For deployment, consider using a WSGI server like Gunicorn and tools like ngrok for local testing.
