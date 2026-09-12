from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

# Create Flask app
app = Flask(__name__)

# Load trained model and preprocessing objects
try:
    model = joblib.load('logistic_regression_diabetes_model.joblib')
    scaler = joblib.load('scaler.joblib')
    encoder = joblib.load('encoder.joblib')

    print("Model, scaler, and encoder loaded successfully.")

except Exception as e:
    print(f"Error loading model or preprocessing objects: {e}")
    raise e


# Expected columns
categorical_cols = [
    'gender',
    'smoking_history'
]

numerical_cols = [
    'age',
    'hypertension',
    'heart_disease',
    'bmi',
    'HbA1c_level',
    'blood_glucose_level'
]


@app.route('/')
def home():
    return jsonify({
        "message": "Diabetes Prediction API is running"
    })


@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()

    if not data:
        return jsonify({
            "error": "No JSON data received"
        }), 400

    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data])

        # Check required columns
        required_columns = categorical_cols + numerical_cols

        missing_columns = [
            col for col in required_columns
            if col not in input_df.columns
        ]

        if missing_columns:
            return jsonify({
                "error": "Missing columns",
                "missing": missing_columns
            }), 400

        # Separate categorical and numerical data
        input_categorical = input_df[categorical_cols]
        input_numerical = input_df[numerical_cols]

        # Encode categorical features
        input_encoded = encoder.transform(input_categorical)

        input_encoded_df = pd.DataFrame(
            input_encoded,
            columns=encoder.get_feature_names_out(categorical_cols)
        )

        # Scale numerical features
        input_scaled = scaler.transform(input_numerical)

        input_scaled_df = pd.DataFrame(
            input_scaled,
            columns=numerical_cols
        )

        # Combine processed features
        input_processed = pd.concat(
            [input_scaled_df, input_encoded_df],
            axis=1
        )

        # Prediction
        prediction = model.predict(input_processed)

        prediction_proba = model.predict_proba(input_processed)

        # Result
        result = {
            "prediction": int(prediction[0]),
            "probability_no_diabetes": float(prediction_proba[0][0]),
            "probability_diabetes": float(prediction_proba[0][1])
        }

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# Run locally / Render
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
