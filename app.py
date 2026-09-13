from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Create Flask app
app = Flask(__name__)


# ==========================================
# Load Model and Preprocessing Files
# ==========================================

try:
    model = joblib.load("logistic_regression_diabetes_model.joblib")
    scaler = joblib.load("scaler.joblib")
    encoder = joblib.load("encoder.joblib")

    print("Model, scaler, and encoder loaded successfully.")

except Exception as e:
    print(f"Error loading model or preprocessing objects: {e}")
    raise e


# ==========================================
# Feature Definitions
# ==========================================

categorical_cols = [
    "gender",
    "smoking_history"
]

numerical_cols = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level"
]


# ==========================================
# Home Route
# ==========================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Diabetes Prediction API is running",
        "status": "success"
    })


# ==========================================
# Prediction Route
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    # Get JSON data
    data = request.get_json()

    # Check if JSON data exists
    if not data:

        return jsonify({
            "error": "No JSON data received"
        }), 400


    try:

        # --------------------------------------
        # Convert input into DataFrame
        # --------------------------------------

        input_df = pd.DataFrame([data])


        # --------------------------------------
        # Check Required Columns
        # --------------------------------------

        required_columns = (
            categorical_cols + numerical_cols
        )

        missing_columns = [
            column
            for column in required_columns
            if column not in input_df.columns
        ]


        if missing_columns:

            return jsonify({
                "error": "Missing required columns",
                "missing_columns": missing_columns
            }), 400


        # --------------------------------------
        # Separate Categorical Data
        # --------------------------------------

        input_categorical = input_df[categorical_cols]


        # --------------------------------------
        # Separate Numerical Data
        # --------------------------------------

        input_numerical = input_df[numerical_cols]


        # --------------------------------------
        # Encode Categorical Features
        # --------------------------------------

        input_encoded = encoder.transform(
            input_categorical
        )


        input_encoded_df = pd.DataFrame(
            input_encoded,
            columns=encoder.get_feature_names_out(
                categorical_cols
            )
        )


        # --------------------------------------
        # Scale Numerical Features
        # --------------------------------------

        input_scaled = scaler.transform(
            input_numerical
        )


        input_scaled_df = pd.DataFrame(
            input_scaled,
            columns=numerical_cols
        )


        # --------------------------------------
        # Combine Processed Features
        # --------------------------------------

        input_processed = pd.concat(
            [
                input_scaled_df,
                input_encoded_df
            ],
            axis=1
        )


        # --------------------------------------
        # Make Prediction
        # --------------------------------------

        prediction = model.predict(
            input_processed
        )


        prediction_proba = model.predict_proba(
            input_processed
        )


        # --------------------------------------
        # Prepare Response
        # --------------------------------------

        result = {

            "prediction": int(
                prediction[0]
            ),

            "probability_no_diabetes": float(
                prediction_proba[0][0]
            ),

            "probability_diabetes": float(
                prediction_proba[0][1]
            )
        }


        return jsonify(result)


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ==========================================
# Run Application
# ==========================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
