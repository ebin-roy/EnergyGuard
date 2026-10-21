import csv
from io import BytesIO, StringIO
import json
from pathlib import Path

from flask import Flask, make_response, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)


@app.route("/")
@app.route("/analyze")
def analyze():
    return render_template("analyze.html")


@app.route("/predict", methods=["POST"])
def predict():
    uploaded_file = request.files.get("csv_file")

    if not uploaded_file or uploaded_file.filename == "":
        return "Please select a CSV file.", 400

    if not uploaded_file.filename.lower().endswith(".csv"):
        return "Please upload a CSV file.", 400

    try:
        # Read uploaded CSV
        df = pd.read_csv(BytesIO(uploaded_file.read()))

        if df.empty:
            return "The uploaded CSV is empty.", 400

        # Load trained model and threshold
        model_path = (
            Path(__file__).resolve().parent
            / "model"
            / "electricity_theft_model_package.pkl"
        )

        package = joblib.load(model_path)
        model = package["model"]
        threshold = package["threshold"]

        # Get the feature names used during training
        if not hasattr(model, "feature_names_in_"):
            return (
                "The model does not contain saved feature names.",
                400,
            )

        feature_columns = list(model.feature_names_in_)

        # Check that all required consumption columns exist
        missing_columns = [
            col for col in feature_columns if col not in df.columns
        ]

        if missing_columns:
            return (
                f"CSV is missing {len(missing_columns)} required "
                "consumption columns. Upload the compatible "
                "SGCC dataset.",
                400,
            )

        # Select features in the correct order
        X_new = df[feature_columns].copy()

        # Convert consumption readings to numeric
        for col in feature_columns:
            X_new[col] = pd.to_numeric(X_new[col], errors="coerce")

        # Reject columns with no usable readings
        empty_columns = X_new.columns[X_new.isna().all()].tolist()

        if empty_columns:
            return (
                f"{len(empty_columns)} consumption columns have "
                "no usable numeric readings. Check the CSV.",
                400,
            )

        # Fill missing readings with column means
        X_new = X_new.fillna(X_new.mean())

        # Confirm no missing values remain
        if X_new.isna().any().any():
            return "Missing values remain. Please check the CSV.", 400

        # Validate consumption readings
        if (X_new < 0).any().any():
            return "Consumption readings cannot be negative.", 400

        # Predict anomaly probabilities
        probabilities = model.predict_proba(X_new)[:, 1]

        # Apply the saved classification threshold
        predictions = (probabilities >= threshold).astype(int)

        # Prepare results
        results = pd.DataFrame({
            "Consumer": range(1, len(df) + 1),
            "Prediction": [
                "Anomalous" if value == 1 else "Normal" for value in predictions
            ],
            "Anomaly Probability (%)": [
                round(float(value) * 100, 2) for value in probabilities
            ],
        })

        total = len(results)
        anomalous = int((predictions == 1).sum())
        normal = total - anomalous

        return render_template(
            "result.html",
            title="CSV Anomaly Analysis",
            prediction="Random Forest Prediction Completed",
            rows=total,
            columns=len(df.columns),
            normal=normal,
            anomalous=anomalous,
            results=results.to_dict(orient="records"),
            source="ml",
        )

    except Exception:
        app.logger.exception("CSV prediction failed")
        return (
            "Unable to analyze this CSV. Check the terminal "
            "for the detailed error.",
            400,
        )


@app.route("/download-results", methods=["POST"])
def download_results():
    data = request.form.get("results_json")

    if not data:
        return "No prediction results available to download.", 400

    try:
        results = json.loads(data)

        if not isinstance(results, list) or not results:
            return "No prediction results available to download.", 400

        output = StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "Consumer",
                "Prediction",
                "Anomaly Probability (%)",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

        response = make_response(output.getvalue())
        response.headers["Content-Type"] = "text/csv; charset=utf-8"
        response.headers["Content-Disposition"] = (
            "attachment; filename=energyguard_predictions.csv"
        )

        return response

    except (ValueError, TypeError):
        return "Unable to generate the CSV download.", 400


if __name__ == "__main__":
    app.run(debug=True)