"""REST API serving a classifier that predicts whether an NBA rookie
will remain in the league for more than 5 years"""
import logging
import pickle
from pathlib import Path

# data processing
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Decision threshold on the predicted probability, as chosen in the notebook
THRESHOLD = 0.5

BASE_DIR = Path(__file__).resolve().parent


def load_pickle(path):
    """Load a pickled object, or return None if the file is missing"""
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except FileNotFoundError:
        logger.warning("%s not found. Train the model first.", path)
        return None
    logger.info("%s loaded", path)
    return obj


model = load_pickle(BASE_DIR / "model.pkl")
model_columns = load_pickle(BASE_DIR / "model_columns.pkl")


def parse_query(payload):
    """Turn the JSON payload into a one-row float DataFrame in training order.
    Raises ValueError if the payload is malformed, a feature is missing,
    or a value is not numeric (null is accepted)."""
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError("Send exactly one player, as a JSON object or a list.")

    query = pd.DataFrame(payload)
    missing = [col for col in model_columns if col not in query.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    return query[model_columns].astype(float)


@app.route('/predict', methods=['POST'])
def predict():
    """Predict whether one rookie will remain in the NBA for more than 5 years.
    Expects a JSON list with one record."""
    if model is None or model_columns is None:
        return jsonify({"error": "Model not loaded. Train the model first."}), 503
    
    try:
        query = parse_query(request.get_json(silent=True))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    try:
        proba = model.predict_proba(query)[0]
    except Exception:
        logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failure."}), 500

    prediction = int(proba[1] >= THRESHOLD)
    label = (
        "Player will remain in the NBA for more than 5 years"
        if prediction == 1
        else "Player will not remain in the NBA for more than 5 years"
    )
    # return the prediction and the probabilities
    return jsonify({"prediction": label, "confidence": round(float(proba[prediction]), 2)})


if __name__ == '__main__':
    app.run()
