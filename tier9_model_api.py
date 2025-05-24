
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Load the trained binary model
model = joblib.load("tier9_bin_model.pkl")

# Define the expected feature columns (trimmed example, replace with full list later)
FEATURES = [
    'rsi', 'macd', 'cmfVal', 'ema9', 'ema21', 'atr', 'signal_strength',
    'options_flow_score', 'sentiment_score', 'trade_quality_score',
    'regime_encoded', 'playbook_class_encoded'
]

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get("features", {})
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400

    # Create dataframe
    X = pd.DataFrame([data])[FEATURES]

    # Predict
    prob = model.predict_proba(X)[0][1]
    signal = "CALL" if prob > 0.65 else "PUT" if prob < 0.35 else "NEUTRAL"

    # Top contributing features (based on raw input values)
    top_feats = sorted(data.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    top_features = [f[0] for f in top_feats]

    return jsonify({
        "signal": signal,
        "score": round(prob, 4),
        "top_features": top_features
    })

@app.route("/", methods=["GET"])
def root():
    return "SPY Tier9 Prediction API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
