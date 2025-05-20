from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd
import joblib
import os

# Load your trained model
model = joblib.load("spy_xgb_model 2.pkl")

# Create the Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])  # Convert input to DataFrame
        prob = model.predict_proba(df)[0][1]  # Probability of class = 1
        label = int(prob > 0.5)  # Binary prediction
        return jsonify({
            "probability": round(prob, 4),
            "prediction": label
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
