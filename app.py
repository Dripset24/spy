from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np

model = joblib.load("spy_xgb_model 2.pkl")
app = Flask(__name__)

@app.route("/")
def home():
    return "✅ SPY model API is live"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("Received JSON:", data)  # <- DEBUG LOG

        df = pd.DataFrame(data)
        print("Converted to DataFrame:", df)

        df = df.astype(np.float32)
        print("Converted to float32")

        prob = model.predict_proba(df)[0][1]
        label = int(prob > 0.5)

        return jsonify({
            "prediction": label,
            "confidence": round(float(prob), 4)
        })

    except Exception as e:
        print("Error:", str(e))  # <- LOG ERROR
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
