from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load your trained model
model = joblib.load("spy_xgb_model 2.pkl")  # Make sure this file is in the same directory

# Create the Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON and convert to DataFrame
        data = request.get_json()
        df = pd.DataFrame(data)

        # Make prediction
        prob = model.predict_proba(df)[0][1]  # Probability of class 1 (CALL)
        label = int(prob > 0.5)

        return jsonify({
            'prediction': label,
            'confidence': round(prob, 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the app
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT',10000))
    app.run(host='0.0.0.0', port=port)
