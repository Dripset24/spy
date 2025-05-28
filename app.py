from flask import Flask, jsonify
import yfinance as yf
import pandas as pd
import joblib
import requests
from datetime import datetime
from ta.volume import OnBalanceVolumeIndicator
from ta.trend import PSARIndicator

app = Flask(__name__)

# --- Discord Webhook ---
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1375065109289893978/t5_rOsW7o5MHZNaJY8KrjkW1PRCYGSUShm_TTlv4OM1QG4-ROfynKd_-nnzAOLMhEXEp"

# --- Features and Models ---
model_features = [
    'obv', 'SPY_volume', 'QQQ_volume_x', 'XLF_volume_x', 'volume_range',
    'XLK_volume_x', 'volume_avg_10', 'volume_avg_20', 'vol_divergence',
    'liquidity_proxy', 'volume', 'volume_QQQ', 'QQQ_volume_y',
    'atr_x_volume', 'XLF_volume_y', 'volume_XLF'
]

models = {
    "Scalp Reversal": joblib.load("Scalp_Reversal_Top16_Model.joblib"),
    "Trend Follow": joblib.load("Trend_Follow_Top16_Model.joblib"),
    "Volatility Breakout": joblib.load("Volatility_Breakout_Top16_Model.joblib")
}

# --- Data Prep ---
def fetch_data(ticker):
    df = yf.download(ticker, period="2d", interval="1m").dropna()
    df["obv"] = OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"]).on_balance_volume()
    df["psar"] = PSARIndicator(high=df["High"], low=df["Low"], close=df["Close"]).psar()
    df["SAR_bullish"] = df["Close"] > df["psar"]
    return df

def engineer_features(spy, qqq, xlk, xlf):
    df = pd.DataFrame()
    df["obv"] = spy["obv"]
    df["SPY_volume"] = spy["Volume"]
    df["QQQ_volume_x"] = qqq["Volume"]
    df["XLF_volume_x"] = xlf["Volume"]
    df["volume_range"] = spy["High"] - spy["Low"]
    df["XLK_volume_x"] = xlk["Volume"]
    df["volume_avg_10"] = spy["Volume"].rolling(10).mean()
    df["volume_avg_20"] = spy["Volume"].rolling(20).mean()
    df["vol_divergence"] = spy["Volume"].diff()
    df["liquidity_proxy"] = spy["Volume"] * (spy["High"] - spy["Low"])
    df["volume"] = spy["Volume"]
    df["volume_QQQ"] = qqq["Volume"]
    df["QQQ_volume_y"] = qqq["Volume"].rolling(2).mean()
    df["atr_x_volume"] = (spy["High"] - spy["Low"]) * spy["Volume"]
    df["XLF_volume_y"] = xlf["Volume"].rolling(2).mean()
    df["volume_XLF"] = xlf["Volume"]
    df["SAR_confirmed"] = spy["SAR_bullish"] & qqq["SAR_bullish"] & xlk["SAR_bullish"]
    df["playbook_strategy"] = "Scalp Reversal"
    return df.dropna()

def get_spy_price():
    return round(yf.Ticker("SPY").history(period="1d", interval="1m")["Close"].iloc[-1], 2)

def send_discord_alert(strategy, signal, confidence, spy_price):
    msg = (
        f"**Prediction Alert**\n"
        f"Strategy: `{strategy}`\n"
        f"Signal: `{signal}` | Confidence: `{confidence:.2%}`\n"
        f"SPY Price: `${spy_price}`\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    try:
        requests.post(DISCORD_WEBHOOK, json={"content": msg})
    except Exception as e:
        print("❌ Discord error:", e)

# --- Routes ---
@app.route('/')
def home():
    return "✅ SPY 0DTE Prediction Web Service is Running"

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Live data
        spy = fetch_data("SPY")
        qqq = fetch_data("QQQ")
        xlk = fetch_data("XLK")
        xlf = fetch_data("XLF")

        df = engineer_features(spy, qqq, xlk, xlf)
        latest = df.iloc[-1]

        strategy = latest["playbook_strategy"]
        model = models[strategy]

        # FIXED: Ensure proper 2D format for model
        input_data = latest[model_features].astype(float).to_dict()
        input_df = pd.DataFrame([input_data])

        signal = int(model.predict(input_df)[0])
        confidence = float(model.predict_proba(input_df).max())
        spy_price = get_spy_price()

        send_discord_alert(strategy, signal, confidence, spy_price)

        return jsonify({
            "strategy": strategy,
            "signal": signal,
            "confidence": round(confidence, 4),
            "spy_price": spy_price,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Launch ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
