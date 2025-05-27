import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import requests
from datetime import datetime
from ta.volume import OnBalanceVolumeIndicator

# Discord webhook
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1375065109289893978/t5_rOsW7o5MHZNaJY8KrjkW1PRCYGSUShm_TTlv4OM1QG4-ROfynKd_-nnzAOLMhEXEp"

# Load models
scalp_model = joblib.load("Scalp_Reversal_Top16_Model.joblib")
trend_model = joblib.load("Trend_Follow_Top16_Model.joblib")
vol_model = joblib.load("Volatility_Breakout_Top16_Model.joblib")

model_features = [
    'obv', 'SPY_volume', 'QQQ_volume_x', 'XLF_volume_x', 'volume_range',
    'XLK_volume_x', 'volume_avg_10', 'volume_avg_20', 'vol_divergence',
    'liquidity_proxy', 'volume', 'volume_QQQ', 'QQQ_volume_y',
    'atr_x_volume', 'XLF_volume_y', 'volume_XLF'
]

def fetch_data(ticker):
    return yf.download(ticker, period="2d", interval="1m").dropna()

def engineer_features():
    spy = fetch_data("SPY")
    qqq = fetch_data("QQQ")
    xlk = fetch_data("XLK")
    xlf = fetch_data("XLF")

    spy['obv'] = OnBalanceVolumeIndicator(close=spy['Close'], volume=spy['Volume']).on_balance_volume()

    df = pd.DataFrame(index=spy.index)
    df['obv'] = spy['obv']
    df['SPY_volume'] = spy['Volume']
    df['QQQ_volume_x'] = qqq['Volume']
    df['XLF_volume_x'] = xlf['Volume']
    df['volume_range'] = spy['High'] - spy['Low']
    df['XLK_volume_x'] = xlk['Volume']
    df['volume_avg_10'] = spy['Volume'].rolling(10).mean()
    df['volume_avg_20'] = spy['Volume'].rolling(20).mean()
    df['vol_divergence'] = spy['Volume'].diff()
    df['liquidity_proxy'] = spy['Volume'] * (spy['High'] - spy['Low'])
    df['volume'] = spy['Volume']
    df['volume_QQQ'] = qqq['Volume']
    df['QQQ_volume_y'] = qqq['Volume'].rolling(2).mean()
    df['atr_x_volume'] = (spy['High'] - spy['Low']) * spy['Volume']
    df['XLF_volume_y'] = xlf['Volume'].rolling(2).mean()
    df['volume_XLF'] = xlf['Volume']

    return df.dropna()

def send_discord_alert(strategy, signal, confidence):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = f"**{strategy.upper()} SIGNAL**: {signal}\nConfidence: {confidence:.2%}\nTime: {timestamp}"
    requests.post(DISCORD_WEBHOOK, json={"content": message})

def predict_and_alert(latest_row):
    for name, model in {
        "Scalp Reversal": scalp_model,
        "Trend Follow": trend_model,
        "Volatility Breakout": vol_model
    }.items():
        try:
            row = latest_row[model_features].astype(float)
            row['obv'] /= 1e6
            X = row.values.reshape(1, -1)
            signal = model.predict(X)[0]
            confidence = model.predict_proba(X).max()
            print(f"{name} | Signal: {signal} | Confidence: {confidence:.2%}")
            send_discord_alert(name, signal, confidence)
        except Exception as e:
            print(f"{name} prediction error:", e)

def main():
    while True:
        try:
            df = engineer_features()
            latest = df.iloc[-1]
            predict_and_alert(latest)
        except Exception as e:
            print(f"[{datetime.now()}] ❌ Loop Error: {e}")
        time.sleep(60)

if __name__ == "__main__":
    main()
