from flask import Flask, jsonify
import pandas as pd
import yfinance as yf
import numpy as np
import joblib
import requests
from datetime import datetime
from ta.volume import OnBalanceVolumeIndicator
from ta.trend import PSARIndicator

app = Flask(__name__)

DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1375065109289893978/t5_rOsW7o5MHZNaJY8KrjkW1PRCYGSUShm_TTlv4OM1QG4-ROfynKd_-nnzAOLMhEXEp"

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
    df["vol_divergence"] = spy["Volume"] - spy["Volume"].shift(1)
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

def get_best_spy_0dte_option(signal_type='CALL'):
    spy = yf.Ticker("SPY")
    spy_price = spy.history(period="1d", interval="1m")['Close'].iloc[-1]
    expiry = datetime.today().strftime('%Y-%m-%d')
    options = spy.option_chain(expiry).calls if signal_type == 'CALL' else spy.option_chain(expiry).puts
    options['strike_diff'] = abs(options['strike'] - spy_price)
    best = options.sort_values(by=['strike_diff', 'volume', 'openInterest'], ascending=[True, False, False]).iloc[0]
    symbol = f"SPY{expiry.replace('-', '')}{int(best['strike']):05d}{'C' if signal_type == 'CALL' else 'P'}"
    return {
        'symbol': symbol,
        'strike': best['strike'],
        'lastPrice': best['lastPrice'],
        'volume': best['volume'],
        'openInterest': best['openInterest'],
        'url': f"https://robinhood.com/options/{symbol}"
    }

def send_discord_alert(strategy, price, signal_type, contract):
    msg = (
        f"**{signal_type.upper()} ALERT**\n"
        f"Strategy: `{strategy}`\nPrice: ${price:.2f}\n"
        f"Contract: `{contract['symbol']}` | Strike: {contract['strike']} | LTP: ${contract['lastPrice']:.2f}\n"
        f"Volume: {contract['volume']} | OI: {contract['openInterest']}\n"
        f"[Robinhood Link]({contract['url']})\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    requests.post(DISCORD_WEBHOOK, json={"content": msg})

@app.route("/")
def home():
    return "SPY 0DTE Prediction Web Service is Running"

@app.route("/run")
def run_model():
    try:
        spy = fetch_data("SPY")
        qqq = fetch_data("QQQ")
        xlk = fetch_data("XLK")
        xlf = fetch_data("XLF")
        df = engineer_features(spy, qqq, xlk, xlf)
        latest = df.iloc[-1]
        strategy = latest['playbook_strategy']
        model = models[strategy]
        input_data = latest[model_features].astype(float)
        input_data['obv'] = input_data['obv'] / 1e6
        input_df = pd.DataFrame([input_data], columns=model_features)
        signal = model.predict(input_df)[0]
        spy_price = spy['Close'].iloc[-1]
        contract = get_best_spy_0dte_option('CALL' if signal == 1 else 'PUT')
        send_discord_alert(strategy, spy_price, "entry" if signal == 1 else "exit", contract)
        return jsonify({"signal": int(signal), "strategy": strategy, "contract": contract})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
