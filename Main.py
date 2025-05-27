# main.py
import pandas as pd
import time
import joblib
import yfinance as yf
import requests
from datetime import datetime

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

last_entry_time = None
last_exit_time = None
cooldown_minutes = 15
active_position = {}

def get_vix_level():
    try:
        vix = yf.Ticker("^VIX").history(period="1d", interval="1m")
        return round(vix['Close'].iloc[-1], 2) if not vix.empty else None
    except:
        return None

def get_best_spy_0dte_option(signal_type='CALL'):
    try:
        spy = yf.Ticker("SPY")
        spy_price = spy.history(period="1d", interval="1m")['Close'].iloc[-1]
        expiry = datetime.today().strftime('%Y-%m-%d')
        options = spy.option_chain(expiry).calls if signal_type == 'CALL' else spy.option_chain(expiry).puts
        options = options.copy()
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
    except Exception as e:
        print("⚠️ Option selection failed:", e)
        return None

def send_discord_alert(strategy, price, signal_type, contract):
    msg = (
        f"**{signal_type.upper()} ALERT**\n"
        f"Strategy: `{strategy}`\nPrice: ${price:.2f}\n"
        f"Contract: `{contract['symbol']}` | Strike: {contract['strike']} | LTP: ${contract['lastPrice']:.2f}\n"
        f"Volume: {contract['volume']} | OI: {contract['openInterest']}\n"
        f"[Robinhood Link]({contract['url']})\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    try:
        requests.post(DISCORD_WEBHOOK, json={"content": msg})
    except Exception as e:
        print("❌ ALERT ERROR:", e)

def predict_and_alert(latest_row):
    global last_entry_time, last_exit_time, active_position

    strategy = latest_row['playbook_strategy']
    model = models.get(strategy)
    if model is None:
        return

    try:
        input_data = latest_row[model_features].astype(float)
        input_data['obv'] = input_data['obv'] / 1e6
        input_df = pd.DataFrame([input_data.values], columns=model_features)
        signal = model.predict(input_df)[0]
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return

    spy_price = yf.Ticker("SPY").history(period="1d", interval="1m")['Close'].iloc[-1]
    now = datetime.now()
    vix = get_vix_level()

    cooldown_entry_ok = not last_entry_time or (now - last_entry_time).total_seconds() > cooldown_minutes * 60
    cooldown_exit_ok = not last_exit_time or (now - last_exit_time).total_seconds() > cooldown_minutes * 60

    print(f"[{now.strftime('%H:%M:%S')}] {strategy} | Signal: {signal} | SPY: {spy_price:.2f} | VIX: {vix}")

    if signal == 1 and cooldown_entry_ok:
        contract = get_best_spy_0dte_option('CALL')
        if contract:
            send_discord_alert(strategy, spy_price, "entry", contract)
            active_position.update({
                "symbol": contract['symbol'],
                "entry_price": contract['lastPrice'],
                "type": "CALL",
                "partial_tp_hit": False
            })
            last_entry_time = now
    elif signal == -1 and cooldown_exit_ok:
        contract = get_best_spy_0dte_option('PUT')
        if contract:
            send_discord_alert(strategy, spy_price, "exit", contract)
            active_position.update({
                "symbol": contract['symbol'],
                "entry_price": contract['lastPrice'],
                "type": "PUT",
                "partial_tp_hit": False
            })
            last_exit_time = now

def check_pnl_trigger():
    global active_position
    try:
        if not active_position:
            return
        symbol = active_position["symbol"]
        entry_price = active_position["entry_price"]
        option_type = active_position["type"]
        partial_hit = active_position["partial_tp_hit"]
        strike = int(symbol[11:16])
        expiry = datetime.today().strftime('%Y-%m-%d')
        chain = yf.Ticker("SPY").option_chain(expiry)
        options = chain.calls if option_type == "CALL" else chain.puts
        match = options[options["strike"] == strike]
        if match.empty:
            return
        current_price = match["lastPrice"].values[0]
        change = (current_price - entry_price) / entry_price * 100

        if change >= 30:
            requests.post(DISCORD_WEBHOOK, json={"content": f"**FINAL TP HIT (+30%)**: {symbol} | Entry: {entry_price:.2f} → Now: {current_price:.2f}"})
            active_position.clear()
        elif change >= 15 and not partial_hit:
            requests.post(DISCORD_WEBHOOK, json={"content": f"**PARTIAL TP HIT (+15%)**: {symbol} | Entry: {entry_price:.2f} → Now: {current_price:.2f}"})
            active_position["partial_tp_hit"] = True
        elif change <= -20:
            requests.post(DISCORD_WEBHOOK, json={"content": f"**STOP HIT (-20%)**: {symbol} | Entry: {entry_price:.2f} → Now: {current_price:.2f}"})
            active_position.clear()
    except Exception as e:
        print("PnL tracking error:", e)

# --- Live Loop ---
while True:
    try:
        df = pd.read_csv("Tier30_Live_Scored_Output_16Features.csv")
        latest = df.dropna().iloc[-1]
        predict_and_alert(latest)
        check_pnl_trigger()
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ Loop Error: {e}")
    time.sleep(60)
