import sys
import os
import pandas as pd
import numpy as np
import datetime
import pickle
import matplotlib.pyplot as plt

# Import quant functions from your script
from spy_quant_bot import (
    engineer_features_core,
    classify_signal_type,
    classify_playbook,
    dynamic_tp_sl,
)

# ===== CONFIGURATION =====
SPY_CSV = "SPY_5m_2023.csv"  # Must be present in your folder
CAPITAL = 10000
OPTION_DELTA = 0.5
FEE_PER_TRADE = 0.0
HORIZONS = [10, 15, 30]
ENTRY_THRESHOLDS = {10: 0.5, 15: 0.5, 30: 0.5}

# === ML Model File Paths ===
MODEL_ENTRY_PATHS = {
    10: "spy_entry_model_10.pkl",
    15: "spy_entry_model_15.pkl",
    30: "spy_entry_model_30.pkl"
}
SCALER_PATH = "spy_scaler.pkl"
FEATURE_COLUMNS = [
    'psar_dir','rsi7','rsi_trend','momentum_diff','macd_line','macd_signal','macd_cross',
    'macd_hist','volume_roc','rolling_sharpe','rel_strength_50',
    'atr_norm','ema_gap_9_21','vwap','vwap_dist','bullish_engulfing','bearish_engulfing',
    'bos', 'choc', 'liq_grab', 'order_block','realized_vol'
]

# ===== LOAD DATA =====
if not os.path.exists(SPY_CSV):
    print(f"ERROR: File {SPY_CSV} not found. Please download historical SPY 5m data as {SPY_CSV}.")
    sys.exit(1)
raw_spy = pd.read_csv(SPY_CSV, index_col=0, parse_dates=True)
features = engineer_features_core(raw_spy.copy())

# ===== LOAD MODELS & SCALER =====
models_entry = {}
if not os.path.exists(SCALER_PATH):
    print(f"ERROR: Scaler file {SCALER_PATH} not found. Train and save your scaler first.")
    sys.exit(1)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
for hr in HORIZONS:
    if not os.path.exists(MODEL_ENTRY_PATHS[hr]):
        print(f"ERROR: Model file {MODEL_ENTRY_PATHS[hr]} not found. Train and save your ML model for horizon {hr}.")
        sys.exit(1)
    with open(MODEL_ENTRY_PATHS[hr], "rb") as f:
        models_entry[hr] = pickle.load(f)

# ===== BACKTEST LOGIC =====
capital = {h: CAPITAL/len(HORIZONS) for h in HORIZONS}
positions = []  # Each position: dict with keys incl. 'horizon'
trade_log = []
cum_pnl = {h: 0.0 for h in HORIZONS}
last_entry_idx = {h: None for h in HORIZONS}
COOLDOWN_BARS = {10: 10, 15: 15, 30: 30}  # Prevent re-entry for this many bars after exit

for i in range(len(features)):
    row = features.iloc[i:i+1]
    now = row.index[0]
    price = row['Close'].iloc[0]

    # --- Check all horizons for entry/exit decisions ---
    for horizon in HORIZONS:
        # --- Check for open position in this horizon ---
        open_pos = next((p for p in positions if p['horizon'] == horizon), None)

        # --- ML Entry Logic ---
        X = row[FEATURE_COLUMNS].values
        X_scaled = scaler.transform(X)
        entry_prob = float(models_entry[horizon].predict_proba(X_scaled)[0][1])
        is_bull = row['psar_dir'].iloc[0] == 1
        signal_type = classify_signal_type(row)
        playbook = classify_playbook(row)
        entry_thresh = ENTRY_THRESHOLDS[horizon]

        # --- ENTRY (no open pos, not in cooldown, threshold passed) ---
        in_cooldown = last_entry_idx[horizon] is not None and (i - last_entry_idx[horizon] < COOLDOWN_BARS[horizon])
        if not open_pos and not in_cooldown and entry_prob > entry_thresh:
            option_price = 1.0
            tp, sl = dynamic_tp_sl(option_price, entry_prob)
            pos = {
                'horizon': horizon,
                'entry_idx': i,
                'entry_time': now,
                'entry_price': option_price,
                'entry_spy': price,
                'is_call': is_bull,
                'tp': tp,
                'sl': sl,
                'signal_type': signal_type,
                'playbook': playbook,
                'holding_period': 0,
                'entry_prob': entry_prob
            }
            positions.append(pos)
            last_entry_idx[horizon] = i

        # --- EXIT/EVALUATE ---
        if open_pos:
            holding_period = i - open_pos['entry_idx']
            exit_signal = False
            exit_reason = ""
            option_exit_price = open_pos['entry_price'] + (
                (price - open_pos['entry_spy']) * (OPTION_DELTA if open_pos['is_call'] else -OPTION_DELTA)
            )
            # TP/SL exit
            if option_exit_price >= open_pos['tp']:
                exit_signal = True
                exit_reason = "tp"
            elif option_exit_price <= open_pos['sl']:
                exit_signal = True
                exit_reason = "sl"
            # Time-based exit
            elif holding_period >= horizon:
                exit_signal = True
                exit_reason = "time"
            # End-of-data exit
            if i == len(features) - 1:
                exit_signal = True
                exit_reason = "eod"

            if exit_signal:
                trade_pnl = (option_exit_price - open_pos['entry_price']) * 100 - FEE_PER_TRADE
                capital[horizon] += trade_pnl
                cum_pnl[horizon] += trade_pnl
                trade_log.append({
                    "horizon": horizon,
                    "entry_time": open_pos["entry_time"],
                    "exit_time": now,
                    "entry_price": open_pos["entry_price"],
                    "exit_price": option_exit_price,
                    "entry_spy": open_pos["entry_spy"],
                    "exit_spy": price,
                    "is_call": open_pos["is_call"],
                    "holding_period": holding_period,
                    "tp": open_pos["tp"],
                    "sl": open_pos["sl"],
                    "pnl": trade_pnl,
                    "signal_type": open_pos["signal_type"],
                    "playbook": open_pos["playbook"],
                    "exit_reason": exit_reason,
                    "cum_pnl": cum_pnl[horizon],
                    "entry_prob": open_pos["entry_prob"]
                })
                positions = [p for p in positions if not (p['horizon'] == horizon)]

# ===== RESULTS =====
trades = pd.DataFrame(trade_log)
if len(trades) == 0:
    print("No trades generated. Check your entry logic, ML models, or data.")
    sys.exit(0)

print("Total trades:", len(trades))
print("Gross PnL (all horizons):", trades['pnl'].sum())
print("Winrate:", (trades['pnl'] > 0).mean())
print("Breakdown by horizon:")
print(trades.groupby('horizon')['pnl'].describe())
print("Signal Type breakdown:")
print(trades.groupby('signal_type')['pnl'].describe())
print("Playbook breakdown:")
print(trades.groupby('playbook')['pnl'].describe())
trades.to_csv("backtest_trades_full_ml.csv")
print("\nTrade log saved to backtest_trades_full_ml.csv")

# Plot equity curve for each horizon and total
plt.figure(figsize=(12, 6))
for hr in HORIZONS:
    hr_trades = trades[trades['horizon'] == hr].copy()
    if len(hr_trades):
        hr_trades['cum_pnl'].plot(label=f"Horizon {hr}m")
if len(trades):
    trades.sort_values(['exit_time'], inplace=True)
    trades['total_cum_pnl'] = trades['pnl'].cumsum()
    trades['total_cum_pnl'].plot(color='black', linewidth=2, label='Total')
plt.xlabel("Trade # / Chronological")
plt.ylabel("Cumulative PnL ($)")
plt.title("Backtest Equity Curve (All Horizons, ML)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
