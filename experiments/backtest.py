import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(root_dir)

from src.model import DifferentiableTrader
from src.data import get_market_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SYMBOL = 'BTC-USD'
INITIAL_CAPITAL = 10000
TRANSACTION_FEE = 0.001
MODEL_PATH = os.path.join(current_dir, "champion.pth")
ENTRY_THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]

# --- CRITICAL FIX: MATCH DATA DIMENSION ---
INPUT_DIM = 4

def run_backtest():
    print(f"💰 STARTING BACKTEST (Colab Optimized)")
    try:
        # This returns 4 columns now
        features = get_market_data(SYMBOL, period="2y", interval="1h")

        # Download prices for calculating returns
        import yfinance as yf
        raw_df = yf.download(SYMBOL, period="2y", interval="1h", progress=False, auto_adjust=True)
        if isinstance(raw_df.columns, pd.MultiIndex): raw_df.columns = raw_df.columns.get_level_values(0)
        prices = raw_df['Close'].values.squeeze()
    except Exception as e:
        print(f"⚠️ Download Error: {e}")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: {MODEL_PATH} not found.")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    d_model = checkpoint['config']['d_model']

    # --- CRITICAL FIX: USE INPUT_DIM=4 ---
    model = DifferentiableTrader(input_dim=INPUT_DIM, d_model=d_model).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)
    print("\n🧠 Generating Signals (Batched)...")

    windows = []
    for t in range(50, len(features_norm) - 1):
        windows.append(features_norm[t-50:t])

    BATCH_SIZE = 2048
    all_signals = []
    with torch.no_grad():
        for i in range(0, len(windows), BATCH_SIZE):
            batch_windows = windows[i:i+BATCH_SIZE]
            tensor = torch.tensor(np.array(batch_windows), dtype=torch.float32).to(DEVICE)
            out = model(tensor)
            all_signals.extend(out[:, -1, 0].cpu().numpy())

    all_signals = np.array(all_signals)
    price_changes = prices[51:] / prices[50:-1] - 1
    min_len = min(len(all_signals), len(price_changes))
    all_signals = all_signals[:min_len]
    price_changes = price_changes[:min_len]

    print("\n📊 HYSTERESIS REPORT:")
    print(f"{'ENTRY':<10} | {'TRADES':<10} | {'FINAL BAL':<15} | {'PROFIT %':<10}")
    print("-" * 55)
    best_roi = -999
    best_thresh = 0

    for thresh in ENTRY_THRESHOLDS:
        balance = INITIAL_CAPITAL
        position = 0
        trade_count = 0
        for i in range(len(all_signals)):
            signal = all_signals[i]
            ret = price_changes[i]
            new_pos = position
            if position == 0:
                if signal > thresh: new_pos = 1
                elif signal < -thresh: new_pos = -1
            elif position == 1:
                if signal < 0: new_pos = 0
                if signal < -thresh: new_pos = -1
            elif position == -1:
                if signal > 0: new_pos = 0
                if signal > thresh: new_pos = 1
            if position != 0: balance += balance * position * ret
            if new_pos != position:
                balance -= balance * TRANSACTION_FEE
                trade_count += 1
            position = new_pos
        profit_pct = (balance - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        print(f"{thresh:<10.2f} | {trade_count:<10} | ${balance:,.2f}      | {profit_pct:+.2f}%")
        if profit_pct > best_roi:
            best_roi = profit_pct
            best_thresh = thresh

    print("-" * 55)
    print(f"🏆 BEST SETTING: Entry {best_thresh:.2f} (ROI: {best_roi:.2f}%)")

    equity_curve = [INITIAL_CAPITAL]
    btc_curve = [INITIAL_CAPITAL]
    position = 0
    balance = INITIAL_CAPITAL
    for i in range(len(all_signals)):
        signal = all_signals[i]
        ret = price_changes[i]
        new_pos = position
        if position == 0:
            if signal > best_thresh: new_pos = 1
            elif signal < -best_thresh: new_pos = -1
        elif position == 1:
            if signal < 0: new_pos = 0
            if signal < -best_thresh: new_pos = -1
        elif position == -1:
            if signal > 0: new_pos = 0
            if signal > best_thresh: new_pos = 1
        if position != 0: balance += balance * position * ret
        if new_pos != position: balance -= balance * TRANSACTION_FEE
        position = new_pos
        equity_curve.append(balance)
        btc_curve.append(btc_curve[-1] * (1 + ret))

    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label='Strategy', color='lime')
    plt.plot(btc_curve, label='Bitcoin Buy & Hold', color='gray', alpha=0.5)
    plt.title(f'Performance (Thresh: {best_thresh})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_backtest()
