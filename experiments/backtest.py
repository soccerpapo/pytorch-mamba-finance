import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import pandas as pd

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(root_dir)

try:
    from src.model import DifferentiableTrader
    from src.data import get_market_data # <--- STANDARD SOURCE
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SYMBOL = 'BTC-USD'
INITIAL_CAPITAL = 10000 
TRANSACTION_FEE = 0.001
MODEL_PATH = os.path.join(current_dir, "champion.pth") # <--- STANDARD NAME
ENTRY_THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]

def run_backtest():
    print(f"💰 STARTING BACKTEST (Colab Optimized)")
    
    # 1. Load Data
    max_retries = 3
    features = None
    prices = None
    
    for attempt in range(max_retries):
        try:
            print(f"📥 Downloading {SYMBOL} (Attempt {attempt+1}/{max_retries})...")
            features = get_market_data(SYMBOL, period="2y", interval="1h")
            
            import yfinance as yf
            raw_df = yf.download(SYMBOL, period="2y", interval="1h", progress=False, auto_adjust=True)
            
            if len(raw_df) == 0: raise ValueError("Empty Data")
            if isinstance(raw_df.columns, pd.MultiIndex):
                raw_df.columns = raw_df.columns.get_level_values(0)
                
            prices = raw_df['Close'].values.squeeze()
            print("✅ Data Ready.")
            break 
        except Exception as e:
            print(f"⚠️ Download Error: {e}")
            if attempt < max_retries - 1: time.sleep(5)
            else: return

    # 2. Load Champion
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: {MODEL_PATH} not found.")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        d_model = checkpoint['config']['d_model']
        input_dim = checkpoint['config'].get('input_dim', 6)
        state_dict = checkpoint['model_state_dict']
    else:
        d_model = 32
        input_dim = 6
        state_dict = checkpoint

    print(f"🧠 Loading Model (Input Dim: {input_dim}, D_Model: {d_model})...")
    model = DifferentiableTrader(input_dim=input_dim, d_model=d_model).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 3. Generate Signals (BATCHED FOR GPU SPEED)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)

    print("\n🧠 Generating Signals (Batched)...")
    all_signals = []
    
    # Create all windows first
    windows = []
    for t in range(50, len(features_norm) - 1):
        windows.append(features_norm[t-50:t])
    
    # Process in large batches (Much faster on Colab)
    BATCH_SIZE = 2048
    
    with torch.no_grad():
        for i in range(0, len(windows), BATCH_SIZE):
            batch_windows = windows[i:i+BATCH_SIZE]
            
            # Convert batch to tensor and move to GPU once
            tensor = torch.tensor(np.array(batch_windows), dtype=torch.float32).to(DEVICE)
            
            out = model(tensor)
            
            # Grab only the last timestep signal
            signals = out[:, -1, 0].cpu().numpy()
            all_signals.extend(signals)
            
    all_signals = np.array(all_signals)
    
    # Align lengths
    price_changes = prices[51:] / prices[50:-1] - 1 
    min_len = min(len(all_signals), len(price_changes))
    all_signals = all_signals[:min_len]
    price_changes = price_changes[:min_len]

    # 4. Hysteresis Scan
    print("\n📊 HYSTERESIS REPORT (Sticky Logic):")
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

    # 5. GENERATE PLOT
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
    plt.plot(equity_curve, label='Model Strategy', color='lime')
    plt.plot(btc_curve, label='Bitcoin Buy & Hold', color='gray', alpha=0.5)
    plt.title(f'Strategy Performance (Thresh: {best_thresh})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('backtest_result.png')
    print("✅ Plot saved to 'backtest_result.png'")

if __name__ == "__main__":
    run_backtest()
