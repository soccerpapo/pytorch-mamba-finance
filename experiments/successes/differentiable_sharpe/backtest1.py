import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(root_dir)

try:
    from src.model import DifferentiableTrader
    from src.data import get_market_data
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SYMBOL = 'BTC-USD'
INITIAL_CAPITAL = 10000 
TRANSACTION_FEE = 0.001
MODEL_PATH = os.path.join(current_dir, "champion_model1.pth")

# 👇 THE WIDE SCANNER CONFIG (Start low to find the signal)
THRESHOLDS_TO_TEST = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

def run_backtest():
    print(f"💰 STARTING SENSITIVITY SCAN (Robust Mode)")
    
    # 1. Load Data (With Retry Logic)
    max_retries = 3
    features = None
    prices = None
    
    for attempt in range(max_retries):
        try:
            print(f"📥 Downloading {SYMBOL} (Attempt {attempt+1}/{max_retries})...")
            
            # Fetch Features
            features = get_market_data(SYMBOL, period="2y", interval="1h")
            
            # Fetch Raw Prices (for PnL calc)
            import yfinance as yf
            raw_df = yf.download(SYMBOL, period="2y", interval="1h", progress=False)
            
            if raw_df is None or len(raw_df) == 0:
                raise ValueError("Received empty data from Yahoo Finance")
                
            prices = raw_df['Close'].values.squeeze()
            
            print("✅ Data Downloaded Successfully.")
            break 
            
        except Exception as e:
            print(f"⚠️ Download Error: {e}")
            if attempt < max_retries - 1:
                print("   Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("❌ Failed to download data after 3 attempts.")
                return

    # 2. Load Champion
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: {MODEL_PATH} not found.")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        d_model = checkpoint['config']['d_model']
        state_dict = checkpoint['model_state_dict']
    else:
        d_model = 32
        state_dict = checkpoint

    model = DifferentiableTrader(input_dim=2, d_model=d_model).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 3. Pre-Calculate All Signals
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)

    print("\n🧠 Generating Signals (One Pass)...")
    all_signals = []
    
    for t in range(50, len(features_norm) - 1):
        window = features_norm[t-50:t]
        tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            s = model(tensor)[0, -1, 0].item()
            all_signals.append(s)
            
    all_signals = np.array(all_signals)
    
    # Ensure prices match the length of signals
    # We started loop at 50, so signals start at index 50
    # Price changes should be from t to t+1
    price_changes = prices[51:] / prices[50:-1] - 1 
    
    # Trim to matching length
    min_len = min(len(all_signals), len(price_changes))
    all_signals = all_signals[:min_len]
    price_changes = price_changes[:min_len]

    # 4. The Scan Loop
    print("\n📊 SENSITIVITY REPORT:")
    print(f"{'THRESH':<10} | {'TRADES':<10} | {'FINAL BAL':<15} | {'PROFIT %':<10}")
    print("-" * 55)

    best_roi = -999
    best_thresh = 0
    
    for thresh in THRESHOLDS_TO_TEST:
        balance = INITIAL_CAPITAL
        position = 0
        trade_count = 0
        
        for i in range(len(all_signals)):
            signal = all_signals[i]
            ret = price_changes[i]
            
            # Logic
            new_pos = 0
            if signal > thresh: new_pos = 1
            elif signal < -thresh: new_pos = -1
            
            # PnL
            if position != 0:
                balance += balance * position * ret
            
            # Fee
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
    print(f"🏆 BEST SETTING: Threshold {best_thresh:.2f} (ROI: {best_roi:.2f}%)")

if __name__ == "__main__":
    run_backtest()
    