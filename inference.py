import torch
import numpy as np
import os
import sys
import time
from datetime import datetime, timedelta
import pytz
from sklearn.preprocessing import StandardScaler

# --- 1. SETUP PATHS & IMPORTS ---
# This ensures we can import from the 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from model import DifferentiableTrader
    from data import get_market_data  # <--- WE USE YOUR NEW TOOL!
except ImportError as e:
    print(f"⚠️ Critical Import Error: {e}")
    print("   -> Check that 'src/model.py' and 'src/data.py' exist.")
    sys.exit(1)

# --- 2. CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SYMBOL = 'BTC-USD'
THRESHOLD = 0.3
LOOKBACK_WINDOW = 500  # How many hours the model needs to see
D_MODEL = 32           # MUST match your champion model size
MODEL_PATH = "experiments/01_differentiable_sharpe/champion_model.pth" # Update this path!

def wait_for_golden_window():
    """
    Pauses execution until 10 seconds after the hour closes.
    This ensures we have the complete 'Closed Candle' data.
    """
    now = datetime.now(pytz.utc)
    # Calculate next hour :00:10
    next_slot = (now + timedelta(hours=1)).replace(minute=0, second=10, microsecond=0)
    
    seconds_to_wait = (next_slot - now).total_seconds()
    
    # If we are already past the :00:10 mark slightly, don't wait negative time
    if seconds_to_wait < 0:
        seconds_to_wait = 10 

    print(f"   ⏳ Sentinel resting for {seconds_to_wait/60:.1f} minutes...")
    print(f"   (Next Scan: {next_slot.strftime('%H:%M:%S')} UTC)")
    
    time.sleep(seconds_to_wait)

# --- 3. INITIALIZATION ---
print(f"🚀 INITIALIZING LIVE TRADER...")
print(f"   - Device: {DEVICE}")
print(f"   - Target: {SYMBOL}")
print(f"   - Model Path: {MODEL_PATH}")

# Initialize Model
model = DifferentiableTrader(input_dim=2, d_model=D_MODEL).to(DEVICE)

# Load Weights
if os.path.exists(MODEL_PATH):
    print(f"   🧠 Loading Champion Weights...")
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("   ✅ Weights loaded successfully.")
    except RuntimeError as e:
        print(f"   ❌ WEIGHT ERROR: {e}")
        print("      Did you change D_MODEL? It must match the trained file.")
        sys.exit(1)
else:
    print("   ⚠️ WARNING: No weights found. Using random brain (Simulation Mode).")

model.eval()
print(f"\n✅ SENTINEL ACTIVE. Scanning hourly at :10 seconds...\n")

# --- 4. MAIN LIFE LOOP ---
while True:
    wait_for_golden_window()
    
    print(f"\n[{datetime.now(pytz.utc).strftime('%H:%M:%S')}] 🕯️ Candle Closed. Fetching Data...")

    try:
        # A. GET DATA (Using src/data.py)
        # We fetch 1 month to ensure we have enough for the 500-hour window
        features = get_market_data(SYMBOL, period="1mo", interval="1h")

        # B. CHECK DATA LENGTH
        if len(features) < LOOKBACK_WINDOW:
            print(f"   ❌ Not enough data (Got {len(features)}, need {LOOKBACK_WINDOW}). Retrying next hour.")
            continue

        # C. PREPARE VISION
        # Slice the last 500 hours
        recent_features = features[-LOOKBACK_WINDOW:]
        
        # Adaptive Scaling: Fit scaler ONLY on this window
        # This helps the model adapt to recent volatility
        scaler = StandardScaler()
        features_norm = scaler.fit_transform(recent_features)

        # D. INFERENCE
        input_tensor = torch.tensor(features_norm, dtype=torch.float32).view(1, -1, 2).to(DEVICE)

        with torch.no_grad():
            prediction = model(input_tensor)

        signal = prediction[0, -1, 0].item()

        # E. DECISION LOGIC
        print("-" * 40)
        print(f"🤖 SIGNAL STRENGTH: {signal:+.4f}")
        print("-" * 40)

        if signal > THRESHOLD:
            print("🚀 ACTION: STRONG BUY")
            # execute_trade('BUY') <-- Your future broker API goes here
        elif signal < -THRESHOLD:
            print("🔻 ACTION: STRONG SELL")
            # execute_trade('SELL')
        else:
            print("⚪ ACTION: HOLD (Noise)")

    except Exception as e:
        print(f"   ❌ Error during inference loop: {e}")
        time.sleep(60) # Prevent rapid crash loops
        