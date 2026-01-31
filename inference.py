import torch
import numpy as np
import os
import sys
import time
from datetime import datetime, timedelta
import pytz
from sklearn.preprocessing import StandardScaler
import pandas as pd

# --- 1. SETUP PATHS & IMPORTS ---
# We add the CURRENT DIRECTORY (Root) to the path so we can import 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from src.model import DifferentiableTrader
    from src.data import get_market_data   # <--- STANDARD SOURCE
except ImportError as e:
    print(f"⚠️ Critical Import Error: {e}")
    print(f"   Ensure you are running this from the root directory.")
    sys.exit(1)

# --- 2. CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SYMBOL = 'BTC-USD'
THRESHOLD = 0.30       # Sticky Hysteresis Entry
LOOKBACK_WINDOW = 500  # Context window
D_MODEL = 32           # Must match training
INPUT_DIM = 6          # Retail Standard (6 features)

# Path to the Consolidated Champion
MODEL_PATH = os.path.join(current_dir, "experiments", "champion.pth")

def wait_for_golden_window():
    """
    Pauses execution until 10 seconds after the hour closes.
    This ensures we have the complete 'Closed Candle' data.
    """
    now = datetime.now(pytz.utc)
    next_slot = (now + timedelta(hours=1)).replace(minute=0, second=10, microsecond=0)
    seconds_to_wait = (next_slot - now).total_seconds()
    
    if seconds_to_wait < 0: seconds_to_wait = 10 

    print(f"   ⏳ Sentinel resting for {seconds_to_wait/60:.1f} minutes...")
    print(f"   (Next Scan: {next_slot.strftime('%H:%M:%S')} UTC)")
    time.sleep(seconds_to_wait)

# --- 3. INITIALIZATION ---
print(f"🚀 INITIALIZING LIVE TRADER (Standard Layout)...")
print(f"   - Device: {DEVICE}")
print(f"   - Target: {SYMBOL}")
print(f"   - Model:  {MODEL_PATH}")

model = DifferentiableTrader(input_dim=INPUT_DIM, d_model=D_MODEL).to(DEVICE)

if os.path.exists(MODEL_PATH):
    print(f"   🧠 Loading Champion Weights...")
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        # Handle both direct state_dict and checkpoint dicts
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("   ✅ Weights loaded successfully.")
    except Exception as e:
        print(f"   ❌ WEIGHT ERROR: {e}")
        print("      (Did you change D_MODEL? It must match the trained file.)")
        sys.exit(1)
else:
    print("   ⚠️ WARNING: No weights found at path! Running in SIMULATION MODE.")

model.eval()
print(f"\n✅ SENTINEL ACTIVE. Scanning hourly at :10 seconds...\n")

# --- 4. MAIN LIFE LOOP ---
while True:
    wait_for_golden_window()
    
    print(f"\n[{datetime.now(pytz.utc).strftime('%H:%M:%S')}] 🕯️ Candle Closed. Fetching Data...")

    try:
        # A. GET DATA (Standard src/data.py)
        # We fetch 1 month to ensure enough buffer for indicators + lookback
        features = get_market_data(SYMBOL, period="1mo", interval="1h")

        # B. CHECK DATA LENGTH
        if len(features) < LOOKBACK_WINDOW:
            print(f"   ❌ Not enough data (Got {len(features)}, need {LOOKBACK_WINDOW}). Retrying next hour.")
            continue

        # C. PREPARE VISION
        # Slice the last 500 hours
        recent_features = features[-LOOKBACK_WINDOW:]
        
        # Adaptive Scaling: Fit scaler ONLY on this window
        scaler = StandardScaler()
        features_norm = scaler.fit_transform(recent_features)

        # D. INFERENCE
        # Shape: [Batch=1, Seq=500, Features=6]
        input_tensor = torch.tensor(features_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            prediction = model(input_tensor)

        # Get the signal for the very last candle
        signal = prediction[0, -1, 0].item()

        # E. DECISION LOGIC (Sticky Hysteresis)
        print("-" * 40)
        print(f"🤖 SIGNAL STRENGTH: {signal:+.4f}")
        print("-" * 40)

        if signal > THRESHOLD:
            print("🚀 ACTION: STRONG BUY (Signal > 0.30)")
            # execute_trade('BUY')
        elif signal < -THRESHOLD:
            print("🔻 ACTION: STRONG SELL (Signal < -0.30)")
            # execute_trade('SELL')
        else:
            print("⚪ ACTION: HOLD (Noise / Low Confidence)")

    except Exception as e:
        print(f"   ❌ Error during inference loop: {e}")
        import traceback
        traceback.print_exc()
        time.sleep(60) # Prevent rapid crash loops
