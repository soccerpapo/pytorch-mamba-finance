import torch
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import os
import sys

# Import Architecture
try:
    from model import DifferentiableTrader
except ImportError:
    print("⚠️ Error: 'model.py' not found in directory.")
    sys.exit()

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SYMBOL = 'BTC-USD'
# Note: In a real deployment, this would be a trained weight file
# MODEL_PATH = 'champion_model.pth' 

print(f"🚀 INITIALIZING LIVE TRADER...")
print(f"   - Device: {DEVICE}")
print(f"   - Target: {SYMBOL}")

# 1. SETUP MODEL
model = DifferentiableTrader(input_dim=2, d_model=32).to(DEVICE)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)) # Uncomment if you have weights
model.eval()

# 2. GET LIVE DATA
print("👀 Watching Market...")
data = yf.download(SYMBOL, period="1mo", interval="1h", progress=False)
if len(data) < 500:
    print("❌ Not enough data.")
    sys.exit()

prices = data['Close'].values.squeeze()
volumes = data['Volume'].values.squeeze()

# 3. PREPARE VISION
log_returns = np.diff(np.log(prices))
log_volume_change = np.diff(np.log(volumes + 1e-8))

# Context Window (Last 500 hours)
log_returns = log_returns[-500:]
log_volume_change = log_volume_change[-500:]

# Normalize (Adaptive Scaling)
scaler = StandardScaler()
features = np.column_stack((log_returns, log_volume_change))
features_norm = scaler.fit_transform(features)

# 4. EXECUTION
input_tensor = torch.tensor(features_norm, dtype=torch.float32).view(1, -1, 2).to(DEVICE)

with torch.no_grad():
    prediction = model(input_tensor)

signal = prediction[0, -1, 0].item()

print("\n" + "="*40)
print(f"🤖 SIGNAL STRENGTH: {signal:+.4f}")
print("="*40)

if signal > 0.3:
    print("🚀 ACTION: STRONG BUY")
elif signal < -0.3:
    print("🔻 ACTION: STRONG SELL")
else:
    print("⚪ ACTION: HOLD (Noise)")
    