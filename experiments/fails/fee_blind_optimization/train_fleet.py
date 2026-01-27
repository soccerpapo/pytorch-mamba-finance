import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import random
import copy
import sys
import os

# Import Architecture
# Assumes model.py is in the same folder
try:
    from model import DifferentiableTrader
except ImportError:
    print("❌ Error: 'model.py' not found. Please ensure it is in the same folder.")
    sys.exit()

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS_PER_RUN = 10   # Fast training per candidate
ITERATIONS = 50       # Total candidates to try (Reduced for demo purposes)
SYMBOL = 'BTC-USD'

def get_real_data(symbol=SYMBOL, period='2y'):
    print(f"📥 Downloading {period} of {symbol}...")
    try:
        data = yf.download(symbol, period=period, interval='1h', progress=False)
        if len(data) == 0: raise ValueError("No data found")
    except Exception as e:
        print(f"❌ Data Error: {e}")
        sys.exit()

    prices = data['Close'].values.squeeze()
    volumes = data['Volume'].values.squeeze()
    
    # Log Returns & Volume Changes
    log_returns = np.diff(np.log(prices))
    log_volume_change = np.diff(np.log(volumes + 1e-8))
    
    features = np.column_stack((log_returns, log_volume_change))
    return features

def make_batches(data, b_size):
    x_batches = []
    # Create sliding windows of 50
    for i in range(len(data) - 50):
        window = data[i:i+50]
        if len(window) == 50:
            x_batches.append(torch.tensor(window, dtype=torch.float32))
    
    # Batch them
    loader = []
    for i in range(0, len(x_batches), b_size):
        batch = x_batches[i:i+b_size]
        if len(batch) > 0:
            loader.append(torch.stack(batch).to(DEVICE))
    return loader

def train_one_candidate(features, run_id):
    # 1. Randomize Hyperparameters (The "Mutation")
    learning_rate = random.choice([0.005, 0.001, 0.0005, 0.0001])
    batch_size = random.choice([32, 64, 128])
    d_model = random.choice([16, 32, 64]) 
    
    # 2. Train/Val Split (80/20)
    split_idx = int(len(features) * 0.8)
    train_data = features[:split_idx]
    val_data = features[split_idx:]
    
    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_data)
    val_norm = scaler.transform(val_data)
    
    train_loader = make_batches(train_norm, batch_size)
    val_loader = make_batches(val_norm, batch_size)

    # 3. Initialize Model
    model = DifferentiableTrader(input_dim=2, d_model=d_model).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 4. Train Loop
    for epoch in range(EPOCHS_PER_RUN):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            positions = model(batch)
            
            # --- THE "FEE-BLIND" OBJECTIVE ---
            # Maximizes raw return without accounting for transaction costs.
            # This is the root cause of the strategy's failure.
            returns = batch[:, :, 0:1] 
            strategy_returns = positions * returns
            loss = -torch.mean(strategy_returns) 
            
            loss.backward()
            optimizer.step()
            
    # 5. Validation
    model.eval()
    val_score = 0
    with torch.no_grad():
        for batch in val_loader:
            positions = model(batch)
            returns = batch[:, :, 0:1]
            val_score += torch.sum(positions * returns).item()
            
    print(f"Run {run_id}: LR={learning_rate} | Size={d_model} | Score={val_score:.4f}")
    return model, val_score, d_model

def main():
    print(f"🔥 STARTING FLEET TRAINING ON {DEVICE}")
    print("   (Experiment: Fee-Blind Optimization)")
    
    data = get_real_data()
        
    best_score = -999999
    best_model = None
    best_config = {}

    for i in range(ITERATIONS):
        candidate, score, size = train_one_candidate(data, i+1)
        
        if score > best_score:
            print(f"🌟 NEW CHAMPION FOUND! (Score: {score:.4f})")
            best_score = score
            best_model = copy.deepcopy(candidate)
            best_config = {'d_model': size}
            torch.save(best_model.state_dict(), "champion_model.pth")
                
    print("\n" + "="*40)
    print(f"🏆 SEARCH COMPLETE.")
    print(f"   - Best Validation Score: {best_score:.4f}")
    print(f"   - Best Model Size: {best_config.get('d_model')}")
    print(f"   - Saved to: champion_model.pth")
    print("="*40)

if __name__ == "__main__":
    main()
    