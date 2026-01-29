import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
import copy
import sys
import os

# --- 1. PATH SETUP ---
# Dynamically calculate the path to 'src' so this script runs from anywhere
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(root_dir)

try:
    from src.model import DifferentiableTrader
    from src.data import get_market_data
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS_PER_RUN = 10   
ITERATIONS = 50       
SYMBOL = 'BTC-USD'
# NEW: Force the save path to be inside this specific folder
SAVE_PATH = os.path.join(current_dir, "champion_model.pth")

def make_batches(data, b_size):
    x_batches = []
    for i in range(len(data) - 50):
        window = data[i:i+50]
        if len(window) == 50:
            x_batches.append(torch.tensor(window, dtype=torch.float32))
    
    loader = []
    for i in range(0, len(x_batches), b_size):
        batch = x_batches[i:i+b_size]
        if len(batch) > 0:
            loader.append(torch.stack(batch).to(DEVICE))
    return loader

def train_one_candidate(features, run_id):
    # Mutation
    learning_rate = random.choice([0.005, 0.001, 0.0005, 0.0001])
    batch_size = random.choice([32, 64, 128])
    d_model = random.choice([16, 32, 64]) 
    
    # Train/Val Split
    split_idx = int(len(features) * 0.8)
    train_data = features[:split_idx]
    val_data = features[split_idx:]
    
    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_data)
    val_norm = scaler.transform(val_data)
    
    train_loader = make_batches(train_norm, batch_size)
    val_loader = make_batches(val_norm, batch_size)

    # Init Model
    model = DifferentiableTrader(input_dim=2, d_model=d_model).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Train Loop
    for epoch in range(EPOCHS_PER_RUN):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            positions = model(batch)
            
            # Fee-Blind Loss (The Trap)
            returns = batch[:, :, 0:1] 
            strategy_returns = positions * returns
            loss = -torch.mean(strategy_returns) 
            
            loss.backward()
            optimizer.step()
            
    # Validation
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
    print(f"STARTING FLEET TRAINING ON {DEVICE}")
    print("   (Experiment: Fee-Blind Optimization)")
    print(f"   (Saving to: {SAVE_PATH})")
    
    try:
        data = get_market_data(SYMBOL, period='2y')
    except Exception as e:
        print(f"Data Error: {e}")
        return
        
    best_score = -999999
    best_config = {}

    for i in range(ITERATIONS):
        candidate, score, size = train_one_candidate(data, i+1)
        
        if score > best_score:
            print(f"NEW CHAMPION FOUND! (Score: {score:.4f})")
            best_score = score
            best_config = {'d_model': size}
            
            checkpoint = {
                'model_state_dict': candidate.state_dict(),
                'config': {
                    'd_model': size,
                    'input_dim': 2,
                    'strategy': 'fee_blind'
                },
                'score': best_score
            }
            # UPDATED: Uses the absolute path we defined at the top
            torch.save(checkpoint, SAVE_PATH)
                
    print("\n" + "="*40)
    print(f"SEARCH COMPLETE.")
    print(f"   - Best Validation Score: {best_score:.4f}")
    print(f"   - Saved to: {SAVE_PATH}")
    print("="*40)

if __name__ == "__main__":
    main()
