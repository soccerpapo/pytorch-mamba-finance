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
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up 3 levels: differentiable_sharpe -> successes -> experiments -> ROOT
root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(root_dir)

try:
    from src.model import DifferentiableTrader
    from src.data import get_market_data
    # THE SECRET SAUCE: Import the Sharpe Loss
    from src.loss import differentiable_sharpe_loss
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS_PER_RUN = 15   # Slightly longer training for stability
ITERATIONS = 50       
SYMBOL = 'BTC-USD'
SAVE_PATH = os.path.join(current_dir, "champion_model.pth")

# We match the broker fee (0.1%) so the model learns the cost of business
TRAIN_FEE = 0.001 

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
    learning_rate = random.choice([0.001, 0.0005, 0.0001]) # Slower learning for stability
    batch_size = random.choice([64, 128])
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
            returns = batch[:, :, 0:1] 
            
            # THE FIX: Differentiable Sharpe Loss
            # This penalizes the model if it trades too much (via transaction_cost)
            loss = differentiable_sharpe_loss(positions, returns, transaction_cost=TRAIN_FEE)
            
            loss.backward()
            optimizer.step()
            
    # Validation (Using Sharpe as the score, not raw profit)
    model.eval()
    val_sharpe = 0
    batches = 0
    with torch.no_grad():
        for batch in val_loader:
            positions = model(batch)
            returns = batch[:, :, 0:1]
            # We calculate the negative sharpe, so we multiply by -1 to get the positive score
            s_loss = differentiable_sharpe_loss(positions, returns, transaction_cost=TRAIN_FEE)
            val_sharpe += (-s_loss.item())
            batches += 1
            
    avg_sharpe = val_sharpe / batches
    print(f"Run {run_id}: LR={learning_rate} | Size={d_model} | Sharpe={avg_sharpe:.4f}")
    return model, avg_sharpe, d_model

def main():
    print(f"STARTING TRAINING (Experiment: Differentiable Sharpe)")
    print(f"   (Fee Penalty: {TRAIN_FEE*100}%)")
    print(f"   (Saving to: {SAVE_PATH})")
    
    try:
        data = get_market_data(SYMBOL, period='2y')
    except Exception as e:
        print(f"Data Error: {e}")
        return
        
    best_score = -999
    best_config = {}

    for i in range(ITERATIONS):
        candidate, score, size = train_one_candidate(data, i+1)
        
        # We look for the highest Sharpe Ratio
        if score > best_score:
            print(f"NEW CHAMPION FOUND! (Sharpe: {score:.4f})")
            best_score = score
            best_config = {'d_model': size}
            
            checkpoint = {
                'model_state_dict': candidate.state_dict(),
                'config': {
                    'd_model': size,
                    'input_dim': 2,
                    'strategy': 'differentiable_sharpe'
                },
                'score': best_score
            }
            torch.save(checkpoint, SAVE_PATH)
                
    print("\n" + "="*40)
    print(f"SEARCH COMPLETE.")
    print(f"   - Best Sharpe Ratio: {best_score:.4f}")
    print(f"   - Saved to: {SAVE_PATH}")
    print("="*40)

if __name__ == "__main__":
    main()
