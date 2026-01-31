import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
import sys
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(root_dir)

from src.model import DifferentiableTrader
from src.data import get_market_data
from src.loss import differentiable_sharpe_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS_PER_RUN = 20
ITERATIONS = 50
SYMBOL = 'BTC-USD'
SAVE_PATH = os.path.join(current_dir, "champion.pth")
TRAIN_FEE = 0.0015
# UPDATED DIMENSION
INPUT_DIM = 4

def make_batches(data, b_size):
    if len(data) == 0: return []
    data_tensor = torch.tensor(data, dtype=torch.float32)
    x_batches = [data_tensor[i:i+50] for i in range(len(data) - 50)]
    loader = []
    for i in range(0, len(x_batches), b_size):
        batch = x_batches[i:i+b_size]
        if len(batch) > 0: loader.append(torch.stack(batch).to(DEVICE))
    return loader

def train_one_candidate(features, run_id):
    start_time = time.time()
    learning_rate = random.choice([0.001, 0.0005])
    batch_size = random.choice([64, 128])
    d_model = random.choice([32, 64])
    split_idx = int(len(features) * 0.8)
    train_data = features[:split_idx]
    val_data = features[split_idx:]
    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_data)
    val_norm = scaler.transform(val_data)
    train_loader = make_batches(train_norm, batch_size)
    val_loader = make_batches(val_norm, batch_size)

    # FIX: Use INPUT_DIM here
    model = DifferentiableTrader(input_dim=INPUT_DIM, d_model=d_model).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    print(f"\n🚀 Run {run_id}: Started (LR={learning_rate} | D={d_model})")

    for epoch in range(EPOCHS_PER_RUN):
        model.train()
        epoch_loss = 0
        count = 0
        for batch in train_loader:
            optimizer.zero_grad()
            positions = model(batch)
            returns = batch[:, :, 0:1]
            loss = differentiable_sharpe_loss(positions, returns, transaction_cost=TRAIN_FEE)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            count += 1
        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss/count if count > 0 else 0
            print(f"   Epoch {epoch+1}/{EPOCHS_PER_RUN} | Loss: {avg_loss:.6f}")

    model.eval()
    val_sharpe = 0
    batches = 0
    with torch.no_grad():
        for batch in val_loader:
            positions = model(batch)
            returns = batch[:, :, 0:1]
            s_loss = differentiable_sharpe_loss(positions, returns, transaction_cost=0.001)
            val_sharpe += (-s_loss.item())
            batches += 1
    avg_sharpe = val_sharpe / batches if batches > 0 else -999
    elapsed = time.time() - start_time
    print(f"🏁 Run {run_id} Complete | Sharpe: {avg_sharpe:.4f} | Time: {elapsed:.1f}s")
    return model, avg_sharpe, d_model

def main():
    print(f"🔥 STARTING TRAINING (Feature Lobotomy Edition)")
    try:
        data = get_market_data(SYMBOL, period='2y')
    except Exception as e:
        print(f"❌ Data Error: {e}")
        return
    best_score = -999
    for i in range(ITERATIONS):
        candidate, score, size = train_one_candidate(data, i+1)
        if score > best_score:
            print(f"🌟 NEW CHAMPION FOUND! (Sharpe: {score:.4f})")
            best_score = score
            checkpoint = {'model_state_dict': candidate.state_dict(), 'config': {'d_model': size, 'input_dim': INPUT_DIM}, 'score': best_score}
            torch.save(checkpoint, SAVE_PATH)

if __name__ == "__main__":
    main()
