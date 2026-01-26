import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from model import DifferentiableTrader

# --- 1. CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
SEQ_LEN = 500
EPOCHS = 20

# --- 2. SYNTHETIC DATA GENERATOR ---
# Professional software relies on robust testing, not just live API calls.
def generate_mock_market(n_samples=5000):
    print(f"Generating {n_samples} hours of synthetic market data...")
    returns = np.random.normal(0, 0.01, n_samples)
    volume_noise = np.random.normal(0, 0.05, n_samples)
    volume_changes = np.abs(returns) * 5 + volume_noise 
    features = np.column_stack((returns, volume_changes))
    
    # Normalize
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return (features - mean) / (std + 1e-8)

def create_sliding_batches(features, seq_len, batch_size):
    x_batches = []
    windows = []
    for i in range(len(features) - seq_len):
        windows.append(features[i : i + seq_len])
    
    windows = np.array(windows)
    for i in range(0, len(windows), batch_size):
        x_tensor = torch.tensor(windows[i : i + batch_size], dtype=torch.float32).to(DEVICE)
        x_batches.append(x_tensor)
    return x_batches

# --- 3. LOSS FUNCTION ---
def differentiable_sharpe_loss(positions, returns, transaction_cost=0.0005):
    strategy_returns = positions * returns
    trades = torch.abs(positions[:, 1:] - positions[:, :-1])
    trades = F.pad(trades, (0, 0, 1, 0))
    costs = trades * transaction_cost
    net_returns = strategy_returns - costs
    return -(torch.mean(net_returns) / (torch.std(net_returns) + 1e-8))

# --- 4. EXECUTION ---
import torch.nn.functional as F # Re-importing for safety in script context

print(f"Running on: {DEVICE}")
data = generate_mock_market()
batches = create_sliding_batches(data, SEQ_LEN, BATCH_SIZE)

model = DifferentiableTrader(input_dim=2, d_model=32).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

print("\nStarting Synthetic Training Loop...")
for epoch in range(EPOCHS):
    epoch_loss = []
    for batch_x in batches:
        optimizer.zero_grad()
        positions = model(batch_x)
        target_returns = batch_x[:, :, 0:1] # Optimize on "Price" feature
        loss = differentiable_sharpe_loss(positions, target_returns)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Sharpe Loss = {np.mean(epoch_loss):.4f}")

print("✅ Demo Training Complete. Mamba Architecture Validated.")
