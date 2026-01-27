import torch
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler

def get_market_data(symbol='BTC-USD', period='2y', interval='1h'):
    """
    Fetches raw data from YFinance and computes features (Log Returns, Log Vol).
    """
    print(f"📥 Downloading {symbol} ({period})...")
    data = yf.download(symbol, period=period, interval=interval, progress=False)
    
    if len(data) == 0:
        raise ValueError(f"No data found for {symbol}")

    prices = data['Close'].values.squeeze()
    volumes = data['Volume'].values.squeeze()
    
    # Feature Engineering
    # Log returns are preferred in finance over raw % change
    log_returns = np.diff(np.log(prices))
    log_volume_change = np.diff(np.log(volumes + 1e-8))
    
    # Stack features
    features = np.column_stack((log_returns, log_volume_change))
    return features

def generate_synthetic_data(n_samples=5000):
    """
    Generates mock data for unit testing model stability.
    """
    print(f"🧪 Generating {n_samples} hours of synthetic data...")
    returns = np.random.normal(0, 0.01, n_samples)
    volume_noise = np.random.normal(0, 0.05, n_samples)
    volume_changes = np.abs(returns) * 5 + volume_noise 
    
    features = np.column_stack((returns, volume_changes))
    return features

def create_dataloaders(features, seq_len=50, batch_size=64, device='cpu'):
    """
    Converts a long time-series array into sliding window batches for training.
    """
    # 1. Normalize
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)
    
    # 2. Create Windows
    windows = []
    for i in range(len(features_norm) - seq_len):
        windows.append(features_norm[i : i + seq_len])
    
    # 3. Batching
    windows = np.array(windows)
    x_batches = []
    
    # Convert to Tensor Batches
    for i in range(0, len(windows), batch_size):
        batch_data = windows[i : i + batch_size]
        if len(batch_data) > 0:
            x_tensor = torch.tensor(batch_data, dtype=torch.float32).to(device)
            x_batches.append(x_tensor)
            
    return x_batches, scaler
