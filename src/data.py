import torch
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler

def get_market_data(symbol='BTC-USD', period='2y', interval='1h'):
    """
    Fetches raw data from YFinance and computes features (Log Returns, Log Vol).
    """
    print(f"📥 Downloading {symbol} ({period})...")
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        if len(data) == 0:
            raise ValueError(f"No data found for {symbol}")
    except Exception as e:
        raise ValueError(f"YFinance Download Error: {e}")

    # .squeeze() handles cases where yfinance returns extra dimensions
    prices = data['Close'].values.squeeze()
    volumes = data['Volume'].values.squeeze()
    
    # Feature Engineering
    # Log returns are preferred in finance over raw % change because they are additive
    log_returns = np.diff(np.log(prices))
    
    # Add 1e-8 to volume to prevent log(0) errors if volume is zero
    log_volume_change = np.diff(np.log(volumes + 1e-8))
    
    # Stack features: [Rows, 2]
    features = np.column_stack((log_returns, log_volume_change))
    return features

def generate_synthetic_data(n_samples=5000):
    """
    Generates mock data for unit testing model stability.
    """
    print(f"🧪 Generating {n_samples} hours of synthetic data...")
    returns = np.random.normal(0, 0.01, n_samples)
    volume_noise = np.random.normal(0, 0.05, n_samples)
    
    # Correlate volume slightly with volatility (common in real markets)
    volume_changes = np.abs(returns) * 5 + volume_noise 
    
    features = np.column_stack((returns, volume_changes))
    return features

def create_dataloaders(features, seq_len=50, batch_size=64, device='cpu'):
    """
    Converts a long time-series array into sliding window batches for training.
    """
    # 1. Normalize
    # We fit the scaler on the entire history provided here.
    # In strict backtesting, you should fit only on 'train' split, 
    # but for this architecture, we usually pass the pre-split data.
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)
    
    # 2. Create Windows
    windows = []
    # Create valid sequences of length seq_len
    for i in range(len(features_norm) - seq_len):
        windows.append(features_norm[i : i + seq_len])
    
    # 3. Batching
    # Note: np.array(windows) can be memory intensive for massive datasets.
    # For 2 years of hourly data (~17k rows), it is perfectly fine (~10MB).
    windows = np.array(windows)
    x_batches = []
    
    # Convert to Tensor Batches and move to DEVICE
    # We move to device here for speed, assuming GPU VRAM fits the dataset.
    for i in range(0, len(windows), batch_size):
        batch_data = windows[i : i + batch_size]
        if len(batch_data) > 0:
            x_tensor = torch.tensor(batch_data, dtype=torch.float32).to(device)
            x_batches.append(x_tensor)
            
    return x_batches, scaler
