import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# --- 1. PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(root_dir)

try:
    from src.model import DifferentiableTrader
    from src.data import get_market_data
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SYMBOL = 'BTC-USD'
INITIAL_CAPITAL = 10000 
TRANSACTION_FEE = 0.001
# Threshold: We only trade if the model is 25% confident
THRESHOLD = 0.25
MODEL_PATH = os.path.join(current_dir, "champion_model.pth")

def run_backtest():
    print(f"STARTING BACKTEST (Sharpe Strategy)")
    
    # 1. Load Data
    try:
        features = get_market_data(SYMBOL, period="2y", interval="1h")
        import yfinance as yf
        raw_df = yf.download(SYMBOL, period="2y", interval="1h", progress=False)
        prices = raw_df['Close'].values.squeeze()
    except Exception as e:
        print(f"Data Error: {e}")
        return

    # 2. Load Champion
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        d_model = checkpoint['config']['d_model']
        state_dict = checkpoint['model_state_dict']
    else:
        d_model = 32 # Fallback
        state_dict = checkpoint

    model = DifferentiableTrader(input_dim=2, d_model=d_model).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 3. Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)

    # 4. Simulation
    balance = INITIAL_CAPITAL
    equity_curve = [balance]
    position = 0 
    trade_count = 0
    
    print("\nRunning Simulation...")
    
    for t in range(50, len(features_norm) - 1):
        window = features_norm[t-50:t]
        tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            signal = model(tensor)[0, -1, 0].item()
            
        # LOGIC: Only buy if signal > 0.25, only sell if < -0.25
        # This, combined with Sharpe training, drastically reduces noise.
        new_position = 0
        if signal > THRESHOLD: new_position = 1
        elif signal < -THRESHOLD: new_position = -1
        
        # PnL
        actual_return = prices[t+1] / prices[t] - 1
        if position != 0:
            balance += balance * position * actual_return
            
        # Fees
        if new_position != position:
            balance -= balance * TRANSACTION_FEE
            trade_count += 1
            
        position = new_position
        equity_curve.append(balance)
        
    # 5. Results
    profit = equity_curve[-1] - INITIAL_CAPITAL
    roi = (profit / INITIAL_CAPITAL) * 100
    
    print("\n" + "="*40)
    print(f"RESULT: ${equity_curve[-1]:,.2f}")
    print(f"PROFIT: ${profit:,.2f} ({roi:.2f}%)")
    print(f"TRADES: {trade_count}")
    print("="*40)
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label='Sharpe Strategy', color='green')
    plt.title(f"Sharpe Optimization: Disciplined Trading\nTrades: {trade_count}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(current_dir, "result.png"))
    print(f"Saved result.png")

if __name__ == "__main__":
    run_backtest()
