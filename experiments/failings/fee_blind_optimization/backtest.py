import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# --- 1. PATH SETUP ---
# Dynamically find the 'src' folder 3 levels up
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(root_dir)

try:
    from src.model import DifferentiableTrader
    from src.data import get_market_data
except ImportError as e:
    print(f"Import Error: {e}")
    print("   -> Check that 'src' is in the root and contains __init__.py")
    sys.exit(1)

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SYMBOL = 'BTC-USD'
INITIAL_CAPITAL = 10000 
TRANSACTION_FEE = 0.001  # 0.1% per trade
MODEL_PATH = os.path.join(current_dir, "champion_model.pth")

def run_backtest():
    print(f"STARTING BACKTEST (Fee-Blind Strategy)")
    print(f"   - Initial Capital: ${INITIAL_CAPITAL}")
    print(f"   - Fee Rate: {TRANSACTION_FEE * 100}%")
    
    # 1. Load Data
    # We use the src tool for features, but we also need raw prices for PnL
    try:
        features = get_market_data(SYMBOL, period="2y", interval="1h")
        
        # We re-fetch raw data just to get the 'Close' prices for the simulation
        import yfinance as yf
        raw_df = yf.download(SYMBOL, period="2y", interval="1h", progress=False)
        prices = raw_df['Close'].values.squeeze()
        
    except Exception as e:
        print(f"Data Error: {e}")
        return

    # 2. Load the Champion Brain (Smart Loader)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        print(f"   -> Please run 'python train_fleet.py' in this folder first.")
        return

    print(f"Loading {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Handle "Smart Save" vs "Legacy"
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        d_model = checkpoint['config']['d_model']
        state_dict = checkpoint['model_state_dict']
        print(f"Smart Checkpoint Detected (Size: {d_model})")
    else:
        # Fallback for old files
        d_model = 64 
        state_dict = checkpoint
        print(f"Legacy Checkpoint (Assuming Size: 64)")

    model = DifferentiableTrader(input_dim=2, d_model=d_model).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 3. Normalize Features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)

    # 4. The Simulation Loop
    balance = INITIAL_CAPITAL
    equity_curve = [balance]
    position = 0 # -1 (Short), 0 (Neutral), 1 (Long)
    trade_count = 0
    
    print("\nRunning Simulation...")
    
    # We need 50 hours of history to make the first prediction
    for t in range(50, len(features_norm) - 1):
        
        # A. Get Context
        window = features_norm[t-50:t]
        tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # B. Ask AI
        with torch.no_grad():
            signal = model(tensor)[0, -1, 0].item()
            
        # C. Decide Action 
        # GREEDY LOGIC: We set threshold to 0.0 to demonstrate the "Fee Trap"
        # The model thinks it can win every small wiggle.
        new_position = 0
        if signal > 0.0: new_position = 1
        elif signal < 0.0: new_position = -1
        
        # D. Calculate PnL
        # Price change for the NEXT hour
        actual_return = prices[t+1] / prices[t] - 1
        
        if position != 0:
            pnl = balance * position * actual_return
            balance += pnl
            
        # E. Apply Fees
        if new_position != position:
            cost = balance * TRANSACTION_FEE
            balance -= cost
            trade_count += 1
            
        position = new_position
        equity_curve.append(balance)
        
    # 5. Results
    final_balance = equity_curve[-1]
    profit = final_balance - INITIAL_CAPITAL
    roi = (profit / INITIAL_CAPITAL) * 100
    buy_hold_return = (prices[-1] / prices[50] - 1) * 100
    
    print("\n" + "="*40)
    print(f"RESULT: ${final_balance:,.2f}")
    print(f"PROFIT: ${profit:,.2f} ({roi:.2f}%)")
    print(f"TRADES: {trade_count}")
    print(f"BUY & HOLD: {buy_hold_return:.2f}%")
    print("="*40)
    
    if profit < 0:
        print("CONFIRMED: The 'Fee Trap' is active.")
        print("    The model over-traded and lost money on fees.")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label='Fee-Blind Strategy', color='red')
    
    # Benchmark
    benchmark = (prices[50:] / prices[50]) * INITIAL_CAPITAL
    plt.plot(benchmark, label='Buy & Hold (BTC)', color='gray', alpha=0.5, linestyle='--')
    
    plt.title(f"The Fee Trap: High Frequency vs. Market Friction\nTrades: {trade_count}")
    plt.xlabel("Hours")
    plt.ylabel("Account Balance ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot instead of just showing it
    plt.savefig("result.png")
    print(f"Chart saved to: {os.path.join(current_dir, 'result.png')}")
    # plt.show() # Uncomment if you have a display

if __name__ == "__main__":
    run_backtest()
