import torch
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys

# Import Architecture
try:
    from model import DifferentiableTrader
except ImportError:
    print("❌ Error: 'model.py' not found. Please ensure it is in the same folder.")
    sys.exit()

# --- CONFIGURATION ---
# We set this to 64 because your "Run 28" winner was Size 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SYMBOL = 'BTC-USD'
D_MODEL = 64             # <--- MATCHES YOUR CHAMPION SIZE
INITIAL_CAPITAL = 10000  # Starting cash ($10,000)
TRANSACTION_FEE = 0.001  # 0.1% per trade (Standard exchange fee)
MODEL_PATH = "disciplined_model.pth"

def run_backtest():
    print(f"💰 STARTING BACKTEST (Mean Return Strategy)")
    print(f"   - Initial Capital: ${INITIAL_CAPITAL}")
    print(f"   - Fee Rate: {TRANSACTION_FEE * 100}%")
    print(f"   - Model Size: {D_MODEL}")
    
    # 1. Load Data (Same 2-year window to verify training)
    print(f"\n📥 Fetching 2y Data for {SYMBOL}...")
    try:
        data = yf.download(SYMBOL, period="2y", interval="1h", progress=False)
        if len(data) == 0: raise ValueError("No data found")
    except Exception as e:
        print(f"❌ Data Error: {e}")
        return

    prices = data['Close'].values.squeeze()
    volumes = data['Volume'].values.squeeze()
    
    # 2. Preprocessing
    log_returns = np.diff(np.log(prices))
    log_volume_change = np.diff(np.log(volumes + 1e-8))
    
    # Create Features
    features = np.column_stack((log_returns, log_volume_change))
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)
    
    # 3. Load the Champion Brain
    model = DifferentiableTrader(input_dim=2, d_model=D_MODEL).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"🧠 Champion Brain Loaded Successfully from {MODEL_PATH}!")
    except FileNotFoundError:
        print(f"❌ Error: {MODEL_PATH} not found in this folder.")
        return
    except RuntimeError as e:
        print(f"❌ Shape Mismatch: {e}")
        print("   -> Did you forget to update D_MODEL? Check train_fleet.py logs.")
        return

    model.eval()
    
    # 4. The Simulation Loop
    balance = INITIAL_CAPITAL
    equity_curve = [balance]
    position = 0 # -1 (Short), 0 (Neutral), 1 (Long)
    trade_count = 0
    
    print("\n🏃 Running Simulation...")
    
    # We need 50 hours of history to make the first prediction
    for t in range(50, len(features_norm) - 1):
        
        # A. Get Context
        window = features_norm[t-50:t]
        tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # B. Ask AI
        with torch.no_grad():
            signal = model(tensor)[0, -1, 0].item()
            
        # C. Decide Action (Threshold 0.0 to catch all "Greedy" moves)
        # Note: Mean Return models are often aggressive, so we use 0.0 or small threshold
        new_position = 0
        # NEW (High Confidence Only)
        if signal > 0.75: new_position = 1
        elif signal < -0.75: new_position = -1
        
        # D. Calculate PnL (Profit/Loss)
        # Price change for the NEXT hour
        actual_return = prices[t+1] / prices[t] - 1
        
        # Apply return to current balance based on position
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
    print(f"🏁 RESULT: ${final_balance:,.2f}")
    print(f"📈 PROFIT: ${profit:,.2f} ({roi:.2f}%)")
    print(f"🔄 TRADES: {trade_count}")
    print(f"📊 BUY & HOLD: {buy_hold_return:.2f}%")
    print("="*40)
    
    if profit < 0:
        print("⚠️  WARNING: Strategy lost money. Likely 'Fee Trap'.")
    elif roi > buy_hold_return:
        print("🏆 SUCCESS: AI beat the market!")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label='AI Strategy (Mean Return)', color='blue')
    
    # Benchmark (Buy and Hold) - Scaled to start at $10k
    benchmark = (prices[50:] / prices[50]) * INITIAL_CAPITAL
    plt.plot(benchmark, label='Buy & Hold (BTC)', color='gray', alpha=0.5, linestyle='--')
    
    plt.title(f"Backtest: {SYMBOL} (Trades: {trade_count})")
    plt.xlabel("Hours")
    plt.ylabel("Account Balance ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_backtest()
    