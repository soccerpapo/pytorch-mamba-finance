import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Setup path to import src
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(root_dir)

from src.data import get_market_data

# CONFIG
SYMBOL = 'BTC-USD'
INITIAL_CAPITAL = 10000
TRANSACTION_FEE = 0.001 # 0.1% per trade

def run_physics_test():
    print(f"🧪 VERIFYING PURE PHYSICS (No AI)...")

    # 1. Get the Tuned Data
    # This uses your src/data.py with winning params (std_meas=0.2, std_acc=1e-05)
    try:
        data = get_market_data(SYMBOL, period="2y")
        # Column 0 = log_ret
        # Column 1 = k_vel_norm (The Signal)
        log_ret = data[:, 0]
        velocity = data[:, 1]
    except Exception as e:
        print(f"❌ Data Error: {e}")
        return

    # 2. The Strategy (Pure Math)
    # If Velocity is positive, we go Long. If negative, we Short.
    # We use a tiny threshold (0.05) to avoid noise around zero.
    THRESHOLD = 0.05

    position = np.zeros(len(velocity))
    position[velocity > THRESHOLD] = 1
    position[velocity < -THRESHOLD] = -1

    # Forward Fill (Hold position if signal is weak/zero)
    for i in range(1, len(position)):
        if position[i] == 0:
            position[i] = position[i-1]

    # 3. Calculate Performance
    # Align signals: Position at t acts on Return at t+1
    strategy_ret = position[:-1] * log_ret[1:]

    # Costs: We pay when position changes
    trades = np.abs(np.diff(position)) / 2
    costs = trades * TRANSACTION_FEE

    # Net Returns
    net_ret = strategy_ret - costs

    # Equity Curve
    equity = INITIAL_CAPITAL * np.exp(np.cumsum(net_ret))
    bnh = INITIAL_CAPITAL * np.exp(np.cumsum(log_ret[1:]))

    # Stats
    total_trades = np.sum(trades)
    final_balance = equity[-1]
    profit_pct = (final_balance - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    bnh_pct = (bnh[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    print("\n📊 PHYSICS REPORT:")
    print("-" * 40)
    print(f"Initial Capital:   ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Balance:     ${final_balance:,.2f}")
    print(f"Net Profit:        {profit_pct:+.2f}%")
    print(f"Total Trades:      {int(total_trades)}")
    print(f"Buy & Hold:        {bnh_pct:+.2f}%")
    print("-" * 40)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(equity, label='Kalman Velocity (Physics)', color='lime')
    plt.plot(bnh, label='Buy & Hold', color='gray', alpha=0.5)
    plt.title(f'Pure Physics Performance (Trades: {int(total_trades)})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('physics_result.png')
    plt.show()

if __name__ == "__main__":
    run_physics_test()
