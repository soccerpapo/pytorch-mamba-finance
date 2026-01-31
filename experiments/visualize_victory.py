import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(root_dir)

from src.data import get_market_data

SYMBOL = 'BTC-USD'

def visualize():
    print(f"🎨 GENERATING VICTORY CHART...")
    
    # 1. Get Data & Price
    data = get_market_data(SYMBOL, period="2y")
    velocity = data[:, 1] # k_vel_norm
    
    # We need raw prices for the chart
    df = yf.download(SYMBOL, period="2y", interval="1h", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    prices = df['Close'].values[-len(velocity):] # Align lengths
    dates = df.index[-len(velocity):]
    
    # 2. Reconstruct Signals
    THRESHOLD = 0.05
    position = np.zeros(len(velocity))
    position[velocity > THRESHOLD] = 1
    position[velocity < -THRESHOLD] = -1
    
    # Forward Fill
    for i in range(1, len(position)):
        if position[i] == 0:
            position[i] = position[i-1]
            
    # 3. Identify Buy/Sell Events
    buy_indices = []
    sell_indices = []
    
    # trades = where position changes
    # if pos goes -1 -> 1 (Buy)
    # if pos goes 1 -> -1 (Sell)
    
    for i in range(1, len(position)):
        if position[i] == 1 and position[i-1] == -1:
            buy_indices.append(i)
        elif position[i] == -1 and position[i-1] == 1:
            sell_indices.append(i)

    # 4. Plot
    plt.figure(figsize=(14, 7))
    plt.plot(dates, prices, label='Bitcoin Price', color='black', alpha=0.6, linewidth=1)
    
    # Plot Buys
    plt.scatter(dates[buy_indices], prices[buy_indices], marker='^', color='lime', s=100, label='Physics Buy', zorder=5)
    
    # Plot Sells
    plt.scatter(dates[sell_indices], prices[sell_indices], marker='v', color='red', s=100, label='Physics Sell', zorder=5)
    
    plt.title(f'Phase 2 Victory: Kalman Filter Signals (+91% ROI)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig('phase2_victory.png')
    print("✅ Chart saved to 'phase2_victory.png'")
    plt.show()

if __name__ == "__main__":
    visualize()
