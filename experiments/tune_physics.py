import sys
import os
import numpy as np
import yfinance as yf
import pandas as pd

def run_physics_grid():
    print("🧪 STARTING PHYSICS GRID SEARCH...", flush=True)

    # --- 1. GET DATA (With Fallback) ---
    prices = None
    try:
        print("   📥 Downloading BTC-USD...", flush=True)
        # Using 3 months of data is faster and sufficient for calibration
        df = yf.download("BTC-USD", period="3mo", interval="1h", progress=False, auto_adjust=True)
        if len(df) > 0:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            prices = df['Close'].values
            print(f"   ✅ Data Loaded: {len(prices)} points", flush=True)
    except Exception as e:
        print(f"   ⚠️ Download failed: {e}", flush=True)

    if prices is None or len(prices) == 0:
        print("   ⚠️ USING SYNTHETIC DATA (Fallback)", flush=True)
        t = np.linspace(0, 100, 2000)
        prices = 100 + 10 * np.sin(t) + np.random.normal(0, 0.5, 2000)

    # --- 2. DEFINE KALMAN FILTER ---
    class KalmanFilter1D:
        def __init__(self, dt=1.0, std_acc=0.1, std_meas=0.1):
            self.dt = dt
            self.std_acc = std_acc
            self.std_meas = std_meas
            self.A = np.matrix([[1, self.dt], [0, 1]])
            self.B = np.matrix([[0], [0]])
            self.H = np.matrix([[1, 0]])
            self.Q = np.matrix([[(self.dt**4)/4, (self.dt**3)/2],
                                [(self.dt**3)/2, self.dt**2]]) * self.std_acc**2
            self.R = np.matrix([[self.std_meas**2]])
            self.P = np.eye(self.A.shape[1])

        def predict(self, x):
            x = np.dot(self.A, x)
            self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
            return x

        def update(self, x, z):
            y = z - np.dot(self.H, x)
            S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
            K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
            x = x + np.dot(K, y)
            I = np.eye(self.H.shape[1])
            self.P = np.dot((I - np.dot(K, self.H)), self.P)
            return x, y, S

    # --- 3. GRID SEARCH ---
    std_meas_options = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0]
    std_acc_options = [0.01, 0.001, 0.0001, 0.00001, 0.000001]

    print(f"\n{'MEAS_NOISE':<12} | {'ACC_NOISE':<12} | {'TRADES':<8} | {'SHARPE':<8} | {'ROI %':<8}", flush=True)
    print("-" * 65, flush=True)

    best_sharpe = -999
    best_params = None

    for meas in std_meas_options:
        for acc in std_acc_options:
            kf = KalmanFilter1D(dt=1.0, std_acc=acc, std_meas=meas)
            x = np.matrix([[prices[0]], [0]])

            velocities = []
            for z in prices:
                x = kf.predict(x)
                x, _, _ = kf.update(x, z)
                velocities.append(x[1,0])

            # --- FIXED LOGIC ---
            velocities = np.array(velocities)
            pos = np.sign(velocities)

            # Returns (Price change from t to t+1)
            # Array Length: N-1
            log_ret = np.diff(np.log(prices))

            # Strategy Return (Position at t * Return at t+1)
            # Array Length: N-1
            strategy_ret = pos[:-1] * log_ret

            # Trades (Did position change between t and t+1?)
            # Array Length: N-1
            trades = np.abs(np.diff(pos)) / 2

            # Costs
            costs = trades * 0.001 # 0.1% cost

            # Net Returns (Subtract directly, sizes match perfectly now)
            net_ret = strategy_ret - costs

            # Stats
            total_trades = np.sum(trades)

            if np.std(net_ret) == 0: sharpe = 0
            else: sharpe = np.mean(net_ret) / (np.std(net_ret) + 1e-8) * np.sqrt(24*365)

            roi = np.sum(net_ret) * 100

            # Filter: Penalize churning (>50% of bars) or dead (<5 trades) strategies
            if total_trades > (len(prices)/2) or total_trades < 5: sharpe = -99.0

            print(f"{meas:<12.4f} | {acc:<12.6f} | {int(total_trades):<8} | {sharpe:<8.2f} | {roi:<8.2f}%", flush=True)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (meas, acc)

    print("-" * 65, flush=True)
    if best_params:
        print(f"🏆 WINNER: std_meas={best_params[0]}, std_acc={best_params[1]}", flush=True)
        print(f"   (Sharpe: {best_sharpe:.2f})")
    else:
        print("❌ No settings passed the filter. Try widening the search.", flush=True)

if __name__ == "__main__":
    run_physics_grid()
