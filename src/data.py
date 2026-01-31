import torch
import numpy as np
import yfinance as yf
import pandas as pd

class KalmanFilter1D:
    def __init__(self, dt=1.0, u=0.0, std_acc=0.1, std_meas=0.1):
        self.dt = dt
        self.u = u
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
        x = np.dot(self.A, x) + np.dot(self.B, self.u)
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

def get_market_data(symbol, period='2y', interval='1h'):
    print(f"📥 [Berkeley] Downloading {symbol}...")
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    except Exception as e:
        raise ValueError(f"YFinance download error: {e}")

    if len(df) == 0: raise ValueError("No data downloaded.")
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

    close = df['Close'].values
    log_ret = np.log(df['Close'] / df['Close'].shift(1)).fillna(0).values

    # 🏆 WINNER: std_meas=0.2, std_acc=1e-05
    # This sedates the filter so it only tracks real trends
    kf = KalmanFilter1D(dt=1.0, std_acc=0.00001, std_meas=0.2)

    x = np.matrix([[close[0]], [0]])

    kalman_velocity = []
    kalman_error = []

    for z in close:
        x = kf.predict(x)
        x, innovation, error_cov = kf.update(x, z)

        # FEATURE LOBOTOMY: We throw away 'innovation' (noise)
        kalman_velocity.append(x[1,0])
        kalman_error.append(error_cov[0,0])

    df['log_ret'] = log_ret
    df['k_vel'] = kalman_velocity
    df['k_err'] = kalman_error

    # Normalize
    df['k_vel_norm'] = (df['k_vel'] - df['k_vel'].rolling(50).mean()) / (df['k_vel'].rolling(50).std() + 1e-8)
    df['vol_20'] = df['log_ret'].rolling(20).std()

    df.dropna(inplace=True)

    # Final Feature Set: 4 Dimensions (Noise Removed)
    design_columns = ['log_ret', 'k_vel_norm', 'k_err', 'vol_20']

    print(f"✅ Berkeley Data Engineered. Matrix Shape: {df[design_columns].shape}")
    print(f"   Features: {design_columns} (Noise Removed)")

    return df[design_columns].values
