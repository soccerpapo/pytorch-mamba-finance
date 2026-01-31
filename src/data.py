import torch
import numpy as np
import yfinance as yf
import pandas as pd

#relative strength index not normalized: detects overbought/oversold
def calculate_rsi_notnorm(series, period = 14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window = period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window = period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

#moving average convergence divergence and signal line: detects momentum shifts
def calculate_macd_signalline(series, fast = 12, slow = 26, signal = 9):
    exp1 = series.ewm(span = fast, adjust = False).mean()
    exp2 = series.ewm(span = slow, adjust = False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span = signal, adjust = False).mean()
    return macd, signal_line

#bollinger bands position: detects volatility breakouts
def calculate_bollinger_position(series, window = 20):
    sma = series.rolling(window = window).mean()
    std = series.rolling(window = window).std()
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    #normalization: scaling data (price position) into [0,1]. contrary to standardization, where data is scaled into (-3,3)
    #>1 means breakout up, <0 means breakout down
    bb_pos = (series - lower) / (upper - lower)
    return bb_pos

#design matrix made out of Log returns (better than raw price), Relative strength index normalized (momentum), Moving average convergence divergence difference (trend), Bollinger bands position (volatility breakout), Rolling volatility (risk context), and Volume change (activity)
def get_market_data(symbol, period = '2y', interval = '1h'):
    print(f"downloading {symbol}...")

    try:
        df = yf.download(symbol, period = period, interval = interval, progress = False, auto_adjust = True)
    except Exception as e:
        raise ValueError(f"YFinance download error: {e}")
    
    #if it doesnt have rows
    if len(df) == 0:
        raise ValueError("no data downloaded. check symbol or internet connection")
    
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    close = df['Close']

    df["log_returns"] = np.log(close / close.shift(1))

    #Relative strength index normalized here
    df["rsi_norm"] = calculate_rsi_notnorm(close) / 100.0

    #histogram value
    macd, signal_line = calculate_macd_signalline(close)
    df["macd_diff"] = macd - signal_line

    df["bollinger_position"] = calculate_bollinger_position(close)

    #volatility: how risky the market is right now
    df["volatility_20"] = df['log_returns'].rolling(window = 20).std()

    vol = df['Volume'].replace(0, np.nan).ffill()
    
    #volume change: activity
    df["volume_change"] = np.log(vol / vol.shift(1)).fillna(0)

    df["volume_change"] = df['volume_change'].replace([np.inf, -np.inf], 0)

    df.dropna(inplace = True)

    design_columns = ['log_returns', 'rsi_norm', 'macd_diff', 'bollinger_position', 'volatility_20', 'volume_change']

    print(f"data engineered. matrix shape: {df[design_columns].shape}")
    print(f"columns: {design_columns}")

    return df[design_columns].values
