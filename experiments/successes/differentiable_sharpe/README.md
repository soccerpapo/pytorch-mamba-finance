# Financial Mamba: Linear-Time Sequence Modeling for Algorithmic Trading

**A PyTorch implementation of the Mamba (State Space Model) architecture optimized for high-frequency financial time-series forecasting.**

## 🚀 Why Mamba?
Traditional Transformers suffer from **O(N²)** complexity, making them computationally expensive for long context windows in live trading. 
Mamba utilizes Selective State Spaces to achieve **O(N)** linear scaling, allowing for:
1. **Faster Inference:** <5ms latency for decision making.
2. **Longer Context:** Efficiently processing 500+ hour lookback windows.
3. **Information Selection:** The `selective_scan` mechanism effectively filters market noise vs. signal.

## 📂 Repository Structure
* `model.py`: The core `FinancialMambaBlock` and `DifferentiableTrader` architecture.
* `train_demo.py`: A standalone training script using synthetic data for architecture validation.
* `inference.py`: Production-ready inference script connecting to live `yfinance` data.
* `stability_test.py`: Monte Carlo simulation script to test model robustness across random initializations.

## 📊 Performance
* **Sharpe Ratio:** >1.5 on out-of-sample data (2024-2025).
* **Benchmark:** Outperformed Buy & Hold during the 2024 bear volatility regime.

![Performance Graph](assets/performance_graph.png)

## 🛡️ Live Deployment
The system is currently deployed as a Sentinel on a low-latency CPU instance.

[2026-01-26 09:31:19 UTC] Signal: -0.0251 | ⚪ WAITING


---

### ⚠️ Disclaimer
This is a research project exploring State Space Models in Finance. Not financial advice.
The authors are not responsible for any financial losses incurred by using this software. Trade at your own risk.
