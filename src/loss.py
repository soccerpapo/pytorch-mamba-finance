import torch
import torch.nn.functional as F

def differentiable_sharpe_loss(positions, returns, transaction_cost=0.0005):
    """
    Computes the Sharpe Ratio loss (Negative Sharpe) in a differentiable way.
    
    Args:
        positions (torch.Tensor): The model's trading decisions (-1 to 1).
        returns (torch.Tensor): The raw price returns of the asset.
        transaction_cost (float): Fee per trade (default 0.05%).
        
    Returns:
        torch.Tensor: Scalar loss value (lower is better).
    """
    # 1. Calculate Strategy Returns
    strategy_returns = positions * returns
    
    # 2. Calculate Transaction Costs
    # We use the change in position (delta) to estimate trade volume
    trades = torch.abs(positions[:, 1:] - positions[:, :-1])
    # Pad the first time step because there is no "previous" position
    trades = F.pad(trades, (0, 0, 1, 0))
    
    costs = trades * transaction_cost
    
    # 3. Net Returns
    net_returns = strategy_returns - costs
    
    # 4. Sharpe Calculation (Mean / StdDev)
    # We add 1e-8 to std to prevent division by zero
    sharpe_ratio = torch.mean(net_returns) / (torch.std(net_returns) + 1e-8)
    
    # Return negative because optimizers want to MINIMIZE loss
    return -sharpe_ratio
