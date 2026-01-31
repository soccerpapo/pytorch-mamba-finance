import torch
import torch.nn.functional as F

# --- OPTIMIZATION: JIT COMPILATION ---
# This decorator compiles the math directly to C++/CUDA.
# It fuses the element-wise operations (multiply, sub, pad) into a single kernel
# which is significantly faster on Colab GPUs.
@torch.jit.script
def differentiable_sharpe_loss(positions: torch.Tensor, returns: torch.Tensor, transaction_cost: float = 0.0005) -> torch.Tensor:
    """
    Computes the Negative Sharpe Ratio loss efficiently across batches.
    Optimized for JIT/GPU execution.
    """
    # 1. Strategy Returns
    # positions: [Batch, Seq, 1]
    # returns:   [Batch, Seq, 1]
    strategy_returns = positions * returns
    
    # 2. Transaction Costs
    # We calculate the diff to find where trades occurred
    trades = torch.abs(positions[:, 1:] - positions[:, :-1])
    
    # Pad the time dimension (dim 1) to match original sequence length
    # Pad args are (last_dim_left, last_dim_right, 2nd_last_left, 2nd_last_right)
    # We pad the beginning of the sequence with 0 (no cost for first entry)
    trades = F.pad(trades, (0, 0, 1, 0)) 
    
    costs = trades * transaction_cost
    
    # 3. Net Returns
    net_returns = strategy_returns - costs
    
    # 4. Sharpe Calculation (Per Batch)
    # dim=1 calculates Mean/Std across TIME (the sequence), not across the batch
    expected_return = torch.mean(net_returns, dim=1)
    
    # Add epsilon to prevent division by zero
    risk = torch.std(net_returns, dim=1) + 1e-8
    
    # Calculate Sharpe for each sample in the batch
    batch_sharpe = expected_return / risk
    
    # 5. Final Loss
    # Average the Sharpe Ratios across the batch, then negate
    return -torch.mean(batch_sharpe)
