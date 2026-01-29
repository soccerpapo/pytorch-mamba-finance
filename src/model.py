import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FinancialMambaBlock(nn.Module):
    """
    A specific Mamba implementation optimized for 1D financial time-series.
    Uses selective state spaces to capture long-range dependencies with linear O(N) complexity.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(self.d_model / 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.activation = nn.SiLU()

        # HiPPO Matrix Initialization for long-term memory
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u):
        xz = self.in_proj(u)
        x, z = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :u.shape[1]]
        x = self.activation(x)
        x = x.transpose(1, 2)
        y = self.selective_scan(x)
        return self.out_proj(y * F.silu(z))

    def selective_scan(self, x):
        batch, seq_len, d_inner = x.shape
        d_state = self.d_state
        A = -torch.exp(self.A_log.float())
        x_dbl = self.x_proj(x)
        delta_raw, B, C = torch.split(x_dbl, [self.dt_rank, d_state, d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta_raw))
        deltaA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        deltaB_u = torch.einsum('bld,bln,bld->bldn', delta, B, x)
        h = torch.zeros(batch, d_inner, d_state, device=x.device)
        ys = []
        for t in range(seq_len):
            h = deltaA[:, t] * h + deltaB_u[:, t]
            y_t = torch.einsum('bdn,bln->bd', h, C[:, t].unsqueeze(1))
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        return y + x * self.D

class DifferentiableTrader(nn.Module):
    """
    A wrapper that connects the Mamba 'Physics Engine' to a trading decision head.
    Outputs: Scaled Tanh signal (-1.0 to 1.0) representing Short/Long conviction.
    """
    def __init__(self, input_dim, d_model):
        super().__init__()
        # Note: We pass input_dim to physics to maintain compatibility with your 
        # existing trained weights (champion_model.pth).
        self.physics = FinancialMambaBlock(d_model=input_dim, d_state=16)
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        context = self.physics(x)
        return self.head(context)
    