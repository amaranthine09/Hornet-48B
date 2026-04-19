import math
import torch 
import torch.nn as nn
import torch.nn.functional as F 
class SwiGLU(nn.Module):
    def __init__(self, dim: int, inmdt_dim : int , dropout: float = 0.0):
        super().__init__()
        self.W_gate = nn.Linear(dim, inmdt_dim, bias = False)
        self.W_down = nn.Linear(dim, inmdt_dim, bias = False)
        self.W_up = nn.Linear(inmdt_dim,  dim, bias = False)
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        nn.init.normal_(self.W_gate.weight, std = 0.006)
        nn.init.normal_(self.W_down.weight, std = 0.006)
        nn.init.normal_(self.W_up.weight, std = 0.006)

    def forward(self, x: torch.Tensor)->torch.Tensor:
        gate = F.silu(self.W_gate(x))
        up = self.W_up(x)
        hidden = gate*up 
        hidden = self.dropout(hidden)
        return self.W_down(hidden)
