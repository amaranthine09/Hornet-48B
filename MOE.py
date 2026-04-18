import math 
import torch
import torch.nn as nn
import torch.nn.functional as F 
from typing import Optional, Tuple,  Dict
from dataclasses import dataclass
from SwishGLU import SwiGlU


@dataclass
class MOEconfg:
    d_model: int = 4096
    d_expert: int = 1024

    shared_expert: int  = 2
    routed_expert: int = 64
    top_k_expert: int = 6

    aux_loss: bool = True
    aux_loss_gemma: float = 0.001
    use_device_limit: bool = True
    max_deice_limit: int = 3

    no_of_device: int = 1
    experts_per_device: Optional[int] = None

    dropout: float = 0.0
    
    def __post__inti__(self):
        if self.experts_per_device is None:
            self.experts_per_device = max(1, self.num_routed_experts // self.num_devices)

class RMSNorm(nn.module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps 
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor)->torch.Tensor:
        x = x*torch.rsqrt(x.pow(2).mean(-1, keepdim = True)+self.eps)
        return x*self.weight

class Expert(nn.Module):
    def __init__(self, config: MOEconfg):
        super().__init__()
        self.FNN = SwiGlU(config.d_model, config.d_expert)
        




    
    