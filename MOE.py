import math 
import torch
import torch.nn as nn
import torch.nn.functional as F 
from typing import Optional, Tuple,  Dict
from dataclasses import dataclass
from SwishGLU import SwiGlU


@dataclass
class MOEconfig:
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
    def __init__(self, config: MOEconfig):
        super().__init__()
        self.FNN = SwiGlU(config.d_model, config.d_expert, config.dropout)
    def forward(self, x: torch.Tensor)->torch.Tensor:
        return self.FNN(x)

class SharedExpert(nn.Module):
    def __init__(self, config: MOEconfig):
        super().__init__()
        self.shared_expert = nn.ModuleList([Expert(config) for _ in range (config.shared_expert)])
        self.scale = nn.Parameter(torch.ones(1)*0.1)
    def forward(self, x: torch.Tensor)->torch.Tensor:
        out = sum(expert(x) for expert in self.shared_expert)
        return out*self.scale

class RoutedExpert(nn.Module):
    def __init__(self, config: MOEconfig):
        super().__init__()
        self.config = config
        self.num_expert = config.routed_expert
        self.top_k = config.top_k_expert
        self.experts = nn.ModuleList([Expert(config) for _ in range (self.num_expert)])
        self.centroids = nn.Parameter(torch.randn(self.num_expert, self.d_model) * 0.006)
        
        self.register_buffer('bias', torch.zeros(self.num_expert))
        self.register_buffer('expert_load', torch.zeros(self.num_expert))

        self.scale = nn.Parameter(torch.ones(1)*0.1)
        experts_per_dev = config.experts_per_device
        device_map = []
        for d in range(config.num_devices):
            device_map.extend([d] * min(experts_per_dev, self.num_experts - len(device_map)))

        while len(device_map) < self.num_experts:
            device_map.append(config.num_devices - 1)
        self.register_buffer('expert_to_device', torch.tensor(device_map[:self.num_experts]))

    def forward(self, x: torch.Tensor)-> tuple[torch.Tensor, dict]:
        num_token = x.shpae[0]
        logits = x @ self.centroids.t()

        if self.config.aux_loss:
            logits = logits + self.bias
        
        scores = torch.sigmoid(logits)
        topk_idx, topk_gates = self._select_experts(scores, num_token)
        gate_sum = topk_gates.sum(dim= -1)
        topk_gates = topk_gates/ gate_sum
    

