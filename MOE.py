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
        gate_sum = topk_gates.sum(dim = -1, keepdim = True).clamp_min(1e-9)
        topk_gates = topk_gates/ gate_sum

        output = self._compute_expert_outputs(x, topk_idx, topk_gates)

        if self.training and self.config.aux_loss:
            self._update_bias(topk_idx, num_token)
        
        states = self._compute_states(topk_idx, topk_gates, num_token)
        return output*self.scale, states
    
    def _select_experts(self, scores: torch.Tensor, num_tokens: int)->tuple[torch.Tensor, torch.Tensor]:
        if not self.config.use_device_limit or self.config.no_of_device:
            gates, idx = torch.topk(scores, self.topk, dim =-1)
            return idx, gates
        M = self.config.no_of_device
        device_max = torch.full(
            (num_tokens, self.config.num_devices),
            float('-inf'),
            device=scores.device,
            dtype=scores.dtype
        )

        device_max.scatter_reduce_(
            1, 
            self.expert_to_device.unsqueeze(0).expand(num_tokens, -1),
            scores,
            reduce='amax'
        )

        _, top_devices = torch.topk(device_max, min(M, self.config.num_devices), dim=-1)
        mask = torch.zeros(num_tokens, self.num_experts, dtype=torch.bool, device=scores.device)
        for m in range(top_devices.shape[1]):
            dev_id = top_devices[:, m]
            on_device = (self.expert_to_device.unsqueeze(0) == dev_id.unsqueeze(1))
            mask |= on_device
        masked_scores = scores.masked_fill(~mask, float('-inf'))
        gates, idx = torch.topk(masked_scores, self.top_k, dim=-1)
        
        return idx, gates

    def _compute_expert_outputs(self, x: torch.Tensor, topk_idx: torch.Tensor, topk_gates: torch.Tensor)->torch.Tensor:
        num_tokens = x.shape[0]
        output = torch.zeros_like(x)
        for e_idx in range (self.num_expert):
            in_topk = (topk_idx==e_idx)
            if not in_topk.any():
                continue
            token_idx, pos_in_topk = torch.where(in_topk)
            gates = topk_gates[token_idx, pos_in_topk]
            
            expert_input = x[token_idx]  
            expert_out = self.experts[e_idx](expert_input)  
            
            weighted = expert_out * gates.unsqueeze(-1)  
            output.index_add_(0, token_idx, weighted)
        
        return output

    def _update_bias(self, topk_idx: torch.Tensor, num_tokens: int):
        counts = torch.zeros(self.num_experts, device=topk_idx.device)
        counts.index_add_(0, topk_idx.view(-1), torch.ones_like(topk_idx.view(-1), dtype=torch.float))
        
        total = num_tokens * self.top_k
        load = counts / total
        expected = 1.0 / self.num_experts
    
        gamma = self.config.aux_loss_gamma
        self.bias = torch.where(load > expected, self.bias - gamma, 
                               torch.where(load < expected, self.bias + gamma, self.bias))
        self.bias = self.bias.clamp(-1.0, 1.0)
        self.expert_load = load


    def _compute_stats(
        self, 
        topk_idx: torch.Tensor, 
        topk_gates: torch.Tensor,
        num_tokens: int
    ) -> Dict[str, torch.Tensor]:
        counts = torch.zeros(self.num_experts, device=topk_idx.device)
        counts.index_add_(0, topk_idx.view(-1), torch.ones_like(topk_idx.view(-1), dtype=torch.float))
        f_i = counts / (num_tokens * self.top_k)
        
        P_i = torch.zeros(self.num_experts, device=topk_idx.device)
        P_i.index_add_(0, topk_idx.view(-1), topk_gates.view(-1))
        P_i = P_i / num_tokens
        
        balance_loss = (f_i * P_i).sum()
        
        if self.config.num_devices > 1:
            device_load = torch.zeros(self.config.num_devices, device=topk_idx.device)
            for d in range(self.config.num_devices):
                mask = (self.expert_to_device == d)
                device_load[d] = f_i[mask].sum()
        else:
            device_load = f_i.sum().unsqueeze(0)
        
        return {
            'expert_load': f_i,          
            'avg_gate': P_i,              
            'balance_loss': balance_loss, 
            'device_load': device_load,   
            'bias_mean': self.bias.mean(),
            'bias_std': self.bias.std(),  
        }
    
class DeepSeekMoELayer(nn.Module):
    def __init__(self, config: MOEconfig):
        super().__init__()
        self.config = config
        self.norm = RMSNorm(config.hidden_dim)
        self.shared = SharedExpert(config)
        self.routed = RoutedExpert(config)
        self.post_norm = RMSNorm(config.hidden_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
       
        orig_shape = x.shape
        if x.dim() == 3:
            x = x.view(-1, x.shape[-1])
        normed = self.norm(x)
        shared_out = self.shared(normed)
    
        routed_out, stats = self.routed(normed)
        routed_out = self.post_norm(routed_out)
        output = x + shared_out + routed_out
        output = output.view(orig_shape)
        
        return output, stats


