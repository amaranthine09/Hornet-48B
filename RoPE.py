import torch 
import torch.nn 

class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: float=10000.0):
        super().__init__()

        