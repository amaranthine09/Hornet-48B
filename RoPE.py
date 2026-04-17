import torch
import torch.nn
import torch.nn.functional as F 

class RoPE(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        