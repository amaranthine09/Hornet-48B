import torch 
import torch.nn as nn 
import torch.nn.fucntional as F 
class multiheadlatentattention(nn.Module):
    def __init__(self, d_model: int, n_head: int, h_dim: int = None, dropout: float):
        super().__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.h_dim = h_dim if h_dim is None else d_model//n_head
        self.dropout = droupout

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.W_O = nn.Linear(d_mdoel, d_model)

    def SplitsHead(self, x: torch.Tensor):
        b_size, seq_len, _ =x.shape
        x = x.view(b_size, seq_len , self.n_head, self.h_dim)
        return x.transpose(1, 2)
    
    def ConcatHead(self, x: torch.Tensor)

    def forward(self, x: torch.Tensor):
        



