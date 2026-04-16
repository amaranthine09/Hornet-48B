import torch 
import torch.nn as nn 
import torch.nn.fucntional as F 
class multiheadlatentattention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.dropout = droupout

        