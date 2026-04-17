import math
import torch 
import torch.nn as nn 
import torch.nn.fucntional as F 
from RoPE import RoPE
class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        d_model: int, 
        n_head: int, 
        d_KV_latent: int,
        d_Q_latent: int,
        d_head: int,
        d_rope: int, 
        max_seq_len: int=4096,
        feq_theta: int=10000.0
        ):

        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.d_rope = d_rope
        self.h_t_dim = d_head + d_rope
        self.scale = (d_head + d_rope)**0.5

        self.KV_down_proj = nn.Linear(d_model, d_KV_latent, bias = False)
        self.K_Content_up_proj = nn.Linear(d_KV_latent, n_head*d_head, bias = False)
        self.V_up_proj = nn.Linear(d_KV_latent, n_head*d_head, bias = False)

        self.Q_down_proj = nn.Linear(d_model, d_Q_latent, bias = False)
        self.Q_Content_up_proj = nn.Linear(d_Q_latent, n_head*d_head, bias = False)

        self.Q_rope_up = nn.Linear(d_Q_latent, n_head*d_rope, bias = False)
        self.K_rope_up = nn.Linear(d_model, n_head*d_rope, bias = False)

        self.out_proj = nn.Linear(n_head*d_head, d_model, bias = False)

        self.register_buffer("freqs_cis", precompute_freqs_cis(d_rope, max_seq_len, theta=rope_theta),persistent=False,)

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(
        self, 
        x: torch.Tensor, 
        mask:Optional[torch.Tensor],
        freqs_cis: Optional[torch.Tensor]
        )->torch.Tensor:

        b_size, seq_len, _ = x.shape

        down_KV = self.KV_down_proj(x)
        K_content = self.Q_Content_up_proj(down_KV).view(b_size, seq_len, self.n_head, self.d_head)
        k_rope = self.K_rope_up(x).view(b_size, seq_len, self.n_head, self.d_head)

        V = self.V_up_proj(down_KV).view(b_size, seq_len, self.n_head, self.d_head)

        down_Q = self.Q_down_proj(x)
        Q_content = self.Q_Content_up_proj(down_Q).view(b_size, seq_len, self.n_head, self.d_head)
        Q_rope = self.Q_rope_up(down_Q).view(b_size, seq_len, self.n_head, self.d_head)

        if freqs_cis is None:
            freqs_cis = self.freqs_cis[:seq_len]

        Q_rope= RoPE(Q_rope, freqs_cis)
        K_rope= RoPE(K_rope, freqs_cis)

        Q = torch.cat(Q_content,Q_rope).transpose(1,2)
        K = torch.cat(K_content,K_rope).transpose(1,2)
        V = V.transpose(1,2)

        attn_waight = torch.matmul(Q,K.transpose(-2,-1))*self.scale
        if mask is not None:
            attn_waight = attn_waight +mask
        attn_prob = F.softmax(attn_waight, dim= -1, dtype=torch.float32).to(Q.dtype)

        out = torch.matmul(attn_prob, V)
        out = out.transpose(1, 2).contiguous().view(b_Size, seq_len, self.n_head*d_head)
        out = self.out_proj(out)

        return out
















