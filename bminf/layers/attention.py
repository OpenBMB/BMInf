import math
import torch
from .linear import Linear

class GLMSelfAttention(torch.nn.Module):
    def __init__(
            self,
            dim_model : int,
            num_heads : int,
            dim_head : int,
            bias : bool,
            dtype : torch.dtype,
        ) -> None:
        super().__init__()

        self.dim_head = dim_head
        self.num_heads = num_heads

        self.weight_q = Linear(dim_model, num_heads * dim_head, bias=bias, dtype=dtype)
        self.weight_k = Linear(dim_model, num_heads * dim_head, bias=bias, dtype=dtype)
        self.weight_v = Linear(dim_model, num_heads * dim_head, bias=bias, dtype=dtype)

        self.attn_out = Linear(dim_model, num_heads * dim_head, bias=bias, dtype=dtype)

        self.softmax = torch.nn.Softmax(dim=-1)
    
    def forward(
            self,
            hidden_state : torch.Tensor,    # (batch, seq_len, dim_model)
            mask : torch.BoolTensor,        # (batch, seq_len, seq_len)
            position : torch.LongTensor,
        ):
        batch_size = hidden_state.size(0)
        len_k = len_q = hidden_state.size(1)

        h_q = self.weight_q(hidden_state)
        h_k = self.weight_k(hidden_state)

        h_q = h_q.view(batch_size, len_q, self.num_heads, self.dim_head).permute(0, 2, 1, 3)   # (batch, num_heads, len_q, dim_head)
        h_k = h_k.view(batch_size, len_k, self.num_heads, self.dim_head).permute(0, 2, 1, 3)   # (batch, num_heads, len_k, dim_head)

        h_q = self.rotary_embedding(h_q, position)  # (batch, num_heads, len_q, dim_head)
        h_k = self.rotary_embedding(h_k, position)  # (batch, num_heads, len_k, dim_head)


        score = torch.matmul(h_q, h_k.transpose(-1, -2)) / math.sqrt(self.dim_head)  # (batch, num_heads, len_q, len_k)
        del h_q
        del h_k

        score = torch.masked_fill(
            score,
            mask.view(batch_size, 1, len_q, len_k)==False,
            torch.scalar_tensor(float("-inf"), device=score.device, dtype=score.dtype)
        )   # (batch, num_heads, len_q, len_k)
        score = self.softmax(score)

        # avoid nan in softmax
        score = torch.masked_fill(
            score,
            mask.view(batch_size, 1, len_q, len_k)==False,
            torch.scalar_tensor(0, device=score.device, dtype=score.dtype)
        )

        h_v = self.weight_v(hidden_state)
        h_v = h_v.view(batch_size, len_k, self.num_heads, self.dim_head).permute(0, 2, 1, 3)   # (batch, num_heads, len_k, dim_head)

        score = torch.matmul(score, h_v)    # (batch, num_heads, len_q, dim_head)
        del h_v
        
        score = score.permute(0, 2, 1, 3).reshape(batch_size, len_q, self.num_heads * self.dim_head)

        return self.attn_out(score)


    def rotary_embedding(self,
            hidden : torch.Tensor,          # (batch, num_heads, seq_len, dim_head)
            position : torch.LongTensor     # (batch, seq_len)
        ):
        dim = hidden.size(-1)

        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2, device=hidden.device).float() / dim))    # (dim_head/2)
        inv_freq = inv_freq.half()
        freqs = torch.einsum('bi,j->bij', position.half(), inv_freq)  # (batch, seq_len, dim_head/2)
        emb = torch.cat((freqs, freqs), dim=-1) # (batch, seq_len, dim_head)
        v_cos = emb.cos()   # (batch, seq_len, dim_head)
        v_sin = emb.sin()   # (batch, seq_len, dim_head)
        
        def rotate_half(x):
            x1, x2 = x[..., :x.size(-1) // 2], x[..., x.size(-1) // 2:]
            return torch.cat((-x2, x1), dim=x1.ndim - 1)
        
        return (hidden * v_cos[:, None, :, :]) + (rotate_half(hidden) * v_sin[:, None, :, :])
