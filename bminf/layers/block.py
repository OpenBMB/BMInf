import torch
from .attention import GLMSelfAttention
from .ff import FeedForward

class GLMBlock(torch.nn.Module):
    def __init__(
            self,
            dim_model : int,
            num_heads : int,
            dim_head : int,
            dim_ff : int,
            gated : bool,
            gelu : bool,
            alpha : float,
            eps : float,
            dtype : torch.dtype,
        ) -> None:
        super().__init__()

        self.input_layernorm = torch.nn.LayerNorm(dim_model, eps=eps, dtype=dtype)
        self.attention = GLMSelfAttention(dim_model, num_heads, dim_head, True, dtype)

        self.post_attention_layernorm = torch.nn.LayerNorm(dim_model, eps=eps, dtype=dtype)
        self.ff = FeedForward(dim_model, dim_ff, gated, gelu, True, dtype)
        self.alpha = alpha
    
    def forward(
            self,
            hidden_state : torch.Tensor,    # (batch, seq_len, dim_model)
            mask : torch.BoolTensor,        # (batch, seq_len, seq_len)
            position : torch.LongTensor
        ):
        attn_input = self.input_layernorm(hidden_state)
        attn_output = self.attention(attn_input, mask, position)
        hidden_state = attn_input * self.alpha + attn_output

        mlp_input = self.post_attention_layernorm(hidden_state)
        mlp_output = self.ff(mlp_input)

        return mlp_input * self.alpha + mlp_output
