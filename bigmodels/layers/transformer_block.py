from typing import Optional

import cupy
from .base import Layer
from .attention import SelfAttention, CrossAttention
from .dense_gelu_dense import DenseGeluDense
from .layer_norm import LayerNorm
from ..tensor import Tensor
from ..allocator import Allocator

class TransformerBlock(Layer):
    def __init__(self, is_decoder, dim_model, dim_ff, dim_qkv, num_heads):
        self.is_decoder = is_decoder

        self.layer_nrom_before_self_attn = LayerNorm(dim_model)
        self.self_attention =  SelfAttention(dim_model, dim_qkv, num_heads)
        if is_decoder:
            self.layer_nrom_before_cross_attn = LayerNorm(dim_model)
            self.cross_attention = CrossAttention(dim_model, dim_qkv, num_heads)

        self.layer_nrom_before_ff = LayerNorm(dim_model)
        self.dense_gelu_dense = DenseGeluDense(dim_model, dim_ff)

    def forward(self, allocator : Allocator, 
            hidden_state : Tensor, 
            attention_mask : cupy.ndarray,
            self_attn_position_bias : Optional[Tensor] = None, 
            cross_attention_mask : Optional[cupy.ndarray] = None, 
            cross_attn_position_bias : Optional[Tensor] = None, 
            encode_kv : Optional[Tensor] = None,
            past_kv : Optional[Tensor] = None,
        ):
        if not cupy.issubdtype(hidden_state.value.dtype, cupy.floating):
            raise NotImplementedError("transformer block for integer input is not implemented")

        batch_size, seq_len, dim_model = hidden_state.value.shape

        assert hidden_state.value.dtype == cupy.float32
        x = self.layer_nrom_before_self_attn.forward(allocator, hidden_state, inplace=False) # copy hidden state, f -> f
        assert x.value.dtype == cupy.float32
        self.self_attention.forward(allocator, x, attention_mask, self_attn_position_bias)

