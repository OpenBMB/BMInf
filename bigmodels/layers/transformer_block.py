from typing import Optional

import cupy
from .base import Layer
from .attention import SelfAttention, CrossAttention
from .dense_gelu_dense import DenseGeluDense
from .layer_norm import LayerNorm
from ..allocator import Allocator

class TransformerBlockEncoder(Layer):
    def __init__(self, dim_model, dim_ff, dim_qkv, num_heads):
        self.dim_model = dim_model

        self.layer_nrom_before_self_attn = LayerNorm(dim_model)
        self.self_attention =  SelfAttention(dim_model, dim_qkv, num_heads)

        self.layer_nrom_before_ff = LayerNorm(dim_model)
        self.dense_gelu_dense = DenseGeluDense(dim_model, dim_ff)

    def forward(self, allocator : Allocator, 
            hidden_state : cupy.ndarray, 
            attention_mask : cupy.ndarray,
            self_attn_position_bias : Optional[cupy.ndarray] = None, 
            inplace : bool = True
        ):
        if not cupy.issubdtype(hidden_state.dtype, cupy.floating):
            raise NotImplementedError("transformer block for integer input is not implemented")

        batch_size, dim_model, seq_len = hidden_state.shape
        assert dim_model == self.dim_model

        tensor_out = hidden_state

        assert hidden_state.dtype == cupy.float32
        x = self.layer_nrom_before_self_attn.forward(allocator, hidden_state, inplace=False) # copy hidden state, f -> f
        
        assert x.dtype == cupy.float32
        assert x.shape == (batch_size, dim_model, seq_len)
        x = self.self_attention.forward(allocator, x, attention_mask, self_attn_position_bias)
        assert x.dtype == cupy.float32
        assert x.shape == (batch_size, dim_model, seq_len)
        if inplace:
            tensor_out += x
        else:
            tensor_out = tensor_out + x # copied here
        
        x = self.layer_nrom_before_ff.forward(allocator, tensor_out, inplace=False)
        assert x.dtype == cupy.float32
        assert x.shape == (batch_size, dim_model, seq_len)
        x = self.dense_gelu_dense.forward(allocator, x)
        assert x.dtype == cupy.float32
        assert x.shape == (batch_size, dim_model, seq_len)

        tensor_out += x
        return tensor_out

class TransformerBlockDecoder(Layer):
    def __init__(self, dim_model, dim_ff, dim_qkv, num_heads):
        self.dim_model = dim_model
        self.layer_nrom_before_self_attn = LayerNorm(dim_model)
        self.self_attention =  SelfAttention(dim_model, dim_qkv, num_heads)

        self.layer_nrom_before_cross_attn = LayerNorm(dim_model)
        self.cross_attention = CrossAttention(dim_model, dim_qkv, num_heads)

        self.layer_nrom_before_ff = LayerNorm(dim_model)
        self.dense_gelu_dense = DenseGeluDense(dim_model, dim_ff)

    def forward(self, allocator : Allocator, 
            hidden_state : cupy.ndarray, 
            attention_mask : cupy.ndarray,
            self_attn_position_bias : Optional[cupy.ndarray] = None, 
            cross_attention_mask : Optional[cupy.ndarray] = None, 
            cross_attn_position_bias : Optional[cupy.ndarray] = None, 
            encode_kv : Optional[cupy.ndarray] = None,
            past_kv : Optional[cupy.ndarray] = None,
            inplace : bool = True
        ):
        # FIXME: cross attn not implemented

        pass