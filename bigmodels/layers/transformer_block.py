from .base import Layer
from .attention import SelfAttention, CrossAttention
from .dense_gelu_dense import DenseGeluDense
from .layer_norm import LayerNorm

class TransformerBlock(Layer):
    TYPE_F32 = 1
    TYPE_F16 = 2
    TYPE_I8 = 3
    def __init__(self, is_decoder, dim_model, dim_ff, dim_qkv, num_heads, ltype):
        self.is_decoder = is_decoder

        self.layer_nrom_before_self_attn = LayerNorm(dim_model, ltype=LayerNorm.TYPE_F32)
        self.self_attention =  SelfAttention(dim_model, dim_qkv, num_heads, ltype)
        if is_decoder:
            self.layer_nrom_before_cross_attn = LayerNorm(dim_model, ltype=LayerNorm.TYPE_F32)
            self.cross_attention = CrossAttention(dim_model, dim_qkv, num_heads, ltype)

        self.layer_nrom_before_ff = LayerNorm(dim_model, ltype=ltype)
        self.dense_gelu_dense = DenseGeluDense(dim_model, dim_ff, ltype)

    def forward(self, x):
        # call function linear

        pass