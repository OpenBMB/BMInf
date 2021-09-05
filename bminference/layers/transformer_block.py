from typing import Optional

import cupy
from .base import Layer
from .attention import SelfAttention, PartialAttention
from .dense_gelu_dense import DenseGeluDense
from .layer_norm import LayerNorm
from ..allocator import Allocator
import logging
logger = logging.getLogger(__name__)

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
        assert hidden_state.dtype == cupy.float16
        logger.info("Encoder transformer block -- layer norm self-attn")
        x = self.layer_nrom_before_self_attn.forward(allocator, hidden_state) # copy hidden state, e -> e
        assert x.dtype == cupy.float16
        assert x.shape == (batch_size, dim_model, seq_len)

        logger.info("Encoder transformer block -- self attention")
        x = self.self_attention.forward(allocator, x, attention_mask, self_attn_position_bias)
        assert x.dtype == cupy.float16
        assert x.shape == (batch_size, dim_model, seq_len)

        if inplace:
            tensor_out += x
        else:
            tensor_out = tensor_out + x # copied here

        logger.info("Encoder transformer block -- layer norm ff")
        x = self.layer_nrom_before_ff.forward(allocator, tensor_out)
        assert x.dtype == cupy.float16
        assert x.shape == (batch_size, dim_model, seq_len)

        logger.info("Encoder transformer block -- ff")
        x = self.dense_gelu_dense.forward(allocator, x)
        assert x.dtype == cupy.float16
        assert x.shape == (batch_size, dim_model, seq_len)

        tensor_out += x
        return tensor_out

class TransformerBlockDecoder(Layer):
    def __init__(self, dim_model, dim_ff, dim_qkv, num_heads):
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_qkv = dim_qkv

        self.layer_nrom_before_self_attn = LayerNorm(dim_model)
        self.self_attention =  PartialAttention(dim_model, dim_qkv, num_heads, is_self_attn=True)

        self.layer_nrom_before_cross_attn = LayerNorm(dim_model)
        self.cross_attention = PartialAttention(dim_model, dim_qkv, num_heads, is_self_attn=False)

        self.layer_nrom_before_ff = LayerNorm(dim_model)
        self.dense_gelu_dense = DenseGeluDense(dim_model, dim_ff)

    def forward(self, allocator : Allocator, 
            curr_hidden_state : cupy.ndarray,   # (batch, dim_model)
            past_kv : cupy.ndarray,             # (batch, 2, num_heads, dim_kv, max_decoder_length)
            decoder_length : int,               # int
            encoder_mask : cupy.ndarray,        # (batch, encoder_len)
            encoder_kv : cupy.ndarray,          #  (batch, 2, num_heads, dim_kv, seq_ipt_len)
            self_attn_position_bias : Optional[cupy.ndarray] = None, # (1, num_heads, max_decoder_length, max_decoder_length)
            inplace : bool = True,
        ):

        # ==================================
        # check shapes

        batch_size, dim_model = curr_hidden_state.shape
        assert dim_model == self.dim_model
        assert curr_hidden_state.dtype == cupy.float16

        max_decoder_length = past_kv.shape[-1]
        assert past_kv.shape == (batch_size, 2, self.num_heads, self.dim_qkv, max_decoder_length)
        assert past_kv.dtype == cupy.float32

        encoder_len = encoder_kv.shape[-1]
        assert encoder_kv.shape == (batch_size, 2, self.num_heads, self.dim_qkv, encoder_len)
        assert encoder_mask.shape == (batch_size, encoder_len)
        assert encoder_kv.dtype == cupy.float32

        # ==================================
        # self attention
        logger.info("Decoder transformer block -- layer norm self-attn")
        normalized_hidden = self.layer_nrom_before_self_attn.forward(allocator, curr_hidden_state[:, :, cupy.newaxis])[:, :, 0]


        assert normalized_hidden.shape == (batch_size, dim_model)
        logger.info("Decoder transformer block -- self attention")
        attn_out = self.self_attention.forward(
            allocator,
            normalized_hidden,  # (batch, dim_model)
            past_kv,            # (batch, 2, num_heads, dim_qkv, max_decoder_length)
            self_attn_position_bias[:, :, :, decoder_length],   # (1, num_heads, max_decoder_length)
            (cupy.arange(max_decoder_length) <= decoder_length)[cupy.newaxis],   # (1#batch_size, max_decoder_length)
            decoder_length,
        )
        assert attn_out.shape == (batch_size, dim_model)
        if inplace:
            curr_hidden_state += attn_out
        else:
            curr_hidden_state = curr_hidden_state + attn_out
        
        # ==================================
        # cross attention
        logger.info("Decoder transformer block -- layer norm cross-attn")
        normalized_hidden = self.layer_nrom_before_cross_attn.forward(allocator, curr_hidden_state[:, :, cupy.newaxis])[:, :, 0]

        assert normalized_hidden.shape == (batch_size, dim_model)
        logger.info("Decoder transformer block -- cross attention")
        attn_out = self.cross_attention.forward(
            allocator,
            normalized_hidden,
            encoder_kv,     # (batch, 2, num_heads, dim_qkv, encoder_len)
            None,
            encoder_mask,   # (batch, encoder_len)
            None
        )
        assert attn_out.shape == (batch_size, dim_model)

        curr_hidden_state += attn_out

        logger.info("Decoder transformer block -- layer norm ff")
        normalized_hidden = self.layer_nrom_before_ff.forward(allocator, curr_hidden_state[:, :, cupy.newaxis])
        assert normalized_hidden.shape == (batch_size, dim_model, 1)
        
        logger.info("Decoder transformer block -- ff")
        ff_out = self.dense_gelu_dense.forward(allocator, normalized_hidden)
        assert ff_out.shape == (batch_size, dim_model, 1)
        curr_hidden_state += ff_out[:, :, 0]
        return curr_hidden_state
        