from typing import Optional
from ..core import Layer, Context, Tensor, Parameter
import numpy as np
from cpm_kernels import kernels as ck
from .layernorm import Layernorm
from .attention import Attention
from .feedforward import FeedForward


class EncoderBlock(Layer):
    def __init__(self, 
            dim_model : int, num_heads : int, dim_head : int, dim_ff : int,
            eps : float, bias : bool = False, gated : bool = True
        ):
        super().__init__()

        self.ln_attn = Layernorm(dim_model, eps, bias)
        self.self_attn = Attention(dim_model, num_heads, dim_head, bias)

        self.ln_ff = Layernorm(dim_model, eps, bias)
        self.ff = FeedForward(dim_model, dim_ff, bias, gated)
    
    def forward(self, 
            ctx : Context, 
            x : Tensor, 
            position_bias : Optional[Tensor],
            mask : Tensor,
            x_out : Tensor, 
        ):
        batch, dim_model, seq_len = x.shape

        x_mid = ctx.allocate(x.shape, x.dtype)
        self.ln_attn.forward(ctx, x, x_mid)
        self.self_attn.forward(ctx, x_mid, x_mid, mask, position_bias, x_mid)
        ck.arith_element_add(
            batch, dim_model * seq_len,
            x.ptr, x_mid.ptr,
            x_out.ptr,
            ctx.current_stream
        )

        self.ln_ff.forward(ctx, x_out, x_mid)
        self.ff.forward(ctx, x_mid, x_mid)
        ck.arith_element_add(
            batch, dim_model * seq_len,
            x_out.ptr, x_mid.ptr,
            x_out.ptr,
            ctx.current_stream
        )
        ctx.free(x_mid)


class DecoderBlock(Layer):
    def __init__(self, 
            dim_model : int, num_heads : int, dim_head : int, dim_ff : int,
            eps : float, bias : bool = False, gated : bool = True
        ):
        super().__init__()

        self.ln_attn = Layernorm(dim_model, eps, bias)
        self.self_attn = Attention(dim_model, num_heads, dim_head, bias)

        self.ln_ff = Layernorm(dim_model, eps, bias)
        self.ff = FeedForward(dim_model, dim_ff, bias, gated)
    
    def forward(self, 
            ctx : Context, 
            x : Tensor,                 # (batch, dim_model, seq_q)
            mask_x : Tensor,            # (batch, seq_q, seq_q)
            bias_self : Optional[Tensor],   # (num_heads, seq_q, seq_q)
            x_out : Tensor, 
        ):
        batch, dim_model, seq_len = x.shape

        x_mid = ctx.allocate(x.shape, x.dtype)
        self.ln_attn.forward(ctx, x, x_mid)
        self.self_attn.forward(ctx, x_mid, x_mid, mask_x, bias_self, x_mid)

        ck.arith_element_add(
            batch, dim_model * seq_len,
            x.ptr, x_mid.ptr,
            x_out.ptr,
            ctx.current_stream
        )

        self.ln_ff.forward(ctx, x_out, x_mid)
        self.ff.forward(ctx, x_mid, x_mid)
        ck.arith_element_add(
            batch, dim_model * seq_len,
            x_out.ptr, x_mid.ptr,
            x_out.ptr,
            ctx.current_stream
        )
        ctx.free(x_mid)

    def step(self, 
            ctx : Context, 
            x : Tensor,                     # (batch, dim_model)
            mask_x : Tensor,                # (batch, kv_buffer_len)
            bias_self : Optional[Tensor],   # (num_heads, kv_buffer_len)
            past_k : Tensor,                # (batch, num_heads, kv_buffer_len, dim_head)
            past_v : Tensor,                # (batch, num_heads, kv_buffer_len, dim_head)
            step_pos : int,
            x_out : Tensor,                 # (batch, dim_model)
        ):
        batch, dim_model = x.shape
        assert past_k.shape == past_v.shape
        assert past_k.shape[0] == batch
        kv_buffer_len = past_k.shape[2]
        assert mask_x.shape == (batch, kv_buffer_len)
        assert kv_buffer_len > step_pos

        x_mid = ctx.allocate(x.shape, x.dtype)
        self.ln_attn.step(ctx, x, x_mid)
        self.self_attn.step(
            ctx, x_mid, past_k, past_v,
            mask_x, bias_self,
            x_mid,
            True, step_pos
        )

        ck.arith_element_add(
            batch, dim_model,
            x.ptr, x_mid.ptr,
            x_out.ptr,
            ctx.current_stream
        )

        self.ln_ff.step(ctx, x_out, x_mid)
        self.ff.step(ctx, x_mid, x_mid)

        ck.arith_element_add(
            batch, dim_model,
            x_out.ptr, x_mid.ptr,
            x_out.ptr,
            ctx.current_stream
        )
        ctx.free(x_mid)


class DecoderBlockWithCrossAttention(Layer):
    def __init__(self, 
            dim_model : int, num_heads : int, dim_head : int, dim_ff : int,
            eps : float, bias : bool = False, gated : bool = True
        ):
        super().__init__()

        self.ln_attn = Layernorm(dim_model, eps, bias)
        self.self_attn = Attention(dim_model, num_heads, dim_head, bias)

        self.ln_cross_attn = Layernorm(dim_model, eps, bias)
        self.cross_attn = Attention(dim_model, num_heads, dim_head, bias)

        self.ln_ff = Layernorm(dim_model, eps, bias)
        self.ff = FeedForward(dim_model, dim_ff, bias, gated)
    
    def forward(self, 
            ctx : Context, 
            x : Tensor,                 # (batch, dim_model, seq_q)
            encoder_output : Tensor,    # (batch, dim_model, seq_k)
            mask_x : Tensor,            # (batch, seq_q, seq_q)
            mask_cross : Tensor,        # (batch, seq_k, seq_q)
            bias_self : Optional[Tensor],   # (num_heads, seq_q, seq_q)
            bias_cross: Optional[Tensor],   # (num_heads, seq_k, seq_q)
            x_out : Tensor, 
        ):
        batch, dim_model, seq_len = x.shape

        x_mid = ctx.allocate(x.shape, x.dtype)
        self.ln_attn.forward(ctx, x, x_mid)
        self.self_attn.forward(ctx, x_mid, x_mid, mask_x, bias_self, x_mid)
        ck.arith_element_add(
            batch, dim_model * seq_len,
            x.ptr, x_mid.ptr,
            x_out.ptr,
            ctx.current_stream
        )

        self.ln_cross_attn.forward(ctx, x_out, x_mid)
        self.cross_attn.forward(ctx, x_mid, encoder_output, mask_cross, bias_cross, x_mid)
        ck.arith_element_add(
            batch, dim_model * seq_len,
            x_out.ptr, x_mid.ptr,
            x_out.ptr,
            ctx.current_stream
        )

        self.ln_ff.forward(ctx, x_out, x_mid)
        self.ff.forward(ctx, x_mid, x_mid)
        ck.arith_element_add(
            batch, dim_model * seq_len,
            x_out.ptr, x_mid.ptr,
            x_out.ptr,
            ctx.current_stream
        )
        ctx.free(x_mid)

    def step(self, 
            ctx : Context, 
            x : Tensor,                     # (batch, dim_model)
            encoder_output : Tensor,        # (batch, dim_model, seq_k)
            mask_x : Tensor,                # (batch, seq_q, seq_q)
            mask_cross : Tensor,            # (batch, seq_k, seq_q)
            bias_self : Optional[Tensor],   # (num_heads, seq_q, seq_q)
            bias_cross : Optional[Tensor],  # (num_heads, seq_k, seq_q)
            past_k : Tensor,                # (batch, num_heads, kv_buffer_len, dim_head)
            past_v : Tensor,                # (batch, num_heads, kv_buffer_len, dim_head)
            step_pos : int,
            x_out : Tensor,                 # (batch, dim_model)
        ):
        batch, dim_model = x.shape
        assert encoder_output.shape == past_k.shape
        assert encoder_output.shape[0] == batch
        kv_buffer_len = past_k.shape[2]
        assert mask_x.shape == (batch, kv_buffer_len)
        assert mask_cross.shape == (batch, kv_buffer_len)
        assert kv_buffer_len > step_pos
    
