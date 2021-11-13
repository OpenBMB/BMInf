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
    
    def backward(self,
            ctx : Context,
            x : Tensor,         # (batch, dim_model, seq_q)
            position_bias : Optional[Tensor],
            mask : Tensor,
            accum_grad : Tensor
        ):
        batch, dim_model, seq_len = x.shape

        x_mid_1 = ctx.allocate((batch, dim_model, seq_len), np.float16)
        self.ln_attn.forward(ctx, x, x_mid_1)

        x_mid_2 = ctx.allocate((batch, dim_model, seq_len), np.float16)
        self.self_attn.forward(ctx, x_mid_1, x_mid_1, mask, position_bias, x_mid_2)

        ck.arith_element_add(
            batch, dim_model * seq_len,
            x.ptr, x_mid_2.ptr,
            x_mid_2.ptr,
            ctx.current_stream
        )
        x_mid_3 = ctx.allocate((batch, dim_model, seq_len), np.float16)
        self.ln_ff.forward(ctx, x_mid_2, x_mid_3)

        # start backward

        grad_tmp = ctx.allocate((batch, dim_model, seq_len), np.float16)
        grad_tmp.zero_(ctx)
        self.ff.backward(ctx, x_mid_3, accum_grad, grad_tmp)
        ctx.free(x_mid_3)

        self.ln_ff.backward(ctx, x_mid_2, grad_tmp, accum_grad)
        ctx.free(x_mid_2)

        grad_tmp.zero_(ctx)
        self.self_attn.backward(
            ctx,
            x_mid_1, x_mid_1, mask, position_bias,
            accum_grad, grad_tmp, grad_tmp
        )
        ctx.free(x_mid_1)
        self.ln_attn.backward(ctx, x, grad_tmp, accum_grad)
        ctx.free(grad_tmp)


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
    
    def backward(self,
            ctx : Context,
            x : Tensor,                     # (batch, dim_model, seq_q)
            mask_x : Tensor,                # (batch, seq_q, seq_q)
            bias_self : Optional[Tensor],   # (num_heads, seq_q, seq_q)
            accum_grad : Tensor             # (batch, dim_model, seq_q)
        ):
        
        batch, dim_model, seq_len = x.shape

        x_mid_1 = ctx.allocate(x.shape, x.dtype)
        self.ln_attn.forward(ctx, x, x_mid_1)

        x_mid_2 = ctx.allocate(x.shape, x.dtype)
        self.self_attn.forward(ctx, x_mid_1, x_mid_1, mask_x, bias_self, x_mid_2)

        ck.arith_element_add(
            batch, dim_model * seq_len,
            x.ptr, x_mid_2.ptr,
            x_mid_2.ptr,
            ctx.current_stream
        )

        x_mid_3 = ctx.allocate(x.shape, x.dtype)
        self.ln_ff.forward(ctx, x_mid_2, x_mid_3)
        
        # start backward

        grad_tmp = ctx.allocate(x.shape, x.dtype)
        grad_tmp.zero_(ctx)

        self.ff.backward(ctx, x_mid_3, accum_grad, grad_tmp)
        ctx.free(x_mid_3)

        self.ln_ff.backward(ctx, x_mid_2, grad_tmp, accum_grad)
        ctx.free(x_mid_2)

        grad_tmp.zero_(ctx)
        self.self_attn.backward(ctx,
            x_mid_1, x_mid_1, mask_x, bias_self, accum_grad,
            grad_tmp, grad_tmp
        )
        ctx.free(x_mid_1)

        self.ln_attn.backward(ctx, x, grad_tmp, accum_grad)
        ctx.free(grad_tmp)

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
            x : Tensor,                         # (batch, dim_model)
            encoder_output : Optional[Tensor],  # (batch, dim_model, seq_k) can be None if step_pos >
            mask_x : Tensor,                    # (batch, buffer_len)
            mask_cross : Tensor,                # (batch, seq_k)
            bias_self : Optional[Tensor],       # (num_heads, buffer_len)
            bias_cross : Optional[Tensor],      # (num_heads, seq_k)

            # buffers
            past_k_self : Tensor,               # (batch, num_heads, buffer_len, dim_head)
            past_v_self : Tensor,               # (batch, num_heads, buffer_len, dim_head)
            past_k_cros : Tensor,               # (batch, num_heads, seq_k, dim_head)
            past_v_cros : Tensor,               # (batch, num_heads, seq_k, dim_head)

            step_pos : int,
            x_out : Tensor,                     # (batch, dim_model)
        ):
        batch, dim_model = x.shape
        assert step_pos > 0 or (encoder_output is not None and encoder_output.shape[:2] == (batch, dim_model))
        assert past_k_self.shape == past_v_self.shape
        assert past_k_cros.shape == past_v_cros.shape
        assert past_k_cros.shape[2] == mask_cross.shape[1]
        assert past_k_self.shape[:2] == past_k_cros.shape[:2]

        x_mid = ctx.allocate(x.shape, x.dtype)
        self.ln_attn.step(ctx, x, x_mid)
        self.self_attn.step(
            ctx, 
            x_mid, past_k_self, past_v_self,
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

        if step_pos == 0:
            # init past_kv_cros for the first step
            assert encoder_output is not None
            self.cross_attn.init_kv(
                ctx,
                encoder_output,
                past_k_cros, past_v_cros
            )

        self.ln_cross_attn.step(ctx, x_out, x_mid)
        self.cross_attn.step(
            ctx,
            x_mid, past_k_cros, past_v_cros,
            mask_cross, bias_cross,
            x_mid,
            False, step_pos
        )
        ck.arith_element_add(
            batch, dim_model,
            x_out.ptr, x_mid.ptr,
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
    
    def backward(self,
            ctx : Context,
            x : Tensor,                 # (batch, dim_model, seq_q)
            encoder_output : Tensor,    # (batch, dim_model, seq_k)
            mask_x : Tensor,            # (batch, seq_q, seq_q)
            mask_cross : Tensor,        # (batch, seq_k, seq_q)
            bias_self : Optional[Tensor],   # (num_heads, seq_q, seq_q)
            bias_cross: Optional[Tensor],   # (num_heads, seq_k, seq_q)
            accum_grad : Tensor,
            encoder_grad : Tensor
        ):
        batch, dim_model, seq_len = x.shape

        x_mid_1 = ctx.allocate((batch, dim_model, seq_len), np.float16)

        self.ln_attn.forward(ctx, x, x_mid_1)

        x_mid_2 = ctx.allocate((batch, dim_model, seq_len), np.float16)
        self.self_attn.forward(
            ctx, x_mid_1, x_mid_1, mask_x, bias_self, x_mid_2
        )
        
        ck.arith_element_add(
            batch, dim_model * seq_len,
            x.ptr, x_mid_2.ptr,
            x_mid_2.ptr,
            ctx.current_stream
        )

        x_mid_3 = ctx.allocate((batch, dim_model, seq_len), np.float16)
        self.ln_cross_attn.forward(ctx, x_mid_2, x_mid_3)

        x_mid_4 = ctx.allocate((batch, dim_model, seq_len), np.float16)
        self.cross_attn.forward(
            ctx, x_mid_3, encoder_output, mask_cross, bias_cross, x_mid_4
        )
        ck.arith_element_add(
            batch, dim_model * seq_len,
            x_mid_2.ptr, x_mid_4.ptr,
            x_mid_4.ptr,
            ctx.current_stream
        )

        x_mid_5 = ctx.allocate((batch, dim_model, seq_len), np.float16)
        self.ln_ff.forward(ctx, x_mid_4, x_mid_5)

        # start backward

        grad_tmp = ctx.allocate((batch, dim_model, seq_len), np.float16)

        grad_tmp.zero_(ctx)
        self.ff.backward(ctx, x_mid_5, accum_grad, grad_tmp)
        ctx.free(x_mid_5)
        
        self.ln_ff.backward(ctx, x_mid_4, grad_tmp, accum_grad)
        ctx.free(x_mid_4)

        grad_tmp.zero_(ctx)
        self.cross_attn.backward(
            ctx, x_mid_3, encoder_output, mask_cross, bias_cross, 
            accum_grad, grad_tmp, encoder_grad
        )
        ctx.free(x_mid_3)
        
        self.ln_cross_attn.backward(ctx, x_mid_2, grad_tmp, accum_grad)
        ctx.free(x_mid_2)

        grad_tmp.zero_(ctx)
        self.self_attn.backward(
            ctx, x_mid_1, x_mid_1, mask_x, bias_self,
            accum_grad, grad_tmp, grad_tmp
        )
        ctx.free(x_mid_1)

        self.ln_attn.backward(ctx, x, grad_tmp, accum_grad)
        ctx.free(grad_tmp)
