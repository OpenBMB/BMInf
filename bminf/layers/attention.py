from typing import Optional
from ..core import Layer, Context, Tensor
import numpy as np
from cpm_kernels import kernels as ck
from .linear import Linear

class Attention(Layer):
    def __init__(self, dim_model, num_heads, dim_head, bias=False) -> None:
        super().__init__()

        self.dim_model = dim_model
        self.dim_head = dim_head
        self.num_heads = num_heads


        self.project_q = Linear(dim_model, dim_head * num_heads, bias=bias)
        self.project_k = Linear(dim_model, dim_head * num_heads, bias=bias)
        self.project_v = Linear(dim_model, dim_head * num_heads, bias=bias)

        self.linear_out = Linear(num_heads * dim_head, dim_model, bias=bias)
        
    def forward(self, 
            ctx : Context, 
            hidden_q : Tensor,                      # (batch, dim_model, seq_q)
            hidden_kv : Tensor,                     # (batch, dim_model, seq_k)
            mask : Tensor,                          # (batch, seq_k, seq_q)
            position_bias : Optional[Tensor],       # (num_heads, seq_k, seq_q)
            x_out : Tensor,                         # (batch, dim_model, seq_q)
            key_out : Optional[Tensor] = None,      # (batch, num_head, seq_k, dim_head)
            value_out : Optional[Tensor] = None,    # (batch, num_head, seq_k, dim_head)
        ):
        assert hidden_q.shape[:2] == hidden_kv.shape[:2]
        assert hidden_q.dtype == np.float16 and hidden_kv.dtype == np.float16
        assert x_out.shape == hidden_q.shape
        batch, dim_model, seq_q = hidden_q.shape
        batch, dim_model, seq_k = hidden_kv.shape

        h_q = ctx.allocate((batch, self.dim_head * self.num_heads, seq_q), dtype=np.float16)
        h_k = ctx.allocate((batch, self.dim_head * self.num_heads, seq_k), dtype=np.float16)
        
        self.project_q.forward(ctx, hidden_q, h_q)
        self.project_k.forward(ctx, hidden_kv, h_k)

        if key_out is not None:
            ck.transpose(
                batch * self.num_heads, self.dim_head, seq_k,
                h_k.ptr, key_out.ptr,
                ctx.current_stream
            )

        h_q.reshape((batch * self.num_heads, self.dim_head, seq_q)) # (batch * num_heads, dim_head, seq_q)
        h_k.reshape((batch * self.num_heads, self.dim_head, seq_k)) # (batch * num_heads, dim_head, seq_k)
        h_attn = ctx.allocate((batch * self.num_heads, seq_k, seq_q), dtype=np.float16)
        ck.gemm_fp16(
            seq_q, self.dim_head, seq_k,
            batch * self.num_heads, batch * self.num_heads,
            False, True,
            h_q.ptr, h_k.ptr,
            h_attn.ptr,
            ctx.current_stream
        )
        ctx.free(h_q)
        ctx.free(h_k)

        if position_bias is not None:
            ck.arith_batch_add_forward(
                batch, self.num_heads * seq_k * seq_q,
                h_attn.ptr,
                position_bias.ptr,
                h_attn.ptr,
                ctx.current_stream
            )
        ck.mask(
            batch, self.num_heads, seq_k * seq_q,
            h_attn.ptr,
            mask.ptr,
            float("-inf"),
            h_attn.ptr,
            ctx.current_stream
        )

        h_attn.reshape((batch * self.num_heads, seq_k, seq_q))  # (batch * num_heads, seq_k, seq_q)
        ck.softmax_inplace_forward(
            batch * self.num_heads, seq_k, seq_q,
            h_attn.ptr,
            ctx.current_stream
        )

        h_v = ctx.allocate((batch, self.dim_head * self.num_heads, seq_k), dtype=np.float16)
        self.project_v.forward(ctx, hidden_kv, h_v)
        if value_out is not None:
            ck.transpose(
                batch * self.num_heads, self.dim_head, seq_k,
                h_v.ptr, value_out.ptr,
                ctx.current_stream
            )

        h_v.reshape((batch * self.num_heads, self.dim_head, seq_k)) # (batch * num_heads, dim_head, seq_k)

        attn_out = ctx.allocate((batch, self.dim_head * self.dim_head, seq_q), dtype=np.float16)


        ck.gemm_fp16(
            seq_q, seq_k, self.dim_head,
            batch * self.num_heads, batch * self.num_heads,
            False, False,
            h_attn.ptr, h_v.ptr,
            attn_out.ptr,
            ctx.current_stream
        )
        ctx.free(h_attn)
        ctx.free(h_v)

        self.linear_out.forward(ctx, attn_out, x_out)
        ctx.free(attn_out)

    def init_kv(self, 
            ctx : Context,
            encoder_output : Tensor,     # (batch, dim_model, seq_k)
            k_out : Tensor,              # (batch, num_head, seq_k, dim_head)
            v_out : Tensor,              # (batch, num_head, seq_k, dim_head) 
        ):
        batch, dim_model, seq_k = encoder_output.shape
        assert k_out.shape == (batch, self.num_heads, seq_k, self.dim_head)
        assert v_out.shape == (batch, self.num_heads, seq_k, self.dim_head)
        assert k_out.dtype == np.float16 and v_out.dtype == np.float16
    
        tmp = ctx.allocate((batch, self.num_heads * self.dim_head, seq_k), dtype=np.float16)
        self.project_k.forward(ctx, encoder_output, tmp)
        tmp.reshape((batch * self.num_heads, self.dim_head, seq_k))
        ck.transpose(
            batch * self.num_heads, self.dim_head, seq_k,
            tmp.ptr, k_out.ptr,
            ctx.current_stream
        )
        tmp.reshape((batch, self.num_heads * self.dim_head, seq_k))
        self.project_v.forward(ctx, encoder_output, tmp)
        tmp.reshape((batch * self.num_heads, self.dim_head, seq_k))
        ck.transpose(
            batch * self.num_heads, self.dim_head, seq_k,
            tmp.ptr, v_out.ptr,
            ctx.current_stream
        )
        ctx.free(tmp)

    def step(self, 
            ctx : Context,
            hidden_q : Tensor,      # (batch, dim_model)
            past_k : Tensor,        # (batch, num_head, past_kv_buffer_len, dim_head)
            past_v : Tensor,        # (batch, num_head, past_kv_buffer_len, dim_head)
            mask : Tensor,          # (batch, past_kv_buffer_len)
            position_bias : Optional[Tensor], # (num_heads, past_kv_buffer_len)
            x_out : Tensor,         # (batch, dim_model)
            is_self_attn : bool,
            decoder_pos : int
        ):
        batch, dim_model = hidden_q.shape
        assert dim_model == self.dim_model and hidden_q.dtype == np.float16
        kv_buffer_len = past_k.shape[2]
        assert past_k.shape == (batch, self.num_heads, kv_buffer_len, self.dim_head) and past_k.dtype == np.float16
        assert past_v.shape == (batch, self.num_heads, kv_buffer_len, self.dim_head) and past_v.dtype == np.float16

        h_q = ctx.allocate((batch, self.num_heads * self.dim_head), dtype=np.float16)
        self.project_q.step(ctx, hidden_q, h_q)

        if is_self_attn:
            h_k = ctx.allocate((batch, self.num_heads * self.dim_head), dtype=np.float16)
            h_v = ctx.allocate((batch, self.num_heads * self.dim_head), dtype=np.float16)
            self.project_k.step(ctx, hidden_q, h_k)
            self.project_v.step(ctx, hidden_q, h_v)

            # put in
            ck.copy_data_to_kv(
                batch * self.num_heads, kv_buffer_len, self.dim_head,
                h_k.ptr,
                past_k.ptr,
                decoder_pos,
                ctx.current_stream
            )
            ck.copy_data_to_kv(
                batch * self.num_heads, kv_buffer_len, self.dim_head,
                h_v.ptr,
                past_v.ptr,
                decoder_pos,
                ctx.current_stream
            )
            ctx.free(h_k)   
            ctx.free(h_v)
        
        attn_score = ctx.allocate((batch, self.num_heads, kv_buffer_len), np.float16)
        ck.gemv_fp16(
            batch * self.num_heads, kv_buffer_len, self.dim_head,
            past_k.ptr, h_q.ptr,
            attn_score.ptr,
            ctx.current_stream
        )
        ctx.free(h_q)
        if position_bias is not None:
            ck.arith_batch_add_forward(
                batch, self.num_heads * kv_buffer_len,
                attn_score.ptr,
                position_bias.ptr,
                attn_score.ptr,
                ctx.current_stream
            )
        ck.mask(
            batch, self.num_heads, kv_buffer_len,
            attn_score.ptr,
            mask.ptr,
            float("-inf"),
            attn_score.ptr,
            ctx.current_stream
        )
        ck.softmax_step_inplace(
            batch * self.num_heads, kv_buffer_len,
            attn_score.ptr,
            ctx.current_stream
        )
        attn_out = ctx.allocate((batch, self.num_heads  * self.dim_head), np.float16)
        ck.gemv_fp16_transpose(
            batch * self.num_heads, self.dim_head, kv_buffer_len,
            past_v.ptr, attn_score.ptr,
            attn_out.ptr,
            ctx.current_stream
        )
        ctx.free(attn_score)
        self.linear_out.step(ctx, attn_out, x_out)
        ctx.free(attn_out)

