from typing import Optional
from ..core import Layer, Context, Tensor
import numpy as np
from cpm_kernels import kernels as ck
from .linear import Linear

class FeedForward(Layer):
    def __init__(self, dim_model, dim_ff, bias=False, gated=True):
        super().__init__()

        self.dim_model = dim_model
        self.dim_ff = dim_ff

        if gated:
            self.linear_gated = Linear(dim_model, dim_ff, bias=bias)
        else:
            self.linear_gated = None
        
        self.linear_in = Linear(dim_model, dim_ff, bias=bias)
        self.linear_out = Linear(dim_ff, dim_model, bias=bias)
    
    def forward(self, ctx : Context, x : Tensor, x_out : Tensor):
        batch, hidden_size, seq_len = x.shape
        assert hidden_size == self.dim_model
        assert x.shape == x_out.shape
        
        x_0 = ctx.allocate((batch, self.dim_ff, seq_len), dtype=np.float16)
        self.linear_in.forward(ctx, x, x_0)

        ck.gelu_forward(
            batch, self.dim_ff * seq_len,
            x_0.ptr, x_0.ptr,
            ctx.current_stream
        )

        if self.linear_gated is not None:
            x_1 = ctx.allocate((batch, self.dim_ff, seq_len), dtype=np.float16)
            self.linear_gated.forward(ctx, x, x_1)
            ck.arith_element_mul(
                batch, self.dim_ff * seq_len,
                x_0.ptr, x_1.ptr, x_0.ptr,
                ctx.current_stream
            )
            ctx.free(x_1)
        self.linear_out.forward(ctx, x_0, x_out)
        ctx.free(x_0)
    
    def step(self, ctx : Context, x : Tensor, x_out : Tensor):
        batch, hidden_size = x.shape
        assert hidden_size == self.dim_model
        assert x.shape == x_out.shape

        x_0 = ctx.allocate((batch, self.dim_ff), dtype=np.float16)
        self.linear_in.step(ctx, x, x_0)
        
        ck.gelu_forward(
            batch, self.dim_ff,
            x_0.ptr, x_0.ptr,
            ctx.current_stream
        )

        if self.linear_gated is not None:
            x_1 = ctx.allocate((batch, self.dim_ff), dtype=np.float16)
            self.linear_gated.step(ctx, x, x_1)
            ck.arith_element_mul(
                batch, self.dim_ff,
                x_0.ptr, x_1.ptr, x_0.ptr,
                ctx.current_stream
            )
            ctx.free(x_1)
        self.linear_out.step(ctx, x_0, x_out)
        ctx.free(x_0)
    
    def backward(self,
            ctx : Context, 
            x : Tensor,             # (batch, dim_model, seq_q)
            grad_output : Tensor,   # (batch, dim_model, seq_q)
            grad : Tensor           # (batch, dim_model, seq_q)
        ):
        batch, hidden_size, seq_len = x.shape
        assert hidden_size == self.dim_model

        x_0 = ctx.allocate((batch, self.dim_ff, seq_len), dtype=np.float16)
        self.linear_in.forward(ctx, x, x_0)

        grad_x_out = ctx.allocate((batch, self.dim_ff, seq_len), dtype=np.float16)
        self.linear_out.backward(ctx, grad_output, grad_x_out)

        if self.linear_gated is not None:
            x_0_gelu = ctx.allocate((batch, self.dim_ff, seq_len), dtype=np.float16)
            ck.gelu_forward(
                batch, self.dim_ff * seq_len,
                x_0.ptr, x_0_gelu.ptr,
                ctx.current_stream
            )
            x_1 = ctx.allocate((batch, self.dim_ff, seq_len), dtype=np.float16)
            self.linear_gated.forward(ctx, x, x_1)
            
            # start backward
            grad_x_1 = ctx.allocate((batch, self.dim_ff, seq_len), dtype=np.float16)
            ck.arith_element_mul(
                batch, self.dim_ff * seq_len,
                grad_x_out.ptr, x_0_gelu.ptr, grad_x_1.ptr,
                ctx.current_stream
            )
            ctx.free(x_0_gelu)
            tmp_grad = ctx.allocate((batch, hidden_size, seq_len), dtype=np.float16)
            self.linear_gated.backward(ctx, grad_x_1, tmp_grad)
            ctx.free(grad_x_1)
            ck.arith_element_add(
                batch, hidden_size * seq_len,
                grad.ptr, tmp_grad.ptr, grad.ptr,
                ctx.current_stream
            )
            ctx.free(tmp_grad)

            grad_x_0_gelu = ctx.allocate((batch, self.dim_ff, seq_len), dtype=np.float16)
            ck.arith_element_mul(
                batch, self.dim_ff * seq_len,
                grad_x_out.ptr, x_1.ptr, grad_x_0_gelu.ptr,
                ctx.current_stream
            )
            ctx.free(x_1)
            ctx.free(grad_x_out)
            grad_x_0 = ctx.allocate((batch, self.dim_ff, seq_len), dtype=np.float16)
            ck.gelu_backward(
                batch, self.dim_ff * seq_len,
                grad_x_0_gelu.ptr,
                x_0.ptr, grad_x_0.ptr,
                ctx.current_stream
            )
            ctx.free(x_0)
            ctx.free(grad_x_0_gelu)
            tmp_grad = ctx.allocate((batch, hidden_size, seq_len), dtype=np.float16)
            self.linear_in.backward(ctx, grad_x_0, tmp_grad)
            ctx.free(grad_x_0)
            ck.arith_element_add(
                batch, hidden_size * seq_len,
                grad.ptr, tmp_grad.ptr, grad.ptr,
                ctx.current_stream
            )
            ctx.free(tmp_grad)
        else:
            # start backward
            grad_x_0 = ctx.allocate((batch, self.dim_ff, seq_len), dtype=np.float16)
            ck.gelu_backward(
                batch, self.dim_ff * seq_len,
                grad_x_out.ptr,
                x_0.ptr, grad_x_0.ptr,
                ctx.current_stream
            )
            ctx.free(x_0)
            ctx.free(grad_x_out)
            tmp_grad = ctx.allocate((batch, hidden_size, seq_len), dtype=np.float16)
            self.linear_in.backward(ctx, grad_x_0, tmp_grad)
            ctx.free(grad_x_0)
            ck.arith_element_add(
                batch, hidden_size * seq_len,
                grad.ptr, tmp_grad.ptr, grad.ptr,
                ctx.current_stream
            )
            ctx.free(tmp_grad)