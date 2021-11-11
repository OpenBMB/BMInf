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
