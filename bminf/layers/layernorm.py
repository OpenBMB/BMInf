from typing import Optional
from ..core import Layer, Context, Tensor, Parameter
import numpy as np
from cpm_kernels import kernels as ck

class Layernorm(Layer):
    def __init__(self, dim_model : int, eps : float = 1e-5, bias=False):
        super().__init__()

        self.dim_model = dim_model
        self.eps = eps
        self.weight = Parameter((dim_model,), np.float16)
        if bias:
            self.bias = Parameter((dim_model,), np.float16)
        else:
            self.bias = None
    
    def forward(self, ctx : Context, x : Tensor, x_out : Tensor):
        batch, hidden_size, seq_len = x.shape
        assert hidden_size == self.dim_model
        assert x.shape == x_out.shape

        if x.ptr != x_out.ptr:
            ck.layernorm_forward(
                batch, hidden_size, seq_len,
                x.ptr, x_out.ptr,
                self.eps, self.bias is not None,
                ctx.current_stream
            )
        else:
            # in-place normalization
            ck.layernorm_inplace_forward(
                batch, hidden_size, seq_len,
                x.ptr,
                self.eps, self.bias is not None,
                ctx.current_stream
            )
        
        if self.bias is None:
            ck.arith_ln_mul(
                batch, hidden_size, seq_len,
                x_out.ptr, self.weight.value.ptr, x_out.ptr,
                ctx.current_stream
            )
        else:
            ck.arith_ln_mul_add(
                batch, hidden_size, seq_len,
                x_out.ptr, self.weight.value.ptr, self.bias.value.ptr, x_out.ptr,
                ctx.current_stream
            )
    
    def step(self, ctx : Context, x : Tensor, x_out : Tensor):
        batch, hidden_size = x.shape
        assert hidden_size == self.dim_model
        assert x.shape == x_out.shape
        assert x.ptr != x_out.ptr

        ck.layernorm_step(
            batch, hidden_size,
            x.ptr,
            x_out.ptr,
            self.eps,
            self.bias is not None,
            ctx.current_stream
        )

        if self.bias is None:
            ck.arith_batch_mul(
                batch, hidden_size,
                x_out.ptr,
                self.weight.value.ptr,
                x_out.ptr,
                ctx.current_stream
            )
        else:
            ck.arith_batch_mul_add(
                batch, hidden_size,
                x_out.ptr,
                self.weight.value.ptr,
                self.bias.value.ptr,
                x_out.ptr,
                ctx.current_stream
            )