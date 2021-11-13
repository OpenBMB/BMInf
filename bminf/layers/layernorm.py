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
        
        if x.ptr != x_out.ptr:
            ck.layernorm_step(
                batch, hidden_size,
                x.ptr,
                x_out.ptr,
                self.eps,
                self.bias is not None,
                ctx.current_stream
            )
        else:
            ck.layernorm_step_inplace(
                batch, hidden_size,
                x.ptr,
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
    
    def backward(self, 
            ctx : Context, 
            x : Tensor,             # (batch, dim_model, seq_q)
            grad_output : Tensor,   # (batch, dim_model, seq_q)
            grad : Tensor           # (batch, dim_model, seq_q)
        ):
        batch, dim_model, seq_len = x.shape
        assert dim_model == self.dim_model
        assert grad_output.shape == grad.shape
        assert grad.shape == x.shape

        grad_norm = ctx.allocate((batch, dim_model, seq_len), np.float16)
        ck.arith_ln_mul(
            batch, dim_model, seq_len,
            grad_output.ptr,
            self.weight.value.ptr,
            grad_norm.ptr,
            ctx.current_stream
        )
        if self.bias is not None:
            mean = ctx.allocate((batch, seq_len), np.float16)
            var = ctx.allocate((batch, seq_len), np.float16)
            x_out = ctx.allocate((batch, dim_model, seq_len), np.float16)
            ck.layernorm_forward_mv(
                batch, dim_model, seq_len,
                x.ptr,
                x_out.ptr,
                mean.ptr, var.ptr,
                self.eps,
                ctx.current_stream
            )
            ck.layernorm_backward_mv(
                batch, dim_model, seq_len,
                x.ptr,
                grad_norm.ptr,
                mean.ptr, var.ptr,
                x_out.ptr,              # reuse x_out to store gradient
                ctx.current_stream
            )
            ctx.free(mean)
            ctx.free(var)
            ck.arith_element_add(
                batch, dim_model * seq_len,
                grad.ptr, x_out.ptr,
                grad.ptr,
                ctx.current_stream
            )
            ctx.free(x_out)
        else:
            var = ctx.allocate((batch, seq_len), np.float16)
            x_out = ctx.allocate((batch, dim_model, seq_len), np.float16)
            ck.layernorm_forward_v(
                batch, dim_model, seq_len,
                x.ptr, x_out.ptr, var.ptr,
                self.eps,
                ctx.current_stream
            )
            ck.layernorm_backward_v(
                batch, dim_model, seq_len,
                x.ptr, grad_norm.ptr, var.ptr,
                x_out.ptr,              # reuse x_out to store gradient
                ctx.current_stream
            )
            ctx.free(var)
            ck.arith_element_add(
                batch, dim_model * seq_len,
                grad.ptr, x_out.ptr,
                grad.ptr,
                ctx.current_stream
            )
            ctx.free(x_out)
        ctx.free(grad_norm)
