from ..core import Layer, Parameter, Context, Tensor
import numpy as np
from cpm_kernels import kernels as ck


class Linear(Layer):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter((out_features, in_features), dtype=np.int8)
        self.scale = Parameter((out_features,), dtype=np.float16)

        if bias:
            self.bias = Parameter((out_features,), dtype=np.float16)
        else:
            self.bias = None
    
    def forward(self, ctx : Context, x : Tensor, x_out : Tensor):
        batch, hidden_size, seq_len = x.shape
        assert x.dtype == np.float16
        assert hidden_size == self.in_features
        assert x_out.shape == (batch, self.out_features, seq_len) and x_out.dtype == np.float16
        
        quantized_x = ctx.allocate((batch, hidden_size, seq_len), dtype=np.int8)
        scale_x = ctx.allocate((batch, seq_len), dtype=np.float16)

        ck.gemm_calc_scale_transpose(
            batch, hidden_size, seq_len,
            x.ptr, scale_x.ptr,
            ctx.current_stream
        )

        ck.gemm_round_transpose(
            batch, hidden_size, seq_len,
            x.ptr, scale_x.ptr,
            quantized_x.ptr,
            ctx.current_stream
        )

        out_i32 = ctx.allocate((batch, self.out_features, seq_len), dtype=np.int32)
        ck.gemm_int8(
            seq_len, hidden_size, self.out_features,
            batch, 1,
            False, False,
            quantized_x.ptr, self.weight.value.ptr,
            out_i32.ptr,
            ctx.current_stream
        )

        ck.gemm_scale(
            batch, self.out_features, seq_len,
            out_i32.ptr,
            self.scale.value.ptr,
            scale_x.ptr,
            x_out.ptr,
            True,
            False,
            ctx.current_stream
        )
        ctx.free(out_i32)
        ctx.free(scale_x)
        ctx.free(quantized_x)

        if self.bias:
            ck.arith_ln_add(
                batch, self.out_features, seq_len,
                x_out.ptr,
                self.bias.value.ptr,
                x_out.ptr,
                ctx.current_stream
            )
    
    def step(self, ctx : Context, x : Tensor, x_out : Tensor):
        batch, hidden_size = x.shape
        assert x.shape == (batch, self.in_features) and x.dtype == np.float16
        assert x_out.shape == (batch, self.out_features) and x_out.dtype == np.float16
        ck.gemv_broadcast_mat_int8(
            batch, self.out_features, hidden_size,
            self.scale.value.ptr,
            self.weight.value.ptr,
            x.ptr,
            x_out.ptr,
            ctx.current_stream
        )

    def backward(self, ctx : Context, grad_output : Tensor, grad : Tensor):
        ## WARNING: backward function of Linear layer does not accumulate gradients
        batch, hidden_size, seq_len = grad_output.shape
        assert hidden_size == self.out_features and grad_output.dtype == np.float16
        assert grad.shape == (batch, self.in_features, seq_len) and grad.dtype == np.float16

        quant_G = ctx.allocate((batch, self.out_features, seq_len), dtype=np.int8)
        scale_G = ctx.allocate((batch, seq_len), dtype=np.float16)
        ck.gemm_backward_scale_round(
            batch, self.out_features, seq_len,
            grad_output.ptr,
            self.scale.value.ptr,
            quant_G.ptr,
            scale_G.ptr,
            True,
            ctx.current_stream
        )
        grad_i32 = ctx.allocate((batch, self.in_features, seq_len), dtype=np.int32)
        ck.gemm_int8(
            seq_len, self.out_features, self.in_features,
            1, batch,
            False, True,
            quant_G.ptr, self.weight.value.ptr, 
            grad_i32.ptr,
            ctx.current_stream
        )
        ctx.free(quant_G)
        ck.gemm_scale_y(
            batch, self.in_features, seq_len,
            grad_i32.ptr,
            scale_G.ptr,
            grad.ptr,
            ctx.current_stream
        )
        ctx.free(scale_G)
        ctx.free(grad_i32)
        