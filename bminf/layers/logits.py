from ..core import Layer, Parameter, Context, Tensor
import numpy as np
from cpm_kernels import kernels as ck

class OutputLogits(Layer):
    def __init__(self, vocab_size, dim_model) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.dim_model = dim_model

        self.weight = Parameter((vocab_size, dim_model), dtype=np.float16)

    def forward(self,
            ctx : Context, 
            x : Tensor,         # (batch, dim_model, seq_q)
            x_out : Tensor      # (batch, seq_q, vocab_size)
        ):
        batch, dim_model, seq_len = x.shape
        assert x_out.shape == (batch, seq_len, self.vocab_size)

        # (batch, dim_model, seq_len)T @ (1#batch, vocab_size, dim_model)T = (batch, seq_len, vocab_size)
        ck.gemm_fp16(
            self.vocab_size, dim_model, seq_len,
            1, batch,
            True, True,
            self.weight.value.ptr, x.ptr,
            x_out.ptr,
            ctx.current_stream
        )

    def step(self, ctx : Context, x : Tensor, x_out : Tensor):
        batch, dim_model = x.shape
        assert x_out.shape == (batch, self.vocab_size)

        ck.gemv_broadcast_mat_fp16(
            batch, self.vocab_size, dim_model,
            self.weight.value.ptr,
            x.ptr,
            x_out.ptr,
            ctx.current_stream
        )
    
    def backward(self, 
            ctx : Context,
            grad_output : Tensor,
            grad : Tensor
        ):
        ## WARNING : logits layer does not accumulate gradients
        
        batch, dim_model, seq_len = grad.shape
        assert grad_output.shape == (batch, seq_len, self.vocab_size)

        ck.gemm_fp16(
            seq_len, self.vocab_size, dim_model,
            batch, 1,
            True, True,
            grad_output.ptr, self.weight.value.ptr,
            grad.ptr,
            ctx.current_stream
        )