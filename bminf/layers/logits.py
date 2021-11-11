from ..core import Layer, Parameter, Context, Tensor
import numpy as np
from cpm_kernels import kernels as ck

class OutputLogits(Layer):
    def __init__(self, vocab_size, dim_model) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.dim_model = dim_model

        self.weight = Parameter((vocab_size, dim_model), dtype=np.float16)

    def forward(self, ctx : Context, x : Tensor, x_out : Tensor):
        batch, dim_model, seq_len = x.shape
        assert x_out.shape == (batch, seq_len, self.vocab_size)

        logits = ctx.allocate((batch, self.vocab_size, seq_len), dtype=np.float16)

        # (1#batch, vocab_size, dim_model)  @ (batch, dim_model, seq_len)
        ck.gemm_fp16(
            seq_len, dim_model, self.vocab_size,
            batch, 1,
            False, False,
            x.ptr, self.weight.value.ptr,
            logits.ptr,
            ctx.current_stream
        )

        ck.transpose(
            batch, self.vocab_size, seq_len,
            logits.ptr,
            x_out.ptr,
            ctx.current_stream
        )
        ctx.free(logits)

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