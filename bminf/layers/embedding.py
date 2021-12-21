from typing import Optional
from ..core import Layer, Context, Tensor, Parameter
import numpy as np
from cpm_kernels import kernels as ck

class Embedding(Layer):
    def __init__(self, vocab_size : int, dim_model : int):
        super().__init__()

        self.dim_model = dim_model
        self.vocab_size = vocab_size

        self.weight = Parameter(
            (vocab_size, dim_model),
            np.float16
        )
    
    def embedding_forward(self,
            ctx : Context, 
            ids : Tensor, 
            x_out : Tensor
        ):
        assert ids.dtype == np.int32
        batch, seq_len = ids.shape
        assert x_out.shape == (batch, self.dim_model, seq_len)

        ck.embedding_forward(
            batch, self.dim_model, seq_len,
            ids.ptr,
            self.weight.value.ptr,
            x_out.ptr,
            ctx.current_stream
        )

    def embedding_step(self,
            ctx : Context, 
            ids : Tensor, 
            x_out : Tensor
        ):
        assert ids.dtype == np.int32
        batch = ids.shape[0]
        assert x_out.shape == (batch, self.dim_model)
        
        ck.embedding_step(
            batch, self.dim_model,
            ids.ptr,
            self.weight.value.ptr,
            x_out.ptr,
            ctx.current_stream
        )
    
    def projection_forward(self,
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
    
    def projection_step(self, ctx : Context, x : Tensor, x_out : Tensor):
        batch, dim_model = x.shape
        assert x_out.shape == (batch, self.vocab_size)

        ck.gemv_broadcast_mat_fp16_light(
            batch, self.vocab_size, dim_model,
            self.weight.value.ptr,
            x.ptr,
            x_out.ptr,
            ctx.current_stream
        )

    def projection_backward(self, 
            ctx : Context,
            grad_output : Tensor,   # (batch, seq_q, vocab_size)
            grad : Tensor           # (batch, dim_model, seq_q)
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