from typing import Optional
from ..core import Layer, Context, Tensor, Parameter
import numpy as np
from cpm_kernels import kernels as ck

class Embedding(Layer):
    def __init__(self, vocab_size : int, embedding_size : int):
        super().__init__()

        self.embedding_size = embedding_size
        self.weight = Parameter(
            (vocab_size, embedding_size),
            np.float16
        )
    
    def forward(self,
            ctx : Context, 
            ids : Tensor, 
            x_out : Tensor
        ):
        assert ids.dtype == np.int32
        batch, seq_len = ids.shape
        assert x_out.shape == (batch, self.embedding_size, seq_len)

        ck.embedding_forward(
            batch, self.embedding_size, seq_len,
            ids.ptr,
            self.weight.value.ptr,
            x_out.ptr,
            ctx.current_stream
        )

    def step(self,
            ctx : Context, 
            ids : Tensor, 
            x_out : Tensor
        ):
        assert ids.dtype == np.int32
        batch = ids.shape[0]
        assert x_out.shape == (batch, self.embedding_size)
        
        ck.embedding_step(
            batch, self.embedding_size,
            ids.ptr,
            self.weight.value.ptr,
            x_out.ptr,
            ctx.current_stream
        )
