from .base import Layer
from ..parameter import Parameter
from ..allocator import Allocator
import cupy
import numpy as np
from ..functions.scale_copy import elementwise_copy

class Embedding(Layer):

    def __init__(self, num_embeddings, embedding_dim):
        self.embedding_dim = embedding_dim
        self.weight = Parameter((num_embeddings, embedding_dim), dtype=cupy.float16)


    def forward(self, allocator : Allocator, x):
        if isinstance(x, list):
            x = np.array(x).astype(np.int64)
        
        assert isinstance(x, np.ndarray)

        
        out = allocator.alloc_array( x.shape + (self.embedding_dim,), dtype=self.weight.dtype )
        cupy.take(self.weight.value, x, axis=0, out=out)

        out_fp16 = allocator.alloc_array(out.shape, dtype=cupy.float16)
        elementwise_copy(out, out_fp16)
        del out

        return out_fp16
    