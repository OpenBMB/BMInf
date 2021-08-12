from .base import Layer
from ..parameter import Parameter
from ..allocator import Allocator
import cupy
import numpy as np
from ..functions.scale_copy import elementwise_copy

class Embedding(Layer):
    TYPE_F32 = 1
    TYPE_F16 = 2
    TYPE_I8 = 3

    def __init__(self, num_embeddings, embedding_dim, ltype):
        self.embedding_dim = embedding_dim
        if ltype == self.TYPE_F32:
            self.weight = Parameter((num_embeddings, embedding_dim), dtype=cupy.float32)
        elif ltype == self.TYPE_F16:
            self.weight = Parameter((num_embeddings, embedding_dim), dtype=cupy.float16)
        else:
            raise TypeError("Unknown type for %s (%s)" % (self.__class__.__name__, ltype))
        
        self.ltype = ltype

    def forward(self, allocator : Allocator, x):
        if isinstance(x, list):
            x = np.array(x).astype(np.int64)
        
        assert isinstance(x, np.ndarray)

        
        out = allocator.alloc_array( x.shape + (self.embedding_dim,), dtype=self.weight.dtype )
        cupy.take(self.weight.value, x, axis=0, out=out)
        if out.dtype == cupy.float16:
            # convert fp16 to fp32
            fp32_out = allocator.alloc_array( out.shape, dtype=cupy.float32 )
            elementwise_copy(out, out=fp32_out)
            out = fp32_out
            del fp32_out
        return out
    