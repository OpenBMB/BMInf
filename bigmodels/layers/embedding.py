from .base import Layer
from ..parameter import Parameter
import cupy

class Embedding(Layer):
    TYPE_F32 = 1
    TYPE_F16 = 2
    TYPE_I8 = 3

    def __init__(self, num_embeddings, embedding_dim, ltype):
        if ltype == self.TYPE_F32:
            self.weight = Parameter((num_embeddings, embedding_dim), dtype=cupy.float32)
        elif ltype == self.TYPE_F16:
            self.weight = Parameter((num_embeddings, embedding_dim), dtype=cupy.float16)
        elif ltype == self.TYPE_I8:
            self.weight = Parameter((num_embeddings, embedding_dim), dtype=cupy.int8)
        else:
            raise TypeError("Unknown type for %s (%s)" % (self.__class__.__name__, ltype))
        
        self.ltype = ltype

    def forward(self, x):
        # call function linear

        pass
    