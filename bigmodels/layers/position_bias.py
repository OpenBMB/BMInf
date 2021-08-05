from .base import Layer
from ..parameter import Parameter
from .embedding import Embedding
import cupy

class PositionBias(Layer):
    TYPE_F32 = 1
    TYPE_F16 = 2
    TYPE_I8 = 3
    def __init__(self, num_buckets, num_heads, ltype):
        
        if ltype == self.TYPE_F32:
            self.embedding = Embedding(num_buckets, num_heads, Embedding.TYPE_F32)
        elif ltype == self.TYPE_F16:
            self.embedding = Embedding(num_buckets, num_heads, Embedding.TYPE_F16)
        elif ltype == self.TYPE_I8:
            self.embedding = Embedding(num_buckets, num_heads, Embedding.TYPE_I8)
        else:
            raise TypeError("Unknown type for %s (%s)" % (self.__class__.__name__, ltype))