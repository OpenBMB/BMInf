from .base import Layer
from ..parameter import Parameter
import cupy

class LayerNorm(Layer):
    TYPE_F32 = 1
    TYPE_F16 = 2
    TYPE_I8 = 3

    def __init__(self, dim_in, ltype, eps=1e-6):
        self.eps = eps
        if ltype == self.TYPE_F32:
            self.weight = Parameter((dim_in,), dtype=cupy.float32)
        elif ltype == self.TYPE_F16:
            self.weight = Parameter((dim_in,), dtype=cupy.float16)
        elif ltype == self.TYPE_I8:
            self.weight = Parameter((dim_in,), dtype=cupy.int8)
        else:
            raise TypeError("Unknown type for %s (%s)" % (self.__class__.__name__, ltype))
        
        self.ltype = ltype
        


    def forward(self, x):
        # call function linear

        pass