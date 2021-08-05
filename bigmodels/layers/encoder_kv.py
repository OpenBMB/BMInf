from ..parameter import Parameter
from .base import Layer
import cupy

class EncoderKeyValueProjection(Layer):
    TYPE_F32 = 1
    TYPE_F16 = 2
    TYPE_I8 = 3
    def __init__(self, num_decoder, dim_in, dim_kv, num_heads, ltype):
        if ltype == self.TYPE_F32:
            self.w_project_kv = Parameter((num_decoder * dim_kv * num_heads * 2, dim_in), cupy.float32)
        elif ltype == self.TYPE_F16:
            self.w_project_kv = Parameter((num_decoder * dim_kv * num_heads * 2, dim_in), cupy.float16)
        elif ltype == self.TYPE_I8:
            self.w_project_kv = Parameter((num_decoder * dim_kv * num_heads * 2, dim_in), cupy.int8)
        else:
            raise TypeError("Unknown type for %s (%s)" % (self.__class__.__name__, ltype))
    
    def forward(self, x):
        pass