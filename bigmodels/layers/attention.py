from ..parameter import Parameter
from .base import Layer
import cupy

class SelfAttention(Layer):
    TYPE_F32 = 1
    TYPE_F16 = 2
    TYPE_I8 = 3
    def __init__(self, dim_in, dim_qkv, num_heads, ltype):
        if ltype == self.TYPE_F32:
            self.w_project_kqv = Parameter((dim_qkv * num_heads * 3, dim_in), cupy.float32)
            self.w_out = Parameter((dim_qkv * num_heads, dim_in), cupy.float32)
        elif ltype == self.TYPE_F16:
            self.w_project_kqv = Parameter((dim_qkv * num_heads * 3, dim_in), cupy.float16)
            self.w_out = Parameter((dim_qkv * num_heads, dim_in), cupy.float16)
        elif ltype == self.TYPE_I8:
            self.w_project_kqv = Parameter((dim_qkv * num_heads * 3, dim_in), cupy.int8)
            self.w_out = Parameter((dim_qkv * num_heads, dim_in), cupy.int8)
        else:
            raise TypeError("Unknown type for %s (%s)" % (self.__class__.__name__, ltype))
    
    def forward(self, x):
        pass

class CrossAttention(Layer):
    TYPE_F32 = 1
    TYPE_F16 = 2
    TYPE_I8 = 3

    def __init__(self, dim_in, dim_qkv, num_heads, ltype = 1):
        if ltype == self.TYPE_F32:
            self.w_project_kqv = Parameter((dim_qkv * num_heads, dim_in), cupy.float32)
            self.w_out = Parameter((dim_qkv * num_heads, dim_in), cupy.float32)
        elif ltype == self.TYPE_F16:
            self.w_project_kqv = Parameter((dim_in, dim_qkv * num_heads), cupy.float16)
            self.w_out = Parameter((dim_qkv * num_heads, dim_in), cupy.float16)
        elif ltype == self.TYPE_I8:
            self.w_project_kqv = Parameter((dim_in, dim_qkv * num_heads), cupy.int8)
            self.w_out = Parameter((dim_qkv * num_heads, dim_in), cupy.int8)
        else:
            raise TypeError("Unknown type for %s (%s)" % (self.__class__.__name__, ltype))
    