import cupy
from .base import Layer
from ..parameter import Parameter

class Linear(Layer):
    TYPE_F32_F32 = 1
    TYPE_F16_F16 = 2
    TYPE_I8_I32 = 3
    TYPE_I8_I8 = 4
    TYPE_I8_F32 = 5

    def __init__(self, dim_in : int, dim_out : int, ltype : int, bias : bool = True):

        if ltype == self.TYPE_F32_F32:
            self.weight = Parameter((dim_out, dim_in), cupy.float32)
        elif ltype == self.TYPE_F16_F16:
            self.weight = Parameter((dim_out, dim_in), cupy.float16)
        elif ltype in [self.TYPE_I8_I32, self.TYPE_I8_I8, self.TYPE_I8_F32]:
            self.weight = Parameter((dim_out, dim_in), cupy.int8)
        else:
            raise TypeError("Unknown type for %s (%s)" % (self.__class__.__name__, ltype))
        
        if bias:
            if ltype == self.TYPE_F32_F32:
                self.bias = Parameter((dim_out,), cupy.float32)
            elif ltype == self.TYPE_F16_F16:
                self.bias = Parameter((dim_out,), cupy.float16)
            elif ltype == self.TYPE_I8_I32 or ltype == self.TYPE_I8_I8:
                self.bias = Parameter((dim_out,), cupy.int32)
            elif ltype == self.TYPE_I8_F32:
                self.bias = Parameter((dim_out,), cupy.float32)

        self.ltype = ltype

    def forward(self, x):
        # call function linear

        pass