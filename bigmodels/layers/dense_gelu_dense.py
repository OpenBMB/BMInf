from .base import Layer
from ..parameter import Parameter
from .linear import Linear
from .gelu import GeLU

class DenseGeluDense(Layer):
    TYPE_F32 = 1
    TYPE_F16 = 2
    TYPE_I8 = 3
    def __init__(self, dim_model, dim_ff, ltype):
        if ltype == self.TYPE_F32:
            self.wi_0 = Linear(dim_model, dim_ff, Linear.TYPE_F32_F32, bias=False)
            self.wi_1 = Linear(dim_model, dim_ff, Linear.TYPE_F32_F32, bias=False)
            self.wo = Linear(dim_ff, dim_model, Linear.TYPE_F32_F32, bias=False)
            self.gelu = GeLU(GeLU.TYPE_FLOAT)
        elif ltype == self.TYPE_F16:
            self.wi_0 = Linear(dim_model, dim_ff, Linear.TYPE_F16_F16, bias=False)
            self.wi_1 = Linear(dim_model, dim_ff, Linear.TYPE_F16_F16, bias=False)
            self.wo = Linear(dim_ff, dim_model, Linear.TYPE_F16_F16, bias=False)
            self.gelu = GeLU(GeLU.TYPE_FLOAT)
        elif ltype == self.TYPE_I8:
            self.wi_0 = Linear(dim_model, dim_ff, Linear.TYPE_I8_I32, bias=False)
            self.wi_1 = Linear(dim_model, dim_ff, Linear.TYPE_I8_I32, bias=False)
            self.wo = Linear(dim_ff, dim_model, Linear.TYPE_I8_I8, bias=False)
            self.gelu = GeLU(GeLU.TYPE_I32)
        else:
            raise TypeError("Unknown type for %s (%s)" % (self.__class__.__name__, ltype))

    def forward(self, x):
        # call function linear

        pass