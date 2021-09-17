from .base import Layer
import cupy
from .linear import Linear
from ..allocator import Allocator
from ..functions.gelu import gelu_kernel, gelu_raw_kernel

class DenseGeluDense(Layer):
    def __init__(self, dim_model, dim_ff):
        self.wi_0 = Linear(dim_model, dim_ff)
        self.wi_1 = Linear(dim_model, dim_ff)
        self.wo = Linear(dim_ff, dim_model)

    def forward(self, allocator : Allocator, x : cupy.ndarray) -> cupy.ndarray:
        # call function linear

        w0 = self.wi_0.forward(allocator, x)
        w1 = self.wi_1.forward(allocator, x)

        assert w0.dtype == cupy.float16
        assert w1.dtype == cupy.float16

        gelu_kernel(w0, w1, out=w0)

        del w1

        wout = self.wo.forward(allocator, w0)
        assert wout.dtype == cupy.float16

        return wout

class GPTDenseGeluDense(Layer):
    def __init__(self, dim_model, dim_ff):
        self.wi = Linear(dim_model, dim_ff, bias=True)
        self.wo = Linear(dim_ff, dim_model, bias=True)

    def forward(self, allocator : Allocator, x : cupy.ndarray) -> cupy.ndarray:
        # call function linear

        w0 = self.wi.forward(allocator, x)

        assert w0.dtype == cupy.float16

        gelu_raw_kernel(w0, out=w0)

        wout = self.wo.forward(allocator, w0)
        assert wout.dtype == cupy.float16

        return wout