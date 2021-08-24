
from .base import Layer
from ..parameter import Parameter
import cupy
from ..allocator import Allocator
from ..functions.scale_copy import elementwise_copy

l2norm_kernel = cupy.ReductionKernel(
    'T x',
    'T y',
    'x * x',
    'a + b',
    'y = rsqrt( a / _type_reduce(_in_ind.size() / _out_ind.size()) + 0.000001 ) ',
    '0',
    'bms_l2norm'
)


class LayerNorm(Layer):
    def __init__(self, dim_in):
        self.dim_model = dim_in
        self.weight = Parameter((dim_in,), dtype=cupy.float16)
        

    def forward(self, allocator : Allocator, x : cupy.ndarray):
        # forward inplace        
        value = x

        batch_size, dim_model, seq_len = value.shape
        assert dim_model == self.dim_model
        assert value.dtype == cupy.float16

        # dqv = value
        dqv = allocator.alloc_array(x.shape, cupy.float32)
        elementwise_copy(value, dqv)
        
        out = allocator.alloc_array((batch_size, 1, seq_len), cupy.float32)

        
        l2norm_kernel(dqv, axis=1, keepdims=True, out=out)

        dqv *= out
        nw_dqv = allocator.alloc_array(dqv.shape, dtype=cupy.float16)
        cupy.multiply(dqv, self.weight.value[cupy.newaxis,:,cupy.newaxis], out=nw_dqv)

        return nw_dqv