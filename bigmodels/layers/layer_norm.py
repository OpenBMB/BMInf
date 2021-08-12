from .base import Layer
from ..parameter import Parameter
import cupy
from ..allocator import Allocator

l2norm_kernel = cupy.ReductionKernel(
    'T x',
    'T y',
    'x * x',
    'a + b',
    'y = sqrt(_type_reduce(_in_ind.size() / _out_ind.size()) / a)',
    '0.000001',
    'bms_l2norm'
)

class LayerNorm(Layer):
    def __init__(self, dim_in):
        self.dim_model = dim_in
        self.weight = Parameter((dim_in,), dtype=cupy.float32)
        

    def forward(self, allocator : Allocator, x : cupy.ndarray, inplace = True):
        # forward inplace        
        value = x

        batch_size, dim_model, seq_len = value.shape
        assert dim_model == self.dim_model
        assert value.dtype == cupy.float32 or value.dtype == cupy.float16

        dqv = value

        out = allocator.alloc_array((batch_size, 1, seq_len), dqv.dtype)
        l2norm_kernel(dqv, axis=1, keepdims=True, out=out)

        if inplace and value.dtype == cupy.float32 :
            dqv *= out
        else:
            nw_dqv = allocator.alloc_array(dqv.shape, dtype=cupy.float32)
            cupy.multiply(dqv, out, out=nw_dqv)
            dqv = nw_dqv
            del nw_dqv

        del out
        
        dqv *= self.weight.value[cupy.newaxis,:,cupy.newaxis]

        return dqv