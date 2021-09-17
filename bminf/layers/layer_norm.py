
from .base import Layer
from ..parameter import Parameter
import cupy
from ..allocator import Allocator
from ..functions.scale_copy import elementwise_copy

l2norm_kernel = cupy.ReductionKernel(
    'T x, T eps',
    'T y',
    'x * x',
    'a + b',
    'y = rsqrt( a / _type_reduce(_in_ind.size() / _out_ind.size()) + eps ) ',
    '0',
    'bms_l2norm'
)

ggg = cupy.ReductionKernel(
    'T x',
    'T y',
    'x * x',
    'a + b',
    'y = a / _type_reduce(_in_ind.size() / _out_ind.size()) ',
    '0',
    'ggg'
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

        
        l2norm_kernel(dqv, 1e-6, axis=1, keepdims=True, out=out)

        dqv *= out
        nw_dqv = allocator.alloc_array(dqv.shape, dtype=cupy.float16)
        cupy.multiply(dqv, self.weight.value[cupy.newaxis,:,cupy.newaxis], out=nw_dqv)

        return nw_dqv

class GPTLayerNorm(Layer):
    def __init__(self, dim_in):
        self.dim_model = dim_in
        self.weight = Parameter((dim_in,), dtype=cupy.float16)
        self.bias = Parameter((dim_in,), dtype=cupy.float16)
        

    def forward(self, allocator : Allocator, x : cupy.ndarray):
        # forward inplace        

        batch_size, dim_model, seq_len = x.shape
        assert dim_model == self.dim_model
        assert x.dtype == cupy.float16

        x_fp32 = allocator.alloc_array(x.shape, cupy.float32)
        elementwise_copy(x, x_fp32)
        x_mean = allocator.alloc_array((batch_size, 1, seq_len), cupy.float32)
        x_var = allocator.alloc_array((batch_size, 1, seq_len), cupy.float32)
        
        x_mean = cupy.mean(x_fp32, axis=1, out=x_mean, keepdims=True)
        l2norm_kernel(x_fp32 - x_mean, 1e-5, axis=1, keepdims=True, out=x_var)

        x_fp32 = ((x_fp32 - x_mean) * x_var) * self.weight.value[cupy.newaxis, :, cupy.newaxis] + self.bias.value[cupy.newaxis, :, cupy.newaxis]
        x_fp16 = allocator.alloc_array(x.shape, cupy.float16)
        elementwise_copy(x_fp32, x_fp16)
        return x_fp16