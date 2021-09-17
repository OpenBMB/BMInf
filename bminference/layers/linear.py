import cupy
from .base import Layer
from ..parameter import Parameter
from ..allocator import Allocator
from ..functions.quantization import quantize
from ..functions.gemm import igemm
from ..functions.scale_copy import elementwise_copy_scale

class Linear(Layer):

    def __init__(self, dim_in : int, dim_out : int, bias=False):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.bias = bias

        self.weight = Parameter((dim_out, dim_in), cupy.int8)
        self.weight_scale = Parameter((dim_out, 1), cupy.float16)
        if self.bias:
            self.weight_bias = Parameter((dim_out,), cupy.float16)
        

    def forward(self, allocator : Allocator, x : cupy.ndarray):
        assert x.dtype == cupy.float16
        value = x

        batch_size, dim_in, seq_len = value.shape
        assert dim_in == self.dim_in

        value_i8 = allocator.alloc_array(value.shape, cupy.int8)
        scale = allocator.alloc_array((batch_size, 1, seq_len), dtype=cupy.float16)
        
        # (batch_size, dim_in, seq_len), (batch_size, 1, seq_len)
        quantize(value, value_i8, scale, axis=1)

        out_i32 = allocator.alloc_array((batch_size, self.dim_out, seq_len), cupy.int32)
        
        igemm(allocator, value_i8, False, self.weight.value[cupy.newaxis], False, out_i32)
        
        out_f16 = allocator.alloc_array(out_i32.shape, cupy.float16)
        elementwise_copy_scale(out_i32, scale, self.weight_scale.value, out_f16)

        if self.bias:
            out_f16 += self.weight_bias.value[cupy.newaxis, :, cupy.newaxis]   # (1#batch_size, dim_out, 1#seq_len)

        return out_f16