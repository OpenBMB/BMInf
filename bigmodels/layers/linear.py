import cupy
from .base import Layer
from ..parameter import Parameter
from ..allocator import Allocator
from ..functions.quantization import quantize
from ..functions.gemm import igemm
from ..functions.scale_copy import elementwise_copy_scale

class Linear(Layer):

    def __init__(self, dim_in : int, dim_out : int):
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.weight = Parameter((dim_out, dim_in), cupy.int8)

    def forward(self, allocator : Allocator, x : cupy.ndarray):
        assert x.dtype == cupy.float32
        value = x

        batch_size, dim_model, seq_len = value.shape
        assert dim_model == self.dim_in

        value_i8 = allocator.alloc_array(value.shape, cupy.int8)
        scale = quantize(value, out=value_i8)

        out_i32 = allocator.alloc_array((batch_size, self.dim_out, seq_len), cupy.int32)
        
        # FIXME: igemm Stried Batched cupy
        for i in range(batch_size):
            igemm(value_i8[i], True, self.weight.value, True, out_i32[i])
        
        out_f32 = allocator.alloc_array(out_i32.shape, cupy.float32)
        elementwise_copy_scale(out_i32, scale * self.weight.scale, out_f32)
        return out_f32