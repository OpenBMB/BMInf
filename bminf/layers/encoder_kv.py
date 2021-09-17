from ..parameter import Parameter
from .base import Layer
import cupy
from ..allocator import Allocator
from ..functions.quantization import quantize
from ..functions.gemm import igemm
from ..functions.scale_copy import elementwise_copy_scale

class EncoderKeyValueProjection(Layer):
    def __init__(self, num_decoder, dim_in, dim_kv, num_heads):
        self.num_decoder = num_decoder
        self.dim_kv = dim_kv
        self.dim_in = dim_in
        self.num_heads = num_heads

        self.w_project_kv = Parameter((num_decoder, 2, dim_kv * num_heads, dim_in), cupy.int8)
        self.w_project_kv_scale = Parameter((num_decoder, 2, dim_kv * num_heads, 1), cupy.float16)
    
    def forward(self, allocator : Allocator, x):
        assert x.dtype == cupy.float16

        value = x

        batch_size, dim_model, seq_len = value.shape
        assert dim_model == self.dim_in

        value_i8 = allocator.alloc_array(value.shape, cupy.int8)
        scale = allocator.alloc_array((batch_size, 1, seq_len), dtype=cupy.float16)
        
        # (batch_size, dim_model, seq_len), (batch_size, 1, seq_len)
        quantize(value, value_i8, scale, axis=1)
        out_f16 = allocator.alloc_array((self.num_decoder, 2, batch_size, self.dim_kv * self.num_heads, seq_len), cupy.float16)
        out_i32 = allocator.alloc_array((batch_size, self.dim_kv * self.num_heads, seq_len), cupy.int32)

        for i in range(self.num_decoder):    
            for j in range(2):
                # (batch_size, dim_model, seq_len) @ (1, dim_kv * num_heads, dim_model)
                igemm(allocator, value_i8, False, self.w_project_kv.value[i, j][cupy.newaxis], False, out_i32)
            
                elementwise_copy_scale(
                    out_i32, 
                    scale, 
                    self.w_project_kv_scale.value[i, j], 
                    out_f16[i, j]
                )
        del out_i32
        assert out_f16._c_contiguous

        reshaped_out_f16 = cupy.ndarray(
            (self.num_decoder, 2, batch_size, self.num_heads, self.dim_kv, seq_len),
            dtype=out_f16.dtype,
            memptr=out_f16.data
        )
        return reshaped_out_f16