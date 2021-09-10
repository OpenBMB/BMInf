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

        self.w_project_kv = Parameter((num_decoder, dim_kv * num_heads * 2, dim_in), cupy.int8)
        self.w_project_kv_scale = Parameter((num_decoder, dim_kv * num_heads * 2, 1), cupy.float16)
    
    def forward(self, allocator : Allocator, x):
        assert x.dtype == cupy.float16

        value = x

        batch_size, dim_model, seq_len = value.shape
        assert dim_model == self.dim_in

        value_i8 = allocator.alloc_array(value.shape, cupy.int8)
        scale = allocator.alloc_array((batch_size, 1, seq_len), dtype=cupy.float16)
        
        # (batch_size, dim_model, seq_len), (batch_size, 1, seq_len)
        quantize(value, value_i8, scale, axis=1)

        
        out_f16 = allocator.alloc_array((batch_size, self.num_decoder, self.dim_kv * self.num_heads * 2, seq_len), cupy.float16)
        tmp_i32 = allocator.alloc_array((self.num_decoder, self.dim_kv * self.num_heads * 2, seq_len), cupy.int32)
        for i in range(batch_size):
            igemm(allocator, value_i8[i][cupy.newaxis], False, self.w_project_kv.value, False, tmp_i32)
            elementwise_copy_scale(tmp_i32, scale[i][cupy.newaxis], self.w_project_kv_scale.value, out_f16[i])
        del tmp_i32
        assert out_f16._c_contiguous

        reshaped_out_f16 = cupy.ndarray(
            (batch_size, self.num_decoder, 2, self.num_heads, self.dim_kv, seq_len),
            dtype=out_f16.dtype,
            memptr=out_f16.data
        )
        return reshaped_out_f16