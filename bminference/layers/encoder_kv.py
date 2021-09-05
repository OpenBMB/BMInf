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

        self.w_project_kv = Parameter((num_decoder * dim_kv * num_heads * 2, dim_in), cupy.int8)
        self.w_project_kv_scale = Parameter((num_decoder * dim_kv * num_heads * 2, 1), cupy.float16)
    
    def forward(self, allocator : Allocator, x):
        assert x.dtype == cupy.float16

        value = x

        batch_size, dim_model, seq_len = value.shape
        assert dim_model == self.dim_in

        value_i8 = allocator.alloc_array(value.shape, cupy.int8)
        scale = allocator.alloc_array((batch_size, 1, seq_len), dtype=cupy.float16)
        
        # (batch_size, dim_model, seq_len), (batch_size, 1, seq_len)
        quantize(value, value_i8, scale, axis=1)

        out_i32 = allocator.alloc_array((batch_size, self.num_decoder * self.dim_kv * self.num_heads * 2, seq_len), cupy.int32)
        
        # FIXME: igemm Stried Batched cupy
        for i in range(batch_size):
            igemm(value_i8[i], True, self.w_project_kv.value, True, out_i32[i])
        
        out_f32 = allocator.alloc_array(out_i32.shape, cupy.float32)
        elementwise_copy_scale(out_i32, scale, self.w_project_kv_scale.value, out_f32)
        
        assert out_f32._c_contiguous

        reshaped_out_f32 = cupy.ndarray(
            (batch_size, self.num_decoder, 2, self.num_heads, self.dim_kv, seq_len),
            dtype=out_f32.dtype,
            memptr=out_f32.data
        )
        return reshaped_out_f32