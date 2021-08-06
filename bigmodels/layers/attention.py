from typing import Optional
from ..parameter import Parameter
from ..allocator import Allocator
from .base import Layer
from ..tensor import Tensor
import cupy
from ..functions.quantization import quantize, dequantize
from ..scalar import get_scalar_ptr

class SelfAttention(Layer):
    def __init__(self, dim_in, dim_qkv, num_heads):
        self.dim_kqv = dim_qkv
        self.num_heads = num_heads

        self.w_project_kqv = Parameter((dim_qkv * num_heads * 3, dim_in), cupy.int8)
        self.w_out = Parameter((dim_qkv * num_heads, dim_in), cupy.int8)
    
    def forward(self, 
            allocator : Allocator,
            hidden_state : Tensor,
            attention_mask : cupy.ndarray,
            self_attn_position_bias : Optional[Tensor] = None
        ):
        
        batch_size, seq_len, dim_in = hidden_state.value.shape
        assert hidden_state._c_contiguous # hidden_state is contiguous in C-order
        
        input_dtype = hidden_state.value.dtype
        assert input_dtype == cupy.float32

        
        device = cupy.cuda.Device(cupy.cuda.get_device_id())

        value = hidden_state.value
        scale = hidden_state.scale

        nw_value = allocator.alloc_array(value.shape, cupy.int8)
        scale = quantize(value, out=nw_value)
        value = nw_value
        del nw_value
        
        # FIXME: cupy cublasGemmStridedBatchedEx
        for i in range(batch_size):
            cupy.cublas.cublas.sgemmEx(
                device.cublas_handle, 
                cupy.cublas.cublas.CUBLAS_OP_T, 
                cupy.cublas.cublas.CUBLAS_OP_N,
                seq_len,    # rows of 
                self.dim_kqv * self.num_heads * 3,
                dim_in,
                get_scalar_ptr(1),  # int32
                hidden_state.value[i].data.ptr, # op(A) = seq_len x dim_in
                cupy.cuda.runtime.CUDA_R_8I,
                dim_in,
                self.w_project_kqv.value.data.ptr,  # op(B) = dim_in x (dim_qkv * num_heads * 3)
                cupy.cuda.runtime.CUDA_R_8I,
                dim_in,
                get_scalar_ptr(0),
                
            )
        
                # input and weight have same dtype

                
# 
                

class CrossAttention(Layer):
    TYPE_F32 = 1
    TYPE_F16 = 2
    TYPE_I8 = 3

    def __init__(self, dim_in, dim_qkv, num_heads, ltype = 1):
        if ltype == self.TYPE_F32:
            self.w_project_kqv = Parameter((dim_qkv * num_heads, dim_in), cupy.float32)
            self.w_out = Parameter((dim_qkv * num_heads, dim_in), cupy.float32)
        elif ltype == self.TYPE_F16:
            self.w_project_kqv = Parameter((dim_in, dim_qkv * num_heads), cupy.float16)
            self.w_out = Parameter((dim_qkv * num_heads, dim_in), cupy.float16)
        elif ltype == self.TYPE_I8:
            self.w_project_kqv = Parameter((dim_in, dim_qkv * num_heads), cupy.int8)
            self.w_out = Parameter((dim_qkv * num_heads, dim_in), cupy.int8)
        else:
            raise TypeError("Unknown type for %s (%s)" % (self.__class__.__name__, ltype))
    