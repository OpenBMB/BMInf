from typing import Optional
from ..parameter import Parameter
from ..allocator import Allocator
from .base import Layer
import cupy
from ..functions.quantization import quantize
from ..functions.scale_copy import elementwise_copy_scale
from ..functions.gemm import igemm, sgemmBatched
from ..functions.attention_mask import mask_attention_kernel

class SelfAttention(Layer):
    def __init__(self, dim_in, dim_qkv, num_heads):
        self.dim_qkv = dim_qkv
        self.num_heads = num_heads
        self.dim_in = dim_in

        self.w_project_qkv = Parameter((dim_qkv * num_heads * 3, dim_in), cupy.int8)
        self.w_out = Parameter((dim_qkv * num_heads, dim_in), cupy.int8)
    
    def quantize(self, allocator : Allocator, value : cupy.ndarray):
        nw_value = allocator.alloc_array(value.shape, cupy.int8)
        scale = quantize(value, out=nw_value)
        return nw_value, scale

    def forward(self, 
            allocator : Allocator,
            hidden_state : cupy.ndarray,
            attention_mask : cupy.ndarray,
            self_attn_position_bias : Optional[cupy.ndarray] = None
        ) -> cupy.ndarray:
        
        batch_size, dim_in, seq_len = hidden_state.shape
        assert hidden_state._c_contiguous # hidden_state is contiguous in C-order
        assert dim_in == self.dim_in

        assert hidden_state.dtype == cupy.float32
        assert attention_mask.dtype == cupy.bool
        assert attention_mask.shape == (batch_size, seq_len, seq_len)
        if self_attn_position_bias is not None:
            assert self_attn_position_bias.dtype == cupy.float32
            assert self_attn_position_bias.shape == (batch_size, self.num_heads, seq_len, seq_len)

        value, scale = self.quantize(allocator, hidden_state)
        
        # FIXME: cupy cublasGemmStridedBatchedEx
        qkv_i32 = allocator.alloc_array((batch_size, 3 * self.num_heads * self.dim_qkv, seq_len), dtype=cupy.int32)
        for i in range(batch_size):
            igemm(
                value[i],
                True,
                self.w_project_qkv.value,
                True,
                qkv_i32[i]
            )
        # release value
        del value

        # convert int32 to fp32
        qkv_f32 = cupy.ndarray( qkv_i32.shape, dtype=cupy.float32, memptr=qkv_i32.data )
        elementwise_copy_scale( qkv_i32, self.w_project_qkv.scale * scale, out=qkv_f32 )
        del qkv_i32
        # reshape
        qkv = cupy.ndarray( (batch_size, 3, self.num_heads, self.dim_qkv, seq_len), dtype=cupy.float32, memptr=qkv_f32.data )
        del qkv_f32
        # qkv: batch, 3, num_heads, dim_qkv，seq
        
        # calc attention score
        attention_score = allocator.alloc_array((batch_size, self.num_heads, seq_len, seq_len), dtype=cupy.float32)
        for i in range(batch_size):
            sgemmBatched(qkv[i][0], True, qkv[i][1], False, attention_score[i])
        
        # attention_score: batch, num_heads, s_q, s_k
        # add bias
        mask_attention_kernel(
            cupy.reshape(attention_mask, (batch_size, 1, seq_len, seq_len), order='c'), 
            attention_score, 
            cupy.float32(-1e10), 
            out=attention_score
        )
        if self_attn_position_bias is not None:
            attention_score += self_attn_position_bias
        
        # softmax
        temp_attn_mx = allocator.alloc_array((batch_size, self.num_heads, seq_len, 1), dtype=cupy.float32)
        cupy.max(attention_score, axis=-1, out=temp_attn_mx, keepdims=True)
        attention_score -= temp_attn_mx
        cupy.exp(attention_score, out=attention_score)
        cupy.sum(attention_score, axis=-1, out=temp_attn_mx, keepdims=True)

        # temp_attn_mx: batch, num_heads, s_q, 1
        attention_score /= temp_attn_mx
        del temp_attn_mx

        # calc value
        # batch, num_heads, s_q, s_k @ batch, num_heads, s_v, dim_value -> batch, num_heads, s_q, dim_qkv
        # attention_score: batch, num_heads, s_q, s_k
        out_raw = allocator.alloc_array((batch_size, self.num_heads, self.dim_qkv, seq_len), dtype=cupy.float32)
        for i in range(batch_size):
            attn = attention_score[i] # num_heads, s_q, s_k
            val = qkv[i][2]    # num_heads, dim_qkv，s_v
            sgemmBatched(attn, False, val, True, out=out_raw[i])
        out = cupy.ndarray((batch_size, self.num_heads * self.dim_qkv, seq_len), dtype=cupy.float32, memptr=out_raw.data)
        del attention_score

        out_i8, scale = self.quantize(allocator, out)
        project_out_i32 = allocator.alloc_array((batch_size, dim_in, seq_len), dtype=cupy.int32)
        for i in range(batch_size):
            igemm(out_i8[i], True, self.w_out.value, True, out=project_out_i32[i])
        project_out_f32 = cupy.ndarray(project_out_i32.shape, dtype=cupy.float32, memptr=project_out_i32.data)
        elementwise_copy_scale(project_out_i32, self.w_out.scale * scale, out=project_out_f32)
        return project_out_f32
        

class CrossAttention(Layer):
    def __init__(self, dim_in, dim_qkv, num_heads, ltype = 1):
        self.w_project_q = Parameter((dim_in, dim_qkv * num_heads), cupy.int8)
        self.w_out = Parameter((dim_qkv * num_heads, dim_in), cupy.int8)
