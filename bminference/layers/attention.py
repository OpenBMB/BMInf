from typing import Optional

from numpy import ndarray
from ..parameter import Parameter
from ..allocator import Allocator
from .base import Layer
import cupy
from ..functions.quantization import quantize
from ..functions.scale_copy import elementwise_copy_scale
from ..functions.gemm import igemm, fgemm
from ..functions.attention_mask import mask_attention_kernel

class SelfAttention(Layer):
    def __init__(self, dim_in, dim_qkv, num_heads):
        self.dim_qkv = dim_qkv
        self.num_heads = num_heads
        self.dim_in = dim_in

        self.w_project_qkv = Parameter((dim_qkv * num_heads * 3, dim_in), cupy.int8)
        self.w_project_qkv_scale = Parameter((dim_qkv * num_heads * 3, 1), cupy.float16)
        self.w_out = Parameter((dim_in, dim_qkv * num_heads), cupy.int8)
        self.w_out_scale = Parameter((dim_in, 1), cupy.float16)
    
    def quantize(self, allocator : Allocator, value : cupy.ndarray, axis = -1):
        if axis < 0:
            axis += len(value.shape)
        nw_value = allocator.alloc_array(value.shape, cupy.int8)
        scale_shape = value.shape[:axis] + (1,) + value.shape[axis + 1:]
        scale = allocator.alloc_array(scale_shape, cupy.float16)
        quantize(value, nw_value, scale, axis=axis)
        return nw_value, scale

    def forward(self, 
            allocator : Allocator,
            hidden_state : cupy.ndarray,
            attention_mask : cupy.ndarray,
            self_attn_position_bias : Optional[cupy.ndarray] = None
        ) -> cupy.ndarray:
        
        batch_size, dim_in, seq_len = hidden_state.shape
        assert dim_in == self.dim_in

        assert hidden_state.dtype == cupy.float16
        assert attention_mask.dtype == cupy.bool
        assert attention_mask.shape == (batch_size, seq_len, seq_len)
        if self_attn_position_bias is not None:
            assert self_attn_position_bias.dtype == cupy.float16
            assert self_attn_position_bias.shape[1:] == (self.num_heads, seq_len, seq_len)

        # (batch_size, dim_in, seq_len), (batch_size, 1, seq_len)
        value, scale = self.quantize(allocator, hidden_state, axis=1) 
        # FIXME: cupy cublasGemmStridedBatchedEx
        qkv_i32 = allocator.alloc_array((3, batch_size, self.num_heads * self.dim_qkv, seq_len), dtype=cupy.int32)
        qkv_f16 = allocator.alloc_array( qkv_i32.shape, dtype=cupy.float16 )

        shaped_project_qkv = self.w_project_qkv.value.reshape(3, self.dim_qkv * self.num_heads, self.dim_in)
        shpaed_project_qkv_scale = self.w_project_qkv_scale.value.reshape(3, self.dim_qkv * self.num_heads, 1)
        
        for i in range(3):
            igemm(
                allocator,
                value,
                False,
                shaped_project_qkv[i:i+1],
                False,
                qkv_i32[i]
            )
            elementwise_copy_scale(
                qkv_i32[i], 
                shpaed_project_qkv_scale[i:i+1],   # (1, dim_qkv * num_heads, 1)
                scale,     # (batch_size, 1, seq_len)
                out=qkv_f16[i]
            )

        # release value
        del value
        del qkv_i32

        # reshape
        assert qkv_f16._c_contiguous
        qkv = cupy.ndarray( (3, batch_size, self.num_heads, self.dim_qkv, seq_len), dtype=cupy.float16, memptr=qkv_f16.data )
        del qkv_f16
        # qkv: batch, 3, num_heads, dim_qkv，seq
        # calc attention score
        attention_score = allocator.alloc_array((batch_size, self.num_heads, seq_len, seq_len), dtype=cupy.float16)

        fgemm(
            allocator, 
            qkv[0].reshape(batch_size * self.num_heads, self.dim_qkv, seq_len), 
            False, 
            qkv[1].reshape(batch_size * self.num_heads, self.dim_qkv, seq_len), 
            True, 
            attention_score.reshape(batch_size * self.num_heads, seq_len, seq_len)
        )
            
        
        # attention_score: batch, num_heads, s_k, s_q
        # add bias
        mask_attention_kernel(
            attention_mask[:, cupy.newaxis, :, :],  # (batch, 1#num_heads, seq_len, seq_len)
            attention_score, 
            cupy.float16(-1e10), 
            out=attention_score
        )
        
        if self_attn_position_bias is not None:
            attention_score += self_attn_position_bias

        # softmax
        temp_attn_mx = allocator.alloc_array((batch_size, self.num_heads, 1, seq_len), dtype=cupy.float16)
        cupy.max(attention_score, axis=-2, out=temp_attn_mx, keepdims=True)
        attention_score -= temp_attn_mx
        cupy.exp(attention_score, out=attention_score)
        cupy.sum(attention_score, axis=-2, out=temp_attn_mx, keepdims=True)

        # temp_attn_mx: batch, num_heads, 1, s_q
        # attention_score: batch, num_heads, s_k, s_q
        attention_score /= temp_attn_mx
        del temp_attn_mx

        # calc value
        # batch, num_heads, s_q, s_k @ batch, num_heads, s_v, dim_value -> batch, num_heads, s_q, dim_qkv
        # attention_score: batch, num_heads, s_q, s_k
        out_raw = allocator.alloc_array((batch_size, self.num_heads, self.dim_qkv, seq_len), dtype=cupy.float16)
        fgemm(
            allocator,
            attention_score.reshape(batch_size * self.num_heads, seq_len, seq_len),
            False, 
            qkv[2].reshape(batch_size * self.num_heads, self.dim_qkv, seq_len), 
            False, 
            out_raw.reshape(batch_size * self.num_heads, self.dim_qkv, seq_len)
        )
            
        assert out_raw._c_contiguous
        # reshape
        out = cupy.ndarray((batch_size, self.num_heads * self.dim_qkv, seq_len), dtype=cupy.float16, memptr=out_raw.data)
        del attention_score
        del out_raw

        # (batch_size, num_heads * dim_qkv, seq_len), (batch_size, 1, seq_len)
        out_i8, scale = self.quantize(allocator, out, axis=1) 

        project_out_i32 = allocator.alloc_array((batch_size, dim_in, seq_len), dtype=cupy.int32)
        igemm(allocator, out_i8, False, self.w_out.value[cupy.newaxis], False, project_out_i32)
        
        assert project_out_i32._c_contiguous
        project_out_f16 = allocator.alloc_array(project_out_i32.shape, dtype=cupy.float16)

        elementwise_copy_scale(
            project_out_i32, 
            self.w_out_scale.value, # (dim_in, 1)
            scale,  # (batch_size, 1, seq_len)
            out=project_out_f16
        )
        return project_out_f16
        

class PartialAttention(Layer):
    def __init__(self, dim_in, dim_qkv, num_heads, is_self_attn):
        self.is_self_attn = is_self_attn
        self.dim_in = dim_in
        self.num_heads = num_heads
        self.dim_qkv = dim_qkv

        if self.is_self_attn:
            self.w_project_qkv = Parameter((dim_qkv * num_heads * 3, dim_in), cupy.int8)
            self.w_project_qkv_scale = Parameter((dim_qkv * num_heads * 3, 1), cupy.float16)
        else:
            self.w_project_q = Parameter((dim_qkv * num_heads, dim_in), cupy.int8)
            self.w_project_q_scale = Parameter((dim_qkv * num_heads, 1), cupy.float16)

        self.w_out = Parameter((dim_in, dim_qkv * num_heads), cupy.int8)
        self.w_out_scale = Parameter((dim_in, 1), cupy.float16)

    def quantize(self, allocator : Allocator, value : cupy.ndarray, axis = -1):
        if axis < 0:
            axis += len(value.shape)
        nw_value = allocator.alloc_array(value.shape, cupy.int8)
        scale_shape = value.shape[:axis] + (1,) + value.shape[axis + 1:]
        scale = allocator.alloc_array(scale_shape, cupy.float16)
        quantize(value, nw_value, scale, axis=axis)
        return nw_value, scale

    def forward(self,
            allocator : Allocator,
            curr_hidden_state : cupy.ndarray,       # (batch, dim_model)
            past_kv: cupy.ndarray,                  # (batch, 2, num_heads, dim_qkv, past_kv_len)
            position_bias : Optional[cupy.ndarray], # (1#batch, num_heads, past_kv_len)
            past_kv_mask : cupy.ndarray,            # (1#batch, past_kv_len)
            decoder_length : Optional[int],         # int
        ):
        batch_size, dim_model = curr_hidden_state.shape
        num_heads, dim_qkv, past_kv_len = past_kv.shape[2:]
        assert past_kv.shape == (batch_size, 2, num_heads, dim_qkv, past_kv_len)
        assert past_kv.dtype == cupy.float16
        assert num_heads == self.num_heads
        assert dim_qkv == self.dim_qkv

        assert curr_hidden_state.dtype == cupy.float16

        if self.is_self_attn:
            assert decoder_length is not None
        if position_bias is not None:
            assert position_bias.shape[1:] == (num_heads, past_kv_len)
            assert position_bias.dtype == cupy.float16
        assert past_kv_mask.shape[-1] == past_kv_len


        # (batch, dim_model), (batch, 1)
        value, scale = self.quantize(allocator, curr_hidden_state[:, :], axis=1) 
        
        # FIXME: cupy cublasGemmStridedBatchedEx
        if self.is_self_attn:
            qkv_i32 = allocator.alloc_array((batch_size, 3 * self.num_heads * self.dim_qkv, 1), dtype=cupy.int32)
        else:
            qkv_i32 = allocator.alloc_array((batch_size, self.num_heads * self.dim_qkv, 1), dtype=cupy.int32)
        
        if self.is_self_attn:
            igemm(
                allocator,
                self.w_project_qkv.value[cupy.newaxis],
                True,
                value[cupy.newaxis],
                False,
                qkv_i32[cupy.newaxis, :, :, 0]
            )
        else:
            igemm(
                allocator,
                self.w_project_q.value[cupy.newaxis],
                True,
                value[cupy.newaxis],
                False,
                qkv_i32[cupy.newaxis, :, :, 0]
            )
        # release value
        del value

        # convert int32 to fp16
        assert qkv_i32._c_contiguous
        qkv_f16 = allocator.alloc_array( qkv_i32.shape, dtype=cupy.float16 )

        if self.is_self_attn:
            elementwise_copy_scale(
                qkv_i32, 
                self.w_project_qkv_scale.value,   # (1#batch_size, dim_qkv * num_heads * 3, 1)
                scale[:, :, cupy.newaxis],     # (batch_size, 1, 1)
                out=qkv_f16
            )
        else:
            elementwise_copy_scale(
                qkv_i32, 
                self.w_project_q_scale.value,   # (dim_qkv * num_heads, 1)
                scale[:, :, cupy.newaxis],     # (batch_size, 1, 1)
                out=qkv_f16
            )
        del qkv_i32
        # reshape
        assert qkv_f16._c_contiguous
        if self.is_self_attn:
            qkv = cupy.ndarray( (batch_size, 3, self.num_heads, self.dim_qkv), dtype=cupy.float16, memptr=qkv_f16.data )
            query = qkv[:, 0, :, :] # (batch, num_heads, dim_qkv)
            past_kv[:, 0, :, :, decoder_length] = qkv[:, 1, :, :]
            past_kv[:, 1, :, :, decoder_length] = qkv[:, 2, :, :]
            del qkv
        else:
            query = cupy.ndarray( (batch_size, self.num_heads, self.dim_qkv), dtype=cupy.float16, memptr=qkv_f16.data )
        del qkv_f16

        # calc attention score
        attention_score = allocator.alloc_array((batch_size, self.num_heads, past_kv_len, 1), dtype=cupy.float16)
        for i in range(batch_size):
            fgemm(
                allocator,
                query[i, :, :, cupy.newaxis],  # (num_heads, dim_qkv, 1)
                False,
                past_kv[i, 0], #(num_heads, dim_qkv, past_kv_len)
                True,
                attention_score[i]  # (num_heads, past_kv_len, 1)
            )
        # mask
        mask_attention_kernel(
            past_kv_mask[:, cupy.newaxis, :, cupy.newaxis], # (batch, 1#num_heads, past_kv_len, 1)
            attention_score,
            cupy.float16(-1e10),
            out=attention_score             # (batch_size, self.num_heads, past_kv_len, 1)
        )
        
        if position_bias is not None:
            attention_score += position_bias[:, :, :, cupy.newaxis] # (1#batch, num_heads, past_kv_len, 1)
        
        # softmax
        temp_attn_mx = allocator.alloc_array((batch_size, self.num_heads, 1, 1), dtype=cupy.float16)
        cupy.max(attention_score, axis=-2, out=temp_attn_mx, keepdims=True)
        attention_score -= temp_attn_mx
        cupy.exp(attention_score, out=attention_score)
        cupy.sum(attention_score, axis=-2, out=temp_attn_mx, keepdims=True)

        attention_score /= temp_attn_mx
        del temp_attn_mx

        out_raw = allocator.alloc_array((batch_size, self.num_heads, self.dim_qkv, 1), dtype=cupy.float16)
        for i in range(batch_size):
            attn = attention_score[i]   # (num_heads, past_kv_len, 1）
            val = past_kv[i, 1]         # (num_heads, dim_qkv, past_kv_len)

            fgemm(allocator, attn, False, val, False, out_raw[i])
        assert out_raw._c_contiguous

        out = cupy.ndarray((batch_size, self.num_heads * self.dim_qkv), dtype=cupy.float16, memptr=out_raw.data)
        del attention_score
        del out_raw

        # (batch_size, num_heads * dim_qkv, 1), (batch_size, 1, 1)
        out_i8, scale = self.quantize(allocator, out, axis=1) 

        project_out_i32 = allocator.alloc_array((batch_size, dim_model, 1), dtype=cupy.int32)

        igemm(
            allocator,
            self.w_out.value[cupy.newaxis], 
            True, 
            out_i8[cupy.newaxis], 
            False, 
            project_out_i32[cupy.newaxis, :, :, 0]
        )
        
        assert project_out_i32._c_contiguous
        
        # (batch, dim_model, 1)
        project_out_f16 = allocator.alloc_array(project_out_i32.shape, dtype=cupy.float16)

        elementwise_copy_scale(
            project_out_i32, 
            self.w_out_scale.value, # (1#batch_size, dim_model, 1)
            scale[:, :, cupy.newaxis],  # (batch, 1, 1)
            out=project_out_f16
        )
        return project_out_f16[:, :, 0] # (batch, dim_model)