from typing import Optional
from ..core import Layer, Context, Tensor, Parameter
import numpy as np
from cpm_kernels import kernels as ck

class PositionEmbedding(Layer):
    def __init__(self, num_heads, num_buckets, max_distance, bidirectional=True):
        super().__init__()

        self.weight = Parameter((num_heads, num_buckets), dtype=np.float16)

        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
    
    def forward(self,
            ctx : Context, 
            key_len : int,
            query_len : int,
            x_out : Tensor          # (num_heads, key_len, query_len)
        ):
        assert x_out.shape == (self.num_heads, key_len, query_len)
        
        mapping = ctx.allocate((self.max_distance,), np.int32)
        ck.position_embedding_init(
            self.num_buckets,
            self.max_distance,
            mapping.ptr,
            self.bidirectional,
            ctx.current_stream
        )
        ck.position_embedding_forward(
            query_len,
            key_len,
            self.num_buckets,
            self.max_distance,
            self.num_heads,
            mapping.ptr,
            self.weight.value.ptr,
            x_out.ptr,
            self.bidirectional,
            ctx.current_stream
        )
        ctx.free(mapping)
    
    def step(self,
            ctx : Context, 
            key_len : int,
            query_pos : int,
            x_out : Tensor
        ):
        assert x_out.shape == (self.num_heads, key_len)
        mapping = ctx.allocate((self.max_distance,), np.int32)
        ck.position_embedding_init(
            self.num_buckets,
            self.max_distance,
            mapping.ptr,
            self.bidirectional,
            ctx.current_stream
        )
        ck.position_embedding_step(
            query_pos,
            key_len,
            self.num_buckets,
            self.max_distance,
            self.num_heads,
            mapping.ptr,
            self.weight.value.ptr,
            x_out.ptr,
            self.bidirectional,
            ctx.current_stream
        )
        ctx.free(mapping)
    