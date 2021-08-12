from ..parameter import Parameter
from .base import Layer
import cupy

class EncoderKeyValueProjection(Layer):
    def __init__(self, num_decoder, dim_in, dim_kv, num_heads):
        self.w_project_kv = Parameter((num_decoder * dim_kv * num_heads * 2, dim_in), cupy.int8)
    
    def forward(self, x):
        pass