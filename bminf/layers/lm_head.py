from .base import Layer
from ..parameter import Parameter
from ..allocator import Allocator
import cupy
from ..functions.gemm import fgemm

class LMHead(Layer):
    def __init__(self, vocab_size, dim_model):
        self.dim_model = dim_model
        self.vocab_size = vocab_size
        self.weight = Parameter((vocab_size, dim_model), dtype=cupy.float16)

    def forward(self, allocator : Allocator, x):
        # call function linear
        batch_size, dim_model = x.shape
        assert dim_model == self.dim_model
        
        ret = allocator.alloc_array((1, batch_size, self.vocab_size), dtype=cupy.float16)
        fgemm(allocator, self.weight.value[cupy.newaxis], True, x[cupy.newaxis], False, ret)

        return ret[0]
