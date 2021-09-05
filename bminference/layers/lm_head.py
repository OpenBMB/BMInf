from .base import Layer
from ..parameter import Parameter
from ..allocator import Allocator
import cupy
from ..functions.gemm import sgemmBatched

class LMHead(Layer):
    def __init__(self, vocab_size, dim_model):
        self.dim_model = dim_model
        self.vocab_size = vocab_size
        self.weight = Parameter((vocab_size, dim_model), dtype=cupy.float16)

    def forward(self, allocator : Allocator, x):
        # call function linear
        batch_size, dim_model = x.shape
        assert dim_model == self.dim_model

        return cupy.matmul(x, self.weight.value.T)
