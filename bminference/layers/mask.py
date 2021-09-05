from typing import List
import cupy
from .base import Layer
from ..allocator import Allocator
import numpy as np

class InputMask(Layer):
    def __init__(self, is_decoder):
        self.is_decoder = is_decoder

    def forward(self, allocator : Allocator, length : List[int], seq_len : int):
        out_host = np.zeros((len(length), seq_len, seq_len), dtype=np.bool8)
        for i, it in enumerate(length):
            out_host[i, :it, :it] = True
        
        return cupy.asarray(out_host, dtype=out_host.dtype, order='C')