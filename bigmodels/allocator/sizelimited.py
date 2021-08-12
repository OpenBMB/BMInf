from typing import List
from .base import Allocator
import cupy

class SizeLimitedAllocator(Allocator):
    def __init__(self, size, temp_size):
        total = size + temp_size
        
        self._pool = cupy.cuda.MemoryPool()
        self._pool.set_limit(total)

        self._common_ptr = self._pool.malloc(temp_size)
    
    def _alloc(self, size):
        return self._pool.malloc(size)
    
    @property
    def temp_ptr(self):
        return self._common_ptr
    
