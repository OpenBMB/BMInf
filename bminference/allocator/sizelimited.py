from typing import List
from .base import Allocator
import cupy

class SizeLimitedAllocator(Allocator):
    def __init__(self, size):
        self._pool = cupy.cuda.MemoryPool()
        self._pool.set_limit(size)

    
    def _alloc(self, size):
        return self._pool.malloc(size)
    