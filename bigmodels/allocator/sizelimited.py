from typing import List
from .base import Allocator, AllocatorConfig
import cupy

class SizeLimitedAllocator(Allocator):
    def __init__(self, config : List[AllocatorConfig]):
        self._pool = cupy.cuda.MemoryPool()
        self._common_ptr = {}
        for cfg in config:
            with cfg.device:
                self._pool.set_limit(cfg.size + cfg.temp_size)
                if cfg.temp_size > 0:
                    self._common_ptr[cfg.device.id] = self._pool.malloc(cfg.temp_size)

    
    def _alloc(self, size):
        return self._pool.malloc(size)
    
    @property
    def temp_ptr(self):
        device_id = cupy.cuda.get_device_id()
        return self._common_ptr[device_id]
    
