from typing import List
from .base import Allocator, AllocatorConfig
import cupy

class ReusedAllocator(Allocator):
    def __init__(self, config : List[AllocatorConfig]):
        self.__base_ptr = {}
        self.__allocate_limit = {}
        self.__offset = {}

        for cfg in config:
            total = cfg.size + cfg.temp_size
            with cfg.device:
                self.__base_ptr[cfg.device.id] = cupy.cuda.Memory(total)
            self.__allocate_limit[cfg.device.id] = cfg.size
            self.__offset[cfg.device.id] = 0

    def reset(self):
        for kw in self.__offset.keys():
            self.__offset[kw] = 0
    
    def _alloc(self, size):
        device_id = cupy.cuda.get_device_id()

        offset = self.__offset[device_id]
        self.__offset[device_id] += size
        if self.__offset[device_id] > self.__allocate_limit[device_id]:
            raise RuntimeError("Memory limit exceeded %d > %d on device" % (self.__offset[device_id], self.__allocate_limit[device_id], device_id))
        return cupy.cuda.MemoryPointer(self.__base_ptr[device_id], offset)
    
    @property
    def temp_ptr(self):
        device_id = cupy.cuda.get_device_id()

        return cupy.cuda.MemoryPointer(self.__base_ptr[device_id], self.__allocate_limit[device_id])