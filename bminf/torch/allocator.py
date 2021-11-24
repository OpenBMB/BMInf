import torch
from ..core import Allocator, Memory, Device
from ..utils import round_up

class TorchAllocator(Allocator):
    def __init__(self):
        self._mem_set = set()
        self.__used = 0
        self.__peak = 0

    def allocate(self, nbytes: int, stream = 0) -> Memory:
        if nbytes <= 0:
            return Memory(0, 0, 0)

        nbytes = round_up(nbytes, 512)  # CUDA 512 bytes aligned
        mem_ptr = torch.cuda.memory.caching_allocator_alloc(nbytes, stream=int(stream))
        self._mem_set.add(mem_ptr)
        self.__used += nbytes
        self.__peak = max(self.__peak, self.__used)
        return Memory(mem_ptr, nbytes, Device(torch.cuda.current_device()))
    
    def free(self, mem: Memory) -> None:
        if mem.ptr not in self._mem_set:
            raise RuntimeError("Memory is already freed")
        
        self.__used -= mem.nbytes
        torch.cuda.memory.caching_allocator_delete(mem.ptr)
        self._mem_set.remove(mem.ptr)
    
    def memory_stats(self):
        return {
            "active": len(self._mem_set) - 1,
            "used": self.__used,
            "peak": self.__peak,
        }

    def free_all(self):
        for it in self._mem_set:
            torch.cuda.memory.caching_allocator_delete(it)
        self._mem_set = set()
        self.__used = 0
        self.__peak = 0
