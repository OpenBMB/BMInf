from typing import List
from ..device import Device
from ..allocator import Allocator, Memory
from cpm_kernels.library import cudart
from ..utils import round_up

class CUDAAllocator(Allocator):
    def __init__(self, device_idx) -> None:
        super().__init__()
        self.device = Device(device_idx)

        self.used = 0
        self.peak = 0

        self.__allocated = set()

    def allocate(self, nbytes: int, stream = 0) -> Memory:
        nbytes = round_up(nbytes, 512)
        with self.device:
            ptr = cudart.cudaMalloc(nbytes).value
            self.used += nbytes
            self.peak = max(self.peak, self.used)

            ret = Memory(ptr, nbytes, self.device)
            self.__allocated.add(ret)
            return ret
    
    def free(self, mem: Memory) -> None:
        with self.device:
            cudart.cudaFree(mem.ptr)
            self.used -= mem.nbytes
            self.__allocated.remove(mem)
    
    def memory_stats(self):
        return {
            "peak": self.peak,
            "used": self.used
        }
    
    def free_all(self):
        with self.device:
            if cudart.MALLOC_AYNC_SUPPORT:
                for mem in self.__allocated:
                    cudart.cudaFreeAsync(mem.ptr, cudart.cudaStreamNonBlocking)
                cudart.cudaStreamSynchronize(cudart.cudaStreamNonBlocking)
            else:
                for mem in self.__allocated:
                    cudart.cudaFree(mem.ptr)

    def __del__(self):
        try:
            self.free_all()
        except Exception:
            pass
