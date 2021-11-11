from typing import List
from ..device import Device
from ..allocator import Allocator, Memory
from cpm_kernels.library import cudart
from ..utils import round_up

class CUDAAllocator(Allocator):
    def __init__(self, device_idx) -> None:
        super().__init__()
        self.device = Device(device_idx)

    def allocate(self, nbytes: int) -> Memory:
        nbytes = round_up(nbytes, 512)
        with self.device:
            ptr = cudart.cudaMalloc(nbytes).value
            return Memory(ptr, nbytes, self.device)
    
    def free(self, mem: Memory) -> None:
        with self.device:
            cudart.cudaFree(mem.ptr)