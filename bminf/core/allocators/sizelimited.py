from typing import List
from ..allocator import Allocator, Memory
from cpm_kernels.library import cudart
from ..utils import round_up

    
def move_memory(dst : int, src : int, nbytes : int, stream : int):
    assert dst < src
    if dst + nbytes > src:
        # memory overlap
        overlap_bytes = dst + nbytes - src
        cudart.cudaMemcpyAsync(
            dst,
            src,
            overlap_bytes,
            cudart.cudaMemcpyDeviceToDevice,
            stream
        )
        dst += overlap_bytes
        src += overlap_bytes
        nbytes -= overlap_bytes
    cudart.cudaMemcpyAsync(
        dst,
        src,
        nbytes,
        cudart.cudaMemcpyDeviceToDevice,
        stream
    )

class SizeLimitedAllocator(Allocator):
    def __init__(self, base_ptr : Memory) -> None:
        self.__base_ptr = base_ptr.ptr
        self.__device = base_ptr.device
        self.__nbytes = base_ptr.nbytes

        self.__used = 0
        self.__peak = 0

        self.__mems : List[Memory] = [ Memory(self.__base_ptr + self.__nbytes, -1, 0) ]   # sentinel element
    
    def _new_mem_pos(self, ptr : int, nbytes : int, pos : int) -> Memory:
        mem = Memory(ptr, nbytes, self.__device)
        self.__mems.insert(pos, mem)

        self.__used += nbytes
        self.__peak = max(self.__peak, self.__used)

        return mem
    
    def allocate(self, nbytes: int, stream = 0) -> Memory:
        if nbytes <= 0:
            return Memory(0, -2, 0)

        nbytes = round_up(nbytes, 512)  # CUDA 512 bytes aligned

        last_end = round_up(self.__base_ptr, 512)
        for idx, mem in enumerate(self.__mems):
            gap = (mem.ptr - last_end)
            if gap >= nbytes:
                # Found a gap large enough
                return self._new_mem_pos(last_end, nbytes, idx)
            # Update last_end
            last_end = round_up(mem.ptr + mem.nbytes, 512)
        
        # Memory compression required
        curr_pos = round_up(self.__base_ptr, 512)

        for idx, mem in enumerate(self.__mems):
            assert mem.ptr >= curr_pos, "Error mem ptr %d < curr_pos %d" % (mem.ptr, curr_pos)
            
            gap = mem.ptr - curr_pos
            if gap >= nbytes:
                # early stop and avoid moving sentinel element
                return self._new_mem_pos(curr_pos, nbytes, idx)

            if mem.ptr == curr_pos or mem.nbytes == 0:
                # do not need to move (already compressed or sentinel)
                curr_pos = round_up(curr_pos + mem.nbytes, 512)
                continue
            
            # compress memory
            move_memory(curr_pos, mem.ptr, mem.nbytes, stream)
            mem.ptr = curr_pos
            curr_pos = round_up(curr_pos + mem.nbytes, 512)
        
        raise RuntimeError("CUDA Error: out of memory. Used %d bytes, capacity: %d btyes." % (self.__used, self.__nbytes, ))

    def free(self, mem: Memory) -> None:
        len_before = len(self.__mems)
        self.__mems.remove(mem)
        len_after = len(self.__mems)

        if len_before == len_after:
            raise RuntimeError("Memory is already freed")
        
        self.__used -= mem.nbytes
    
    def memory_stats(self):
        return {
            "active": len(self.__mems) - 1,
            "used": self.__used,
            "peak": self.__peak,
        }

    def free_all(self):
        self.__mems = [ Memory(self.__base_ptr + self.__nbytes, -1, 0) ]   # sentinel element
        self.__used = 0
        self.__peak = 0