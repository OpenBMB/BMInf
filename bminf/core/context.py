from typing import List, Tuple, Type
from .tensor import Tensor
from .device import Device
from .allocator import Allocator
from cpm_kernels.library import cudart
import numpy as np

class Context:
    requires_grad : bool

    def __init__(self, 
            device_idx : List[int], 
            allocators : List[Allocator],
            requires_grad : bool
        ) -> None:

        assert len(device_idx) > 0, "device_idx must be a non-empty list"
        assert len(device_idx) == len(allocators)

        self.requires_grad = requires_grad
        self.__devices = [
            Device(idx) for idx in device_idx
        ]
        self.__calc_streams = {}
        for d in self.__devices:
            with d:
                self.__calc_streams[d.idx] = cudart.cudaStreamCreate()

        self.__allocators = {
            device_idx : allocator for device_idx, allocator in zip(device_idx, allocators)
        }
        

    def allocate(self, shape : int, dtype : np.dtype) -> Tensor:
        device = Device(cudart.cudaGetDevice())
        allocator = self.__allocators[device.idx]
        
        dtype = np.dtype(dtype)

        itemsize = dtype.itemsize
        
        size = 1
        for dim in shape:
            size *= dim
        
        nbytes = size * itemsize

        mem = allocator.allocate(nbytes)
        return Tensor(mem, shape, dtype)

    def free(self, tensor : Tensor):
        allocator = self.__allocators[tensor.device_id]
        tensor._released = True
        allocator.free(tensor._memory)
    
    def device(self, device_idx : int) -> Device:
        return self.__devices[device_idx]
    
    @property
    def current_stream(self):
        device_idx = cudart.cudaGetDevice()
        return self.__calc_streams[device_idx]
