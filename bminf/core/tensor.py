from typing import Tuple

from cpm_kernels.library import cudart
from .memory import Memory
from .device import Device
import numpy as np

class Tensor:
    def __init__(self, memory : Memory, shape : Tuple[int, ...], dtype : np.dtype) -> None:
        self.__shape = shape
        self.__memory = memory
        self.__dtype = dtype
        self.__nbytes = int(np.prod(shape)) * dtype.itemsize
        self._released = False
    
    @property
    def device_id(self):
        return self._memory.device.idx

    @property
    def device(self):
        return self._memory.device
    
    @property
    def dtype(self):
        return self.__dtype

    @property
    def ptr(self):
        if self._released:
            raise RuntimeError("Tensor has been released")
        return self.__memory.ptr
    
    @property
    def shape(self):
        return self.__shape
    
    @property
    def _memory(self):
        return self.__memory
    
    def reshape(self, nw_shape : Tuple[int, ...]) -> None:
        nw_size = 1
        for dim in nw_shape:
            nw_size *= dim
        
        old_size = 1
        for dim in self.__shape:
            old_size *= dim
        
        assert nw_size == old_size, "New shape must be the same size as old shape"

        self.__shape = nw_shape

    def __str__(self):
        tmp = np.empty(self.__shape, dtype=self.__dtype)
        cudart.cudaMemcpy(tmp.ctypes.data, self.ptr, tmp.nbytes, cudart.cudaMemcpyDeviceToHost)
        return f"Tensor(shape={self.__shape}, dtype={self.__dtype}, device={self.__memory.device})\n" + str(tmp)
    
    @property
    def nbytes(self):
        return self.__nbytes

    @staticmethod
    def from_numpy(ctx, numpy_array : np.ndarray) -> "Tensor":
        if not numpy_array.flags["C_CONTIGUOUS"]:
            numpy_array = np.ascontiguousarray(numpy_array)
        tensor = ctx.allocate(numpy_array.shape, numpy_array.dtype)
        cudart.cudaMemcpy(
            tensor.ptr,
            numpy_array.ctypes.data,
            numpy_array.nbytes,
            cudart.cudaMemcpyHostToDevice
        )
        return tensor
    
    def zero_(self, ctx):
        cudart.cudaMemsetAsync(
            self.ptr,
            0,
            self.__nbytes,
            ctx.current_stream
        )
