from .allocator.base import Allocator
from .functions.scale_copy import elementwise_copy_scale
import cupy
import numpy as np
import logging
logger = logging.getLogger(__name__)

class Parameter:
    def __init__(self, shape, dtype):
        self.__shape = shape
        self.__size = 1
        for dim in self.__shape:
            self.__size *= dim
        self.__dtype = np.dtype(dtype)

        self.value = None

        self.data = None
        self.pinned = None
    
    @property
    def nbytes(self):
        return self.__dtype.itemsize * self.__size
    
    @property
    def dtype(self):
        return self.__dtype
    
    @property
    def size(self):
        return self.__size
    
    @property
    def shape(self):
        return self.__shape

    @property
    def ptr(self):
        if self.value is None:
            raise RuntimeError("Reading empty parameter %s" % self)
        if isinstance(self.value, np.ndarray):
            return self.value.__array_interface__["data"][0]
        return self.value.data.mem.ptr

    
    def put_data(self, shape, data : bytes, dtype : np.dtype):
        if not shape == self.__shape:
            raise ValueError("Parameter shape not aligned: requires %s, but got %s" % (self.__shape, shape))
        if dtype != self.__dtype:
            raise ValueError("Parameter dtype error")
        self.data = np.frombuffer(data, self.__dtype)

    def to_device(self, allocator : Allocator, load_stream):
        if self.data is None:
            raise RuntimeError("data is not loaded.")
        
        addr = allocator.alloc(self.nbytes)
        arr = self.data
        self.value = cupy.ndarray(self.shape, dtype=self.__dtype, memptr=addr, order='C')
        cupy.cuda.runtime.memcpyAsync( self.value.data.ptr, arr.ctypes.data, arr.nbytes, cupy.cuda.runtime.memcpyHostToDevice, load_stream.ptr)
    
    def _remove_data(self):
        self.data = None
    
    def _try_pinned(self):
        if self.data is None:
            raise RuntimeError("data is not loaded.")
        try:
            self.pinned = cupy.cuda.alloc_pinned_memory(self.nbytes)
            dst = np.frombuffer(self.pinned, self.data.dtype, self.data.size)
            dst[...] = self.data
            self.data = dst
            logger.info("Allocate pinned %d", self.nbytes)
        except cupy.cuda.runtime.CUDARuntimeError:
            # out of memory
            pass
    
    def __del__(self):
        if self.pinned is not None:
            self.pinned.mem.free()
