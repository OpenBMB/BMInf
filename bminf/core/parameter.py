from .tensor import Tensor
from .memory import Memory
import numpy as np

class Parameter:
    def __init__(self, shape, dtype : np.dtype):
        self.offset = 0
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.nbytes = self.dtype.itemsize * int(np.prod(self.shape))
        self.__data = None
        self.__base_ptr = None
    
    @property
    def value(self) -> Tensor:
        return Tensor(Memory(self.__base_ptr.ptr + self.offset, self.nbytes, self.__base_ptr.device), self.shape, self.dtype)
    
    def _update_ptrs(self, base_ptr : Memory):
        self.__base_ptr = base_ptr

    @property
    def data(self) -> np.ndarray:
        return self.__data
    
    def put_data(self, shape, data : bytes, dtype : np.dtype):
        if not shape == self.shape:
            raise ValueError("Parameter shape not aligned: requires %s, but got %s" % (self.shape, shape))
        if dtype != self.dtype:
            raise ValueError("Parameter dtype error")
        if self.__data is None:
            raise RuntimeError("Parameter memory is not allocated")
        self.__data[...] = np.frombuffer(data, np.byte)
    
    def _init_data(self, data : np.ndarray):
        self.__data = data