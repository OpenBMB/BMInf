import cupy
import logging
import numpy as np
import cupy

logger = logging.getLogger(__name__)
class Allocator:
    def __init__(self):
        pass

    def alloc(self, size) -> cupy.cuda.MemoryPointer:
        logger.info("Allocate %d" % size)
        return self._alloc(size)
    
    def alloc_array(self, shape, dtype) -> cupy.ndarray:
        size = 1
        for it in shape:
            size *= it
        dtype = np.dtype(dtype)

        mem = self.alloc( dtype.itemsize * size )
        return cupy.ndarray(shape, dtype=dtype, memptr=mem, order='C')

    def _alloc(self, size):
        raise NotImplementedError()
