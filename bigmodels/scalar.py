import numpy as np

_ptrs = np.arange(256, dtype=np.int8).astype(np.int32)

def get_scalar_ptr(v):
    return _ptrs[v:].ctypes.data