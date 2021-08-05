from .allocator.base import Allocator
from .context import Context
import cupy
import numpy as np
import logging
logger = logging.getLogger(__name__)

elementwise_copy_scale = cupy._core.create_ufunc('bms_scaled_copy', ('bf->f', 'be->e', 'ef->f'), 'out0 = in0 * in1')

class Parameter:
    def __init__(self, shape, dtype, device_idx = 0, readonly = True):
        self.__shape = shape
        self.__size = 1
        for dim in self.__shape:
            self.__size *= dim
        self.__dtype = np.dtype(dtype)

        self.value = None
        self.scale = None
        self.device = None

        self.data = None
        self.data_scale = None
        self.data_dtype = None
        self.__data_device_idx = device_idx
        self.readonly = readonly
    
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

    @property
    def data_device_idx(self):
        return self.__data_device_idx
    
    @property
    def _prepare_size(self):
        if self.data_dtype is None:
            logger.warning("Getting preapre size before initialization")
            return 0
        if self.__dtype == self.data_dtype:
            return 0
        return self.data_dtype.itemsize * self.size
    
    def put_data(self, shape, data : bytes, scale : float, dtype : np.dtype):
        if not shape == self.__shape:
            raise ValueError("Parameter shape not aligned: requires %s, but got %s" % (self.__shape, shape))
        self.data = data
        self.data_dtype = np.dtype(dtype)
        self.data_scale = scale
    
    def assign_device(self, device_list):
        if self.__data_device_idx > len(device_list):
            raise RuntimeError("Device idx %d > len(device_list) %d" % (self.__data_device_idx, len(device_list)))
        self.device = device_list[self.__data_device_idx]

    def to_device(self, allocator : Allocator, ctx : Context):
        if self.data is None:
            raise RuntimeError("Parameter not intialized")
        

        with self.device:
            addr = allocator.alloc(self.nbytes)

            arr = np.frombuffer(self.data, self.data_dtype)
            self.value = cupy.ndarray(self.shape, dtype=self.__dtype, memptr=addr, order='C')

            if self.__dtype != self.data_dtype:
                if not np.issubdtype(self.__dtype, np.floating): # convert * to floating
                    raise AssertionError("Converting dtype from float to int8")

                self.scale = 1 # scale

                allocator.temp_ptr.copy_from_host_async( arr.ctypes.data, arr.nbytes, ctx.load_stream)
                
                arr = cupy.ndarray( self.shape, dtype=self.data_dtype, memptr=allocator.temp_ptr, order='C' )   # temp var for arr on gpu
                with ctx.load_stream:
                    elementwise_copy_scale(arr, self.__dtype.type(self.data_scale),  self.value)

            else:
                self.scale = self.data_scale
                self.value.data.copy_from_host_async( arr.ctypes.data, arr.nbytes, ctx.load_stream )
    
    def _remove_data(self):
        self.data = None
        self.data_scale = None
        self.data_dtype = None