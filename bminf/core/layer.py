import ctypes
from typing import Dict, Tuple
from bminf.core import device

from bminf.core.memory import Memory
from .parameter import Parameter
import logging, struct
import numpy as np
from cpm_kernels.library import cudart
from .utils import round_up

logger = logging.getLogger(__name__)

def load_dtype(fp):
    v = struct.unpack("B", fp.read(1))[0]
    if v == 0:
        return np.int8
    elif v == 1:
        return np.float16
    else:
        raise TypeError("Unknown dtype %d" % v)

def load_string(fp):
    size = struct.unpack("I", fp.read(4))[0]
    v = fp.read(size)
    return v.decode("utf-8")

def load_tuple(fp):
    dim_tuple = struct.unpack("B", fp.read(1))[0]
    ret = []
    for _ in range(dim_tuple):
        ret.append(struct.unpack("I", fp.read(4))[0]) 
    return tuple(ret)

def load_parameter(fp):    
    shape = load_tuple(fp)
    value_size = struct.unpack("I", fp.read(4))[0]
    dtype = load_dtype(fp)
    value = fp.read(value_size)
    return shape, value, dtype

def dump_dtype(v, fp):
    sv = -1
    if v == np.int8:
        sv = 0
    elif v == np.float16:
        sv = 1
    if sv == -1:
        raise TypeError("Unknown dtype %s" % v)
    fp.write( struct.pack("B", sv) )

def dump_string(v, fp):
    v = v.encode("utf-8")
    fp.write( struct.pack("I", len(v)) )
    fp.write(v)

def dump_tuple(v, fp):
    fp.write( struct.pack("B", len(v)) )
    for i in v:
        fp.write( struct.pack("I", i) )

def dump_parameter(shape : Tuple[int, ...], data : bytes, data_dtype : np.dtype, fp):
    dump_tuple(shape, fp)
    fp.write( struct.pack("I", len(data)) )
    dump_dtype(data_dtype, fp)
    fp.write(data)


class Layer:
    def __init__(self) -> None:
        self.__ensure_variables()
        self.data = None
        self._parent = None

        self.is_fixed : bool = False
        self.locked : bool = False
        self.on_device : bool = False
        self.loader_event = None
    
    def _set_parent(self, node : 'Layer', offset_bytes):
        if self._parent is not None:
            if self._parent is not node:
                raise RuntimeError("Layer %s is already parrent of %s" % (self._parent.__class__.__name__, self.__class__.__name__))
        self._parent = node
        for _, val in self._parameters.items():
            val.offset += offset_bytes
        for _, val in self._sub_layers.items():
            val._set_parent(self, offset_bytes)
        
    def load(self, fp):
        logger.debug("Parameter Loader [%s]: size %s", self.__class__.__name__, self.nbytes)
        num_parameters, num_sub_layers = struct.unpack("II", fp.read(8))
        logger.debug("Parameter Loader [%s]: parameters %d, sub_layers %d", self.__class__.__name__, num_parameters, num_sub_layers)

        for _ in range(num_parameters):
            name = load_string(fp)
            shape, value, dtype = load_parameter(fp)
            self._parameters[name].put_data(shape, value, dtype)
        for _ in range(num_sub_layers):
            name = load_string(fp)
            logger.debug("In %s: ==", name)
            self._sub_layers[name].load(fp)
            logger.debug("Out %s: ==", name)
    
    def dump(self, fp):
        logger.debug("Parameter Dumper [%s]: size %s", self.__class__.__name__, self.nbytes)
        num_parameters, num_sub_layers = len(self._parameters), len(self._sub_layers)
        fp.write( struct.pack("II", num_parameters, num_sub_layers) )
        for kw, val in self._parameters.items():
            dump_string(kw, fp)
            if val.data is not None:
                dump_parameter(val.shape, val.data.tobytes(), val.dtype, fp)
            else:
                raise RuntimeError("Paraemter %s has no data" % kw)
        for kw, val in self._sub_layers.items():
            dump_string(kw, fp)
            val.dump(fp)
    

    def __ensure_variables(self):
        if not hasattr(self, "_parameters"):
            self._parameters : Dict[str, Parameter] = {}
        if not hasattr(self, "_parameter_bytes"):
            self._parameter_bytes = 0
        if not hasattr(self, "_sub_layers"):
            self._sub_layers : Dict[str, Layer] = {}
    
    @property
    def nbytes(self):
        return self._parameter_bytes

    def _add_parameter(self, name, v : Parameter):
        if self._parent is not None:
            raise RuntimeError("Adding parameter after intialization")
        
        v.offset = self._parameter_bytes
        self._parameters[name] = v
        self._parameter_bytes += round_up(v.nbytes, 512)

    def _add_sublayer(self, name, layer : 'Layer'):
        if self._parent is not None:
            raise RuntimeError("Adding parameter after intialization")
        layer._set_parent(self, self._parameter_bytes)
        self._sub_layers[name] = layer
        self._parameter_bytes += layer.nbytes

    def __setattr__(self, name: str, value):
        if not name.startswith("_"):
            if isinstance(value, Parameter):
                self._add_parameter(name, value)
            elif isinstance(value, Layer):
                self._add_sublayer(name, value)
        super().__setattr__(name, value)

    def init_data(self, pinned : bool = False):
        if self.data is None:
            if pinned:
                try:
                    ptr = cudart.cudaMallocHost(self.nbytes)
                    self.data = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_byte)), shape=(self.nbytes,))
                except RuntimeError: # out of memory
                    
                    # fallback to non-pinned memory
                    self.data = np.empty(self.nbytes, dtype=np.uint8)
            else:
                self.data = np.empty(self.nbytes, dtype=np.uint8)
        
        for _, val in self._sub_layers.items():
            assert val._parent == self
            val.data = self.data
            val.init_data(pinned)
        
        for _, val in self._parameters.items():
            val._init_data(self.data[val.offset: val.offset + val.nbytes])

    def _update_ptrs(self, base_ptr):
        for _, val in self._parameters.items():
            val._update_ptrs(base_ptr)
        for _, val in self._sub_layers.items():
            val._update_ptrs(base_ptr)
    
    def _to_device(self, device_ptr : Memory, stream = None):
        if self.data is None:
            raise RuntimeError("Layer %s is not initialized" % self.__class__.__name__)
        ptr = self.data.ctypes.data
        with device_ptr.device:
            if stream is None:
                cudart.cudaMemcpy(
                    device_ptr.ptr,
                    ptr,
                    self.nbytes,
                    cudart.cudaMemcpyHostToDevice
                )
            else:
                cudart.cudaMemcpyAsync(
                    device_ptr.ptr,
                    ptr,
                    self.nbytes,
                    cudart.cudaMemcpyHostToDevice,
                    stream
                )
        self._update_ptrs(device_ptr)

class Model(Layer):
    def _add_parameter(self, name, v : Parameter):
        raise NotImplementedError()

    def _add_sublayer(self, name, layer : 'Layer'):
        self._sub_layers[name] = layer
        self._parameter_bytes += layer.nbytes
    
    def init_data(self):
        for _, val in self._sub_layers.items():
            val.init_data(pinned=False)
    
    def _to_device(self, device_ptr):
        raise NotImplementedError()
        