from ..parameter import Parameter
import logging
import struct
import cupy

logger = logging.getLogger(__name__)

def load_dtype(fp):
    v = struct.unpack("B", fp.read(1))[0]
    if v == 0:
        return cupy.int8
    elif v == 1:
        return cupy.float16
    elif v == 2:
        return cupy.float32
    else:
        raise TypeError("Unknown dtype %d" % v)

def load_string(fp):
    size = struct.unpack("I", fp.read(4))[0]
    return fp.read(size).decode("utf-8")

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
    if v == cupy.int8:
        sv = 0
    elif v == cupy.float16:
        sv = 1
    elif v == cupy.float32:
        sv = 2
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

def dump_parameter(shape, data, data_dtype, fp):
    dump_tuple(shape, fp)
    fp.write( struct.pack("I", len(data)) )
    dump_dtype(data_dtype, fp)
    fp.write(data)


class Layer:
    def load(self, fp):
        self.__ensure_variables()
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
        self.__ensure_variables()
        logger.debug("Parameter Dumper [%s]: size %s", self.__class__.__name__, self.nbytes)
        num_parameters, num_sub_layers = len(self._parameters), len(self._sub_layers)
        fp.write( struct.pack("II", num_parameters, num_sub_layers) )
        for kw, val in self._parameters.items():
            dump_string(kw, fp)
            if val.data is not None:
                dump_parameter(val.shape, val.data, val.dtype, fp)
            else:
                raise RuntimeError("Paraemter %s has no data" % kw)
        for kw, val in self._sub_layers.items():
            dump_string(kw, fp)
            val.dump(fp)
    
    def __ensure_variables(self):
        if not hasattr(self, "_parameters"):
            self._parameters = {}
        if not hasattr(self, "_parameter_bytes"):
            self._parameter_bytes = 0
        if not hasattr(self, "_sub_layers"):
            self._sub_layers = {}
    
    @property
    def nbytes(self):
        self.__ensure_variables()
        return self._parameter_bytes


    def _add_parameter(self, name, v : Parameter):
        self.__ensure_variables()

        self._parameters[name] = v
        self._parameter_bytes += v.nbytes

    def _add_sublayer(self, name, layer : 'Layer'):
        self.__ensure_variables()

        self._sub_layers[name] = layer
        self._parameter_bytes += layer.nbytes

    def __setattr__(self, name: str, value):
        if not name.startswith("_"):
            if isinstance(value, Parameter):
                self._add_parameter(name, value)
            elif isinstance(value, Layer):
                self._add_sublayer(name, value)
        super().__setattr__(name, value)

    def info(self, prefix="") -> str:
        self.__ensure_variables()

        ret = ""
        for name, param in self._parameters.items():
            ret += prefix + f"Parameter {name}, size: {param.nbytes} bytes\n"
        for name, layer in self._sub_layers.items():
            ret += prefix + f"Layer {name}, size: {layer.nbytes} bytes\n"
            ret += layer.info(prefix + "    ")
        return ret
    
    def to_device(self, allocator, load_stream):
        self.__ensure_variables()

        for param in self._parameters.values():
            param.to_device(allocator, load_stream)
        for layer in self._sub_layers.values():
            layer.to_device(allocator, load_stream)

    def _remove_data(self):
        for param in self._parameters.values():
            param._remove_data()
        for layer in self._sub_layers.values():
            layer._remove_data()
    
    def _try_pinned(self):
        for param in self._parameters.values():
            param._try_pinned()
        for layer in self._sub_layers.values():
            layer._try_pinned()