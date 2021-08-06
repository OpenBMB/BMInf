from cupy._statistics.order import quantile
from .base import Layer
from ..parameter import Parameter
import cupy
from ..allocator import Allocator
from ..tensor import Tensor
from ..functions.quantization import dequantize, quantize

l2norm_kernel = cupy.ReductionKernel(
    'T x',
    'T y',
    'x * x',
    'a + b',
    'y = sqrt(_type_reduce(_in_ind.size() / _out_ind.size()) / a)',
    '0.000001',
    'bms_l2norm'
)

scaled_add = cupy._core.create_ufunc(
    "bms_scaled_add",
    ('efe->e', 'fff->f'),
    'out0 = in0 + in1 * in2'
)

class LayerNorm(Layer):
    def __init__(self, dim_in):
        self.weight = Parameter((dim_in,), dtype=cupy.float32)
        

    def forward(self, allocator : Allocator, x : Tensor, inplace = True):
        # forward inplace        
        value = x.value
        scale = x.scale

        orig_type = value.dtype

        if orig_type == cupy.int8:
            # dequantize
            dqv = allocator.alloc_array(value.shape, dtype=cupy.float32)
            dequantize(value, scale, dqv)
        else:
            dqv = value

        out = allocator.alloc_array(dqv.shape[:-1] + (1,), dqv.dtype)
        l2norm_kernel(dqv, axis=-1, keepdims=True, out=out)

        if inplace or orig_type == cupy.int8:
            dqv *= out
        else:
            dqv = dqv * out # copied here

        del out
        
        dqv *= self.weight.value

        if orig_type == cupy.int8:
            if inplace:
                out = x.value
            else:
                out = allocator.alloc_array(x.value.shape, orig_type)

            quantize(dqv, out)
            value = out
        else:
            value = dqv
            

        return Tensor(value, scale)