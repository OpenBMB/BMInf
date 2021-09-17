from ..allocator import Allocator
import cupy
from ..backend import create_reduction_func, create_ufunc

_min_max_preamble = '''
template <typename T>
struct min_max_st{
    T value;
    int index;
    __device__ min_max_st() : index(-1) { }
    __device__ min_max_st(T v) : value(v), index(0) { }
    __device__ min_max_st(T v, int i) : value(v), index(i) { }
};
template <typename T>
__device__ min_max_st<T> my_max(
        const min_max_st<T>& a, const min_max_st<T>& b) {
    if (a.index == -1) return b;
    if (b.index == -1) return a;
    return min_max_st<T>(max(abs(a.value), abs(b.value)));
}
'''

quantize_scale_kernel = create_reduction_func(
    'bms_quantize_scale',
    ('e->e', 'f->e', 'e->f', 'f->f'),
    ('min_max_st<type_in0_raw>(in0)', 'my_max(a, b)', 'out0 = abs(a.value) / 127',
     'min_max_st<type_in0_raw>'), None, _min_max_preamble)

quantize_copy_half = create_ufunc('bms_quantize_copy_half', ('ee->b', 'fe->b', 'ff->b'), 'out0 = nearbyintf(half(in0 / in1))')
quantize_copy_float = create_ufunc('bms_quantize_copy_float', ('ee->b', 'fe->b', 'ff->b'), 'out0 = nearbyintf(float(in0 / in1))')

def quantize(x : cupy.ndarray, out : cupy.ndarray, scale : cupy.ndarray, axis=-1):
    assert x.dtype == cupy.float16 or x.dtype == cupy.float32
    assert x.shape == out.shape
    
    if axis < 0:
        axis += len(x.shape)
    
    assert scale.dtype == cupy.float16
    assert scale.shape == x.shape[:axis] + (1,) + x.shape[axis + 1:]

    quantize_scale_kernel(x, axis=axis, keepdims=True, out=scale) # scale on gpu
    quantize_copy_half(x, scale, out=out)
