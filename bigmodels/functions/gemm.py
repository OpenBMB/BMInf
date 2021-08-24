from ..backend import cublas
import cupy
import numpy as np
import logging
from ..scalar import get_scalar_ptr
from ..utils import round_up

logger = logging.getLogger(__name__)


def round_matrix(x):
    m, n = x.shape
    round_n = round_up(n, 16)
    round_m = round_up(m, 16)
    if round_n == n and round_m == m:
        return x
    
    nw_x = cupy.zeros( (round_m, round_n), dtype=x.dtype )
    nw_x[:m, :n] = x
    return nw_x

def igemm(a, aT, b, bT, out):
    
    round_a = round_matrix(a)
    round_b = round_matrix(b)
    round_out = round_matrix(out)

    _igemm(round_a, aT, round_b, bT, round_out)

    if round_out.shape != out.shape:
        m, n = out.shape
        out[:, :] = round_out[:m, :n]
    

def _igemm(a, aT, b, bT, out):
    assert isinstance(a, cupy.ndarray)
    assert isinstance(b, cupy.ndarray)
    assert isinstance(out, cupy.ndarray)
    assert len(a.shape) == 2
    assert len(b.shape) == 2
    assert len(out.shape) == 2
    assert a._c_contiguous
    assert b._c_contiguous
    assert out._c_contiguous
    assert a.device == b.device
    assert b.device == out.device
    
    if aT:
        k1, m = a.shape
        transA = cublas.CUBLAS_OP_N
    else:
        m, k1 = a.shape
        transA = cublas.CUBLAS_OP_T

    if bT:
        n, k2 = b.shape
        transB = cublas.CUBLAS_OP_N
    else:
        k2, n = b.shape
        transB = cublas.CUBLAS_OP_T
    assert k1 == k2
    k = k1
    assert a.dtype == b.dtype
    assert out.shape[0] == n
    assert out.shape[1] == m
    if a.dtype == cupy.int8:
        type_in = cupy.cuda.runtime.CUDA_R_8I
    else:
        raise TypeError("Unknown type %s for gemm" % a.dtype)
    
    if out.dtype == cupy.float32:
        type_out = cupy.cuda.runtime.CUDA_R_32F
        ct = cublas.CUBLAS_COMPUTE_32F
    elif out.dtype == cupy.int32:
        type_out = 10
        ct = cublas.CUBLAS_COMPUTE_32I
    else:
        raise TypeError("Unknown type %s for gemm" % out.dtype)

    device = a.device

    lda = m if aT else k
    ldb = k if bT else n
    ldc = m

    if m % 8 != 0:
        logger.warning("[WARN] igemm m % 8 != 0")
    if k % 8 != 0:
        logger.warning("[WARN] igemm k % 8 != 0")
    if not (bT or n % 8 == 0):
        logger.warning("[WARN] igemm n % 8 != 0 and bT == False")
    if a.data.ptr % 16 != 0:
        logger.warning("[WARN] igemm intptr_t(A) % 16 != 0")
    if b.data.ptr % 16 != 0:
        logger.warning("[WARN] igemm intptr_t(B) % 16 != 0")
    if out.data.ptr % 16 != 0:
        logger.warning("[WARN] igemm intptr_t(C) % 16 != 0")
    if lda % 16 != 0:
        logger.warning("[WARN] igemm lda % 16 != 0")
    if ldb % 16 != 0:
        logger.warning("[WARN] igemm ldb % 16 != 0")
    if ldc % 16 != 0:
        logger.warning("[WARN] igemm ldc % 16 != 0")
    cublas.gemmEx(
        device.cublas_handle, 
        transA,
        transB,
        m, n, k,
        get_scalar_ptr(1),
        a.data.ptr,
        type_in,
        lda,
        b.data.ptr,
        type_in,
        ldb,
        get_scalar_ptr(0),
        out.data.ptr,
        type_out,
        ldc,
        ct,
        cublas.CUBLAS_GEMM_DEFAULT
    )

def sgemmBatched(a, aT, b, bT, out):
    assert isinstance(a, cupy.ndarray)
    assert isinstance(b, cupy.ndarray)
    assert isinstance(out, cupy.ndarray)
    assert len(a.shape) == 3
    assert len(b.shape) == 3
    assert len(out.shape) == 3
    assert a._c_contiguous
    assert b._c_contiguous
    assert out._c_contiguous
    assert a.device == b.device
    assert b.device == out.device
    assert a.dtype == b.dtype
    assert out.dtype == b.dtype
    assert a.dtype == cupy.float32
    assert b.dtype == cupy.float32
    assert out.dtype == cupy.float32
    
    if aT:
        batch1, k1, m = a.shape
        transA = cublas.CUBLAS_OP_N
    else:
        batch1, m, k1 = a.shape
        transA = cublas.CUBLAS_OP_T

    if bT:
        batch2, n, k2 = b.shape
        transB = cublas.CUBLAS_OP_N
    else:
        batch2, k2, n = b.shape
        transB = cublas.CUBLAS_OP_T
    
    assert k1 == k2
    assert batch1 == batch2

    batch = batch1
    k = k1
    
    assert out.shape[0] == batch
    assert out.shape[1] == n
    assert out.shape[2] == m

    device = a.device

    lda = m if aT else k
    ldb = k if bT else n
    ldc = m

    itemsize = cupy.dtype(cupy.float32).itemsize
    stride_a = a._strides[0] // itemsize
    stride_b = b._strides[0] // itemsize
    stride_c = out._strides[0] // itemsize

    if m % 8 != 0 and m > 1:
        logger.warning("[WARN] gemm m % 8 != 0")
    if k % 8 != 0 and k > 1:
        logger.warning("[WARN] gemm k % 8 != 0")
    if not (bT or n % 8 == 0):
        logger.warning("[WARN] gemm n % 8 != 0 and bT == False")
    if (stride_a * itemsize) % 16 != 0:
        logger.warning("[WARN] gemm stride_a % 16 != 0")
    if (stride_b * itemsize) % 16 != 0:
        logger.warning("[WARN] gemm stride_b % 16 != 0")
    if (stride_c * itemsize) % 16 != 0:
        logger.warning("[WARN] gemm stride_c % 16 != 0")
    if a.data.ptr % 16 != 0:
        logger.warning("[WARN] gemm intptr_t(A) % 16 != 0")
    if b.data.ptr % 16 != 0:
        logger.warning("[WARN] gemm intptr_t(B) % 16 != 0")
    if out.data.ptr % 16 != 0:
        logger.warning("[WARN] gemm intptr_t(C) % 16 != 0")
    if (lda * itemsize) % 16 != 0 and lda > 1:
        logger.warning("[WARN] gemm lda % 16 != 0")
    if (ldb * itemsize) % 16 != 0 and ldb > 1:
        logger.warning("[WARN] gemm ldb % 16 != 0")
    if (ldc * itemsize) % 16 != 0 and ldc > 1:
        logger.warning("[WARN] gemm ldc % 16 != 0")

    one = np.array(1, dtype=np.float32)
    zero = np.array(0, dtype=np.float32)
    cublas.sgemmStridedBatched(
        device.cublas_handle, 
        transA,
        transB,
        m, n, k,
        one.ctypes.data,
        a.data.ptr,
        lda,
        stride_a,
        b.data.ptr,
        ldb,
        stride_b,
        zero.ctypes.data,
        out.data.ptr,
        ldc,
        stride_c,
        batch
    )
