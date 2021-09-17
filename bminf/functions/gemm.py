import cupy
import numpy as np
import logging
from ..utils import round_up
from ..utils.cache import HandleCache
from ..backend import cublasLt
from ..allocator import Allocator
import ctypes

logger = logging.getLogger(__name__)

cublasLt_handles = {}

class LayoutCache(HandleCache):
    def create(self, rt, m, n, ld, order, batch_count, batch_offset):
        ret = cublasLt.cublasLtMatrixLayout_t()
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutCreate(ret, rt, m, n, ld) )
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(ret, cublasLt.CUBLASLT_MATRIX_LAYOUT_ORDER, ctypes.byref(ctypes.c_int32(order)), ctypes.sizeof(ctypes.c_int32)) )
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(ret, cublasLt.CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, ctypes.byref(ctypes.c_int32(batch_count)), ctypes.sizeof(ctypes.c_int32)) )
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(ret, cublasLt.CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, ctypes.byref(ctypes.c_int64(batch_offset)), ctypes.sizeof(ctypes.c_int64)) )
        return ret
    
    def release(self, x):
        cublasLt.checkCublasStatus(cublasLt.cublasLtMatrixLayoutDestroy(x))

class TransformCache(HandleCache):
    def create(self, rt, aT):
        ret = cublasLt.cublasLtMatrixTransformDesc_t()
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixTransformDescCreate(ret, rt) )
        if aT:
            cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixTransformDescSetAttribute(
                ret,
                cublasLt.CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, 
                ctypes.byref( ctypes.c_int32(cublasLt.CUBLAS_OP_T)),
                ctypes.sizeof(ctypes.c_int32)
            ) )
        return ret

    def release(self, x):
        cublasLt.checkCublasStatus(cublasLt.cublasLtMatrixTransformDescDestroy(x))

class MatmulCache(HandleCache):
    def create(self, rt, ct, aT, bT):
        ret = cublasLt.cublasLtMatmulDesc_t()
        if cublasLt.VERSION == 10:
            cublasLt.checkCublasStatus( cublasLt.cublasLtMatmulDescCreate(ret, rt) )
        else:
            cublasLt.checkCublasStatus( cublasLt.cublasLtMatmulDescCreate(ret, ct, rt) )
        if aT:
            cublasLt.checkCublasStatus( cublasLt.cublasLtMatmulDescSetAttribute(ret, cublasLt.CUBLASLT_MATMUL_DESC_TRANSA, ctypes.byref( ctypes.c_int32(cublasLt.CUBLAS_OP_T)), ctypes.sizeof(ctypes.c_int32)) )
        if bT:
            cublasLt.checkCublasStatus( cublasLt.cublasLtMatmulDescSetAttribute(ret, cublasLt.CUBLASLT_MATMUL_DESC_TRANSB, ctypes.byref( ctypes.c_int32(cublasLt.CUBLAS_OP_T)), ctypes.sizeof(ctypes.c_int32)) )
        return ret
    
    def release(self, x):
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatmulDescDestroy(x) )

layout_cache = LayoutCache(32)
transform_cache = TransformCache()
matmul_cache = MatmulCache()

def get_handle(device):
    global cublasLt_handles
    if device.id not in cublasLt_handles:
        v = cublasLt.cublasLtHandle_t()
        cublasLt.checkCublasStatus(cublasLt.cublasLtCreate( v ))
        cublasLt_handles[device.id] = v
    return cublasLt_handles[device.id]
        
def round_matrix(allocator : Allocator, x, d, stream):
    batch, m, n = x.shape
    round_n = round_up(n, d)
    round_m = round_up(m, d)
    if round_n == n and round_m == m:
        return x
    nw_x = allocator.alloc_array((batch, round_m, round_n), dtype=x.dtype)
    cupy.cuda.runtime.memsetAsync(nw_x.data.ptr, 0, nw_x.nbytes, stream.ptr)
    nw_x[:, :m, :n] = x
    logger.info("Round matrix (%d, %d) -> (%d, %d)", m, n, round_m, round_n)
    return nw_x


def igemm(allocator : Allocator, a, aT, b, bT, c):
    device = a.device
    stream = cupy.cuda.get_current_stream()
    _igemm(allocator, a, aT, b, bT, c, device, stream)
    return c

def _igemm(allocator : Allocator, a, aT, b, bT, c, device, stream):
    assert isinstance(a, cupy.ndarray)
    assert isinstance(b, cupy.ndarray)
    assert isinstance(c, cupy.ndarray)
    assert len(a.shape) == 3    # (batch, k, m)
    assert len(b.shape) == 3    # (batch, n, k)
    assert len(c.shape) == 3  # (batch, n, m)
    assert a._c_contiguous
    assert b._c_contiguous
    assert c._c_contiguous
    assert a.device == device
    assert b.device == device
    assert c.device == device
    lthandle = get_handle(device)

    num_batch = 1
    if a.shape[0] > 1 and b.shape[0] > 1:
        assert a.shape[0] == b.shape[0]
        num_batch = a.shape[0]
    elif a.shape[0] > 1:
        num_batch = a.shape[0]
    else:
        num_batch = b.shape[0]
    
    if a.shape[0] == 1:
        stride_a = 0
    else:
        stride_a = a.shape[1] * a.shape[2]
    if b.shape[0] == 1:
        stride_b = 0
    else:
        stride_b = b.shape[1] * b.shape[2]

    
    if aT:
        m, k1 = a.shape[1:]
    else:
        k1, m = a.shape[1:]

    if bT:
        k2, n = b.shape[1:]
    else:
        n, k2 = b.shape[1:]

    assert k1 == k2
    k = k1
    assert c.shape == (num_batch, n, m)
    stride_c = n * m

    ## compute capability:
    #  Ampere >= 80
    #  Turing >= 75
    cc = int(device.compute_capability)

    v1 = ctypes.c_int(1)
    v0 = ctypes.c_int(0)

    layout_a = layout_cache(cublasLt.CUDA_R_8I, a.shape[2], a.shape[1], a.shape[2], cublasLt.CUBLASLT_ORDER_COL, a.shape[0], stride_a)
    layout_b = layout_cache(cublasLt.CUDA_R_8I, b.shape[2], b.shape[1], b.shape[2], cublasLt.CUBLASLT_ORDER_COL, b.shape[0], stride_b)
    layout_c = layout_cache(cublasLt.CUDA_R_32I, c.shape[2], c.shape[1], c.shape[2], cublasLt.CUBLASLT_ORDER_COL, c.shape[0], stride_c)

    if cc >= 75:
        # use tensor core
        trans_lda = 32 * m
        if cc >= 80:
            trans_ldb = 32 * round_up(n, 32)
        else:
            trans_ldb = 32 * round_up(n, 8)
        trans_ldc = 32 * m
        stride_trans_a = round_up(k, 32) // 32 * trans_lda
        stride_trans_b = round_up(k, 32) // 32 * trans_ldb
        stride_trans_c = round_up(n, 32) // 32 * trans_ldc

        trans_a = allocator.alloc( stride_trans_a * a.shape[0] )
        trans_b = allocator.alloc( stride_trans_b * b.shape[0] )
        trans_c = allocator.alloc( ctypes.sizeof(ctypes.c_int32) * stride_trans_c * c.shape[0] )

        layout_trans_a = layout_cache(cublasLt.CUDA_R_8I, m, k, trans_lda, cublasLt.CUBLASLT_ORDER_COL32, a.shape[0], stride_trans_a)
        if cc >= 80:
            layout_trans_b = layout_cache(cublasLt.CUDA_R_8I, n, k, trans_ldb, cublasLt.CUBLASLT_ORDER_COL32_2R_4R4, b.shape[0], stride_trans_b)
        else:
            layout_trans_b = layout_cache(cublasLt.CUDA_R_8I, n, k, trans_ldb, cublasLt.CUBLASLT_ORDER_COL4_4R2_8C, b.shape[0], stride_trans_b)
        layout_trans_c = layout_cache(cublasLt.CUDA_R_32I, m, n, trans_ldc, cublasLt.CUBLASLT_ORDER_COL32, num_batch, stride_trans_c)

        transform_desc_a = transform_cache(cublasLt.CUDA_R_32I, aT)

        transform_desc_b = transform_cache(cublasLt.CUDA_R_32I, not bT)

        transform_desc_c = transform_cache(cublasLt.CUDA_R_32I, False)

        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixTransform(lthandle, transform_desc_a, ctypes.byref(v1), a.data.ptr, layout_a, ctypes.byref(v0), 0, 0, trans_a.ptr, layout_trans_a, stream.ptr) )
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixTransform(lthandle, transform_desc_b, ctypes.byref(v1), b.data.ptr, layout_b, ctypes.byref(v0), 0, 0, trans_b.ptr, layout_trans_b, stream.ptr) )

        if a.shape[0] != num_batch:
            layout_trans_a = layout_cache(cublasLt.CUDA_R_8I, m, k, trans_lda, cublasLt.CUBLASLT_ORDER_COL32, num_batch, 0)
        if b.shape[0] != num_batch:
            if cc >= 80:
                layout_trans_b = layout_cache(cublasLt.CUDA_R_8I, n, k, trans_ldb, cublasLt.CUBLASLT_ORDER_COL32_2R_4R4, num_batch, 0)
            else:
                layout_trans_b = layout_cache(cublasLt.CUDA_R_8I, n, k, trans_ldb, cublasLt.CUBLASLT_ORDER_COL4_4R2_8C, num_batch, 0)

        matmul_desc = matmul_cache(cublasLt.CUDA_R_32I, cublasLt.CUBLAS_COMPUTE_32I, False, True)

        cublasLt.checkCublasStatus( cublasLt.cublasLtMatmul(
            lthandle, 
            matmul_desc, 
            ctypes.byref(ctypes.c_int32(1)), 
            trans_a.ptr, 
            layout_trans_a, 
            trans_b.ptr, 
            layout_trans_b, 
            ctypes.byref(ctypes.c_int32(0)), 
            trans_c.ptr,
            layout_trans_c,
            trans_c.ptr,
            layout_trans_c,
            0,
            0,
            0,
            stream.ptr
        ))
        cublasLt.checkCublasStatus(cublasLt.cublasLtMatrixTransform(lthandle, transform_desc_c, ctypes.byref(v1), trans_c.ptr, layout_trans_c, ctypes.byref(v0), 0, 0, c.data.ptr, layout_c, stream.ptr))
    else:
        trans_lda = round_up(a.shape[2], 4)
        trans_ldb = round_up(b.shape[2], 4)
        trans_ldc = round_up(c.shape[2], 4)
        stride_trans_a = trans_lda * a.shape[1]
        stride_trans_b = trans_ldb * b.shape[1]
        stride_trans_c = trans_ldc * c.shape[1]

        transform_desc = transform_cache(cublasLt.CUDA_R_32I, False)

        if a.shape[2] == trans_lda:
            trans_a = a.data
            layout_trans_a = layout_a
        else:
            trans_a = allocator.alloc( stride_trans_a * a.shape[0] )
            layout_trans_a = layout_cache(cublasLt.CUDA_R_8I, a.shape[2], a.shape[1], trans_lda, cublasLt.CUBLASLT_ORDER_COL, a.shape[0], stride_trans_a)
            cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixTransform(lthandle, transform_desc, ctypes.byref(v1), a.data.ptr, layout_a, ctypes.byref(v0), 0, 0, trans_a.ptr, layout_trans_a, stream.ptr) )
        if b.shape[2] == trans_ldb:
            trans_b = b.data
            layout_trans_b = layout_b
        else:
            trans_b = allocator.alloc( stride_trans_b * b.shape[0] )
            layout_trans_b = layout_cache(cublasLt.CUDA_R_8I, b.shape[2], b.shape[1], trans_ldb, cublasLt.CUBLASLT_ORDER_COL, b.shape[0], stride_trans_b)
            cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixTransform(lthandle, transform_desc, ctypes.byref(v1), b.data.ptr, layout_b, ctypes.byref(v0), 0, 0, trans_b.ptr, layout_trans_b, stream.ptr) )
        if c.shape[2] == trans_ldc:
            trans_c = c.data
            layout_trans_c = layout_c
        else:
            trans_c = allocator.alloc( ctypes.sizeof(ctypes.c_int32) * stride_trans_c * c.shape[0] )
            layout_trans_c = layout_cache(cublasLt.CUDA_R_32I, c.shape[2], c.shape[1], trans_ldc, cublasLt.CUBLASLT_ORDER_COL, num_batch, stride_trans_c)

        matmul_desc = matmul_cache(cublasLt.CUDA_R_32I, cublasLt.CUBLAS_COMPUTE_32I, aT, bT)
        
        if a.shape[0] != num_batch:
            layout_trans_a = layout_cache(cublasLt.CUDA_R_8I, a.shape[2], a.shape[1], trans_lda, cublasLt.CUBLASLT_ORDER_COL, num_batch, 0)
        if b.shape[0] != num_batch:
            layout_trans_b = layout_cache(cublasLt.CUDA_R_8I, b.shape[2], b.shape[1], trans_ldb, cublasLt.CUBLASLT_ORDER_COL, num_batch, 0)
        
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatmul(
            lthandle, 
            matmul_desc, 
            ctypes.byref(ctypes.c_int32(1)), 
            trans_a.ptr, 
            layout_trans_a, 
            trans_b.ptr, 
            layout_trans_b, 
            ctypes.byref(ctypes.c_int32(0)), 
            trans_c.ptr,
            layout_trans_c,
            trans_c.ptr,
            layout_trans_c,
            0,
            0,
            0,
            stream.ptr
        ))
        if c.shape[2] != trans_ldc:
            cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixTransform(lthandle, transform_desc, ctypes.byref(v1), trans_c.ptr, layout_trans_c, ctypes.byref(v0), 0, 0, c.data.ptr, layout_c, stream.ptr) )



def fgemm(allocator : Allocator, a, aT, b, bT, c):
    device = a.device
    stream = cupy.cuda.get_current_stream()
    return _fgemm(a, aT, b, bT, c, device, stream)

def _fgemm(a, aT, b, bT, c, device, stream):
    assert isinstance(a, cupy.ndarray)
    assert isinstance(b, cupy.ndarray)
    assert isinstance(c, cupy.ndarray)
    assert len(a.shape) == 3
    assert len(b.shape) == 3
    assert len(c.shape) == 3
    assert a._c_contiguous
    assert b._c_contiguous
    assert c._c_contiguous
    assert a.device == device
    assert b.device == device
    assert c.device == device
    dtype = a.dtype 
    assert a.dtype == dtype
    assert b.dtype == dtype
    assert c.dtype == dtype

    if aT:
        batch1, m, k1 = a.shape
    else:
        batch1, k1, m = a.shape

    if bT:
        batch2, k2, n = b.shape
    else:
        batch2, n, k2 = b.shape
    
    assert k1 == k2

    if batch1 == 1:
        batch = batch2
    elif batch2 == 1:
        batch = batch1
    elif batch1 == batch2:
        batch = batch1
    else:
        raise ValueError("batch A(%d) != batch B(%d)" % (batch1, batch2))

    assert c.shape == (batch,n, m)
    
    cc = int(device.compute_capability)

    if dtype == cupy.float16:
        rt_type = cublasLt.CUDA_R_16F
        ct_type = cublasLt.CUBLAS_COMPUTE_16F
    elif dtype == cupy.float32:
        rt_type = cublasLt.CUDA_R_32F
        ct_type = cublasLt.CUBLAS_COMPUTE_32F
    else:
        raise TypeError("Unsupported type %s" % dtype)

    ltHandle = get_handle(device)


    if batch1 == 1:
        stride_a = 0
    else:
        stride_a = a.shape[1] * a.shape[2]
    if batch2 == 1:
        stride_b = 0
    else:
        stride_b = b.shape[1] * b.shape[2]
    stride_c = c.shape[1] * c.shape[2]

    layout_A = layout_cache(rt_type, a.shape[2], a.shape[1], a.shape[2], cublasLt.CUBLASLT_ORDER_COL, batch, stride_a)
    layout_B = layout_cache(rt_type, b.shape[2], b.shape[1], b.shape[2], cublasLt.CUBLASLT_ORDER_COL, batch, stride_b)
    layout_C = layout_cache(rt_type, c.shape[2], c.shape[1], c.shape[2], cublasLt.CUBLASLT_ORDER_COL, batch, stride_c)

    if cc >= 70:
        # has fp16 tensor core
        matmul_desc = matmul_cache(rt_type, ct_type, aT, bT)
    else:
        # fp32 is faster
        matmul_desc = matmul_cache(cublasLt.CUDA_R_32F, cublasLt.CUBLAS_COMPUTE_32F, aT, bT)

    if dtype == cupy.float32 or cc < 70:
        alpha = ctypes.byref(ctypes.c_float(1))
        beta = ctypes.byref(ctypes.c_float(0))
    elif dtype == cupy.float16:
        tmp = np.array([1, 0], dtype=np.float16)
        alpha = tmp[0:].ctypes.data
        beta = tmp[1:].ctypes.data
    else:
        raise NotImplementedError()

    cublasLt.checkCublasStatus( cublasLt.cublasLtMatmul(
        ltHandle, 
        matmul_desc, 
        alpha, 
        a.data.ptr, 
        layout_A, 
        b.data.ptr, 
        layout_B, 
        beta, 
        c.data.ptr,
        layout_C,
        c.data.ptr,
        layout_C,
        0,
        0,
        0,
        stream.ptr
    ))

