import cupy
import numpy as np
import logging
from ..utils import round_up
from ..backend import cublasLt
from ..allocator import Allocator
import ctypes

logger = logging.getLogger(__name__)

cublasLt_handles = {}

def get_handle(device):
    global cublasLt_handles
    if device.id not in cublasLt_handles:
        v = cublasLt.cublasLtHandle_t()
        cublasLt.checkCublasStatus(cublasLt.cublasLtCreate( v ))
        cublasLt_handles[device.id] = v
    return cublasLt_handles[device.id]
        
def igemm(allocator : Allocator, a, aT, b, bT, c):
    assert isinstance(a, cupy.ndarray)
    assert isinstance(b, cupy.ndarray)
    assert isinstance(c, cupy.ndarray)
    assert len(a.shape) == 3    # (batch, k, m)
    assert len(b.shape) == 3    # (batch, n, k)
    assert len(c.shape) == 3  # (batch, n, m)
    assert a._c_contiguous
    assert b._c_contiguous
    assert c._c_contiguous
    assert a.device == b.device
    assert b.device == c.device
    device = a.device
    stream = cupy.cuda.get_current_stream()
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

    layout_a, layout_b, layout_c = cublasLt.cublasLtMatrixLayout_t(), cublasLt.cublasLtMatrixLayout_t(), cublasLt.cublasLtMatrixLayout_t()
    cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutCreate(layout_a, cublasLt.CUDA_R_8I, a.shape[2], a.shape[1], a.shape[2]) )
    cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(layout_a, cublasLt.CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, ctypes.byref(ctypes.c_int32(a.shape[0])), ctypes.sizeof(ctypes.c_int32)) )
    cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(layout_a, cublasLt.CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, ctypes.byref(ctypes.c_int64(stride_a)), ctypes.sizeof(ctypes.c_int64)) )
    
    cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutCreate(layout_b, cublasLt.CUDA_R_8I, b.shape[2], b.shape[1], b.shape[2]) )
    cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(layout_b, cublasLt.CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, ctypes.byref(ctypes.c_int32(b.shape[0])), ctypes.sizeof(ctypes.c_int32)) )
    cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(layout_b, cublasLt.CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, ctypes.byref(ctypes.c_int64(stride_b)), ctypes.sizeof(ctypes.c_int64)) )
    
    cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutCreate(layout_c, cublasLt.CUDA_R_32I, c.shape[2], c.shape[1], c.shape[2]) )
    cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(layout_c, cublasLt.CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, ctypes.byref(ctypes.c_int32(c.shape[0])), ctypes.sizeof(ctypes.c_int32)) )
    cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(layout_c, cublasLt.CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, ctypes.byref(ctypes.c_int64(stride_c)), ctypes.sizeof(ctypes.c_int64)) )

    if cc >= 75:
        # use tensor core
        trans_lda = 32 * m
        trans_ldb = 32 * round_up(n, 8)
        trans_ldc = 32 * m
        stride_trans_a = round_up(k, 32) // 32 * trans_lda
        stride_trans_b = round_up(k, 32) // 32 * trans_ldb
        stride_trans_c = ctypes.sizeof(ctypes.c_int32) * round_up(n, 32) // 32 * trans_ldc

        trans_a = allocator.alloc( stride_trans_a * a.shape[0] )
        trans_b = allocator.alloc( stride_trans_b * b.shape[0] )
        trans_c = allocator.alloc( stride_trans_c * c.shape[0] )

        layout_trans_a, layout_trans_b, layout_trans_c = cublasLt.cublasLtMatrixLayout_t(), cublasLt.cublasLtMatrixLayout_t(), cublasLt.cublasLtMatrixLayout_t()
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutCreate(layout_trans_a, cublasLt.CUDA_R_8I, m, k, trans_lda) )
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(layout_trans_a, cublasLt.CUBLASLT_MATRIX_LAYOUT_ORDER, ctypes.byref(ctypes.c_int32(cublasLt.CUBLASLT_ORDER_COL32)), ctypes.sizeof(ctypes.c_int32)) )
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(layout_trans_a, cublasLt.CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, ctypes.byref(ctypes.c_int32(a.shape[0])), ctypes.sizeof(ctypes.c_int32)) )
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(layout_trans_a, cublasLt.CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, ctypes.byref(ctypes.c_int64(stride_trans_a)), ctypes.sizeof(ctypes.c_int64)) )

        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutCreate(layout_trans_b, cublasLt.CUDA_R_8I, n, k, trans_ldb) )
        if cc >= 80:
            cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(layout_trans_b, cublasLt.CUBLASLT_MATRIX_LAYOUT_ORDER, ctypes.byref(ctypes.c_int32(cublasLt.CUBLASLT_ORDER_COL32_2R_4R4)), ctypes.sizeof(ctypes.c_int32)) )
        else:
            cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(layout_trans_b, cublasLt.CUBLASLT_MATRIX_LAYOUT_ORDER, ctypes.byref(ctypes.c_int32(cublasLt.CUBLASLT_ORDER_COL4_4R2_8C)), ctypes.sizeof(ctypes.c_int32)) )
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(layout_trans_b, cublasLt.CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, ctypes.byref(ctypes.c_int32(b.shape[0])), ctypes.sizeof(ctypes.c_int32)) )
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(layout_trans_b, cublasLt.CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, ctypes.byref(ctypes.c_int64(stride_trans_b)), ctypes.sizeof(ctypes.c_int64)) )

        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutCreate(layout_trans_c, cublasLt.CUDA_R_32I, m, n, trans_ldc) )
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(layout_trans_c, cublasLt.CUBLASLT_MATRIX_LAYOUT_ORDER, ctypes.byref(ctypes.c_int32(cublasLt.CUBLASLT_ORDER_COL32)), ctypes.sizeof(ctypes.c_int32)) )
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(layout_trans_c, cublasLt.CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, ctypes.byref(ctypes.c_int32(c.shape[0])), ctypes.sizeof(ctypes.c_int32)) )
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(layout_trans_c, cublasLt.CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, ctypes.byref(ctypes.c_int64(stride_trans_c)), ctypes.sizeof(ctypes.c_int64)) )

        transform_desc_a = cublasLt.cublasLtMatrixTransformDesc_t()
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixTransformDescCreate(transform_desc_a, cublasLt.CUDA_R_32I) )
        if aT:
            cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixTransformDescSetAttribute(
                transform_desc_a,
                cublasLt.CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, 
                ctypes.byref( ctypes.c_int32(cublasLt.CUBLAS_OP_T)),
                ctypes.sizeof(ctypes.c_int32)
            ) )
        

        transform_desc_b = cublasLt.cublasLtMatrixTransformDesc_t()
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixTransformDescCreate(transform_desc_b, cublasLt.CUDA_R_32I) )
        if not bT:
            cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixTransformDescSetAttribute(
                transform_desc_b,
                cublasLt.CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, 
                ctypes.byref( ctypes.c_int32(cublasLt.CUBLAS_OP_T)),
                ctypes.sizeof(ctypes.c_int32)
            ) )
    else:
        round_n = round_up(n, 8)
        round_m = round_up(m, 8)
        round_k = round_up(k, 8)
        stride_trans_a = round_m * round_k
        stride_trans_b = round_n * round_k
        stride_trans_c = round_m * round_n
        trans_a = allocator.alloc( stride_trans_a * a.shape[0] )
        trans_b = allocator.alloc( stride_trans_b * b.shape[0] )
        trans_c = allocator.alloc( ctypes.sizeof(ctypes.c_int32) * stride_trans_c * c.shape[0] )

        layout_trans_a, layout_trans_b, layout_trans_c = cublasLt.cublasLtMatrixLayout_t(), cublasLt.cublasLtMatrixLayout_t(), cublasLt.cublasLtMatrixLayout_t()
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutCreate(layout_trans_a, cublasLt.CUDA_R_8I, m, k, round_m) )
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(layout_trans_a, cublasLt.CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, ctypes.byref(ctypes.c_int32(a.shape[0])), ctypes.sizeof(ctypes.c_int32)) )
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(layout_trans_a, cublasLt.CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, ctypes.byref(ctypes.c_int64(stride_trans_a)), ctypes.sizeof(ctypes.c_int64)) )

        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutCreate(layout_trans_b, cublasLt.CUDA_R_8I, k, n, round_k) )
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(layout_trans_b, cublasLt.CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, ctypes.byref(ctypes.c_int32(b.shape[0])), ctypes.sizeof(ctypes.c_int32)) )
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(layout_trans_b, cublasLt.CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, ctypes.byref(ctypes.c_int64(stride_trans_b)), ctypes.sizeof(ctypes.c_int64)) )

        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutCreate(layout_trans_c, cublasLt.CUDA_R_32I, m, n, round_m) )
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(layout_trans_c, cublasLt.CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, ctypes.byref(ctypes.c_int32(c.shape[0])), ctypes.sizeof(ctypes.c_int32)) )
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(layout_trans_c, cublasLt.CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, ctypes.byref(ctypes.c_int64(stride_trans_c)), ctypes.sizeof(ctypes.c_int64)) )

        transform_desc_a = cublasLt.cublasLtMatrixTransformDesc_t()
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixTransformDescCreate(transform_desc_a, cublasLt.CUDA_R_32I) )
        if aT:
            cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixTransformDescSetAttribute(
                transform_desc_a,
                cublasLt.CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, 
                ctypes.byref( ctypes.c_int32(cublasLt.CUBLAS_OP_T)),
                ctypes.sizeof(ctypes.c_int32)
            ) )
        

        transform_desc_b = cublasLt.cublasLtMatrixTransformDesc_t()
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixTransformDescCreate(transform_desc_b, cublasLt.CUDA_R_32I) )
        if bT:
            cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixTransformDescSetAttribute(
                transform_desc_b,
                cublasLt.CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, 
                ctypes.byref( ctypes.c_int32(cublasLt.CUBLAS_OP_T)),
                ctypes.sizeof(ctypes.c_int32)
            ) )
        
    transform_desc_c = cublasLt.cublasLtMatrixTransformDesc_t()
    cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixTransformDescCreate(transform_desc_c, cublasLt.CUDA_R_32I) )

    cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixTransform(lthandle, transform_desc_a, ctypes.byref(ctypes.c_int32(1)), a.data.ptr, layout_a, ctypes.byref(ctypes.c_int32(0)), 0, 0, trans_a.ptr, layout_trans_a, stream.ptr) )
    cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixTransform(lthandle, transform_desc_b, ctypes.byref(ctypes.c_int32(1)), b.data.ptr, layout_b, ctypes.byref(ctypes.c_int32(0)), 0, 0, trans_b.ptr, layout_trans_b, stream.ptr) )

    cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(layout_trans_a, cublasLt.CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, ctypes.byref(ctypes.c_int32(num_batch)), ctypes.sizeof(ctypes.c_int32)) )
    if a.shape[0] != num_batch:
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(layout_trans_a, cublasLt.CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, ctypes.byref(ctypes.c_int64(0)), ctypes.sizeof(ctypes.c_int64)) )
    cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(layout_trans_b, cublasLt.CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, ctypes.byref(ctypes.c_int32(num_batch)), ctypes.sizeof(ctypes.c_int32)) )
    if b.shape[0] != num_batch:
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatrixLayoutSetAttribute(layout_trans_b, cublasLt.CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, ctypes.byref(ctypes.c_int64(0)), ctypes.sizeof(ctypes.c_int64)) )
    matmul_desc = cublasLt.cublasLtMatmulDesc_t()
    
    if cublasLt.VERSION == 10:
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatmulDescCreate(matmul_desc, cublasLt.CUDA_R_32I) )
    else:
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatmulDescCreate(matmul_desc, cublasLt.CUBLAS_COMPUTE_32I, cublasLt.CUDA_R_32I) )
    if cc >= 75:
        cublasLt.checkCublasStatus( cublasLt.cublasLtMatmulDescSetAttribute(matmul_desc, cublasLt.CUBLASLT_MATMUL_DESC_TRANSB, ctypes.byref( ctypes.c_int32(cublasLt.CUBLAS_OP_T)), ctypes.sizeof(ctypes.c_int32)) )

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
    cublasLt.checkCublasStatus(cublasLt.cublasLtMatrixTransform(lthandle, transform_desc_c, ctypes.byref(ctypes.c_int32(1)), trans_c.ptr, layout_trans_c, ctypes.byref(ctypes.c_int32(0)), 0, 0, c.data.ptr, layout_c, stream.ptr))
    cublasLt.checkCublasStatus(cublasLt.cublasLtMatrixLayoutDestroy(layout_a))
    cublasLt.checkCublasStatus(cublasLt.cublasLtMatrixLayoutDestroy(layout_b))
    cublasLt.checkCublasStatus(cublasLt.cublasLtMatrixLayoutDestroy(layout_c))
    cublasLt.checkCublasStatus(cublasLt.cublasLtMatrixLayoutDestroy(layout_trans_a))
    cublasLt.checkCublasStatus(cublasLt.cublasLtMatrixLayoutDestroy(layout_trans_b))
    cublasLt.checkCublasStatus(cublasLt.cublasLtMatrixLayoutDestroy(layout_trans_c))
    cublasLt.checkCublasStatus(cublasLt.cublasLtMatmulDescDestroy(matmul_desc))
    cublasLt.checkCublasStatus(cublasLt.cublasLtMatrixTransformDescDestroy(transform_desc_a))
    cublasLt.checkCublasStatus(cublasLt.cublasLtMatrixTransformDescDestroy(transform_desc_b))
    cublasLt.checkCublasStatus(cublasLt.cublasLtMatrixTransformDescDestroy(transform_desc_c))
    

        

def round_matrix(x):
    m, n = x.shape
    round_n = round_up(n, 16)
    round_m = round_up(m, 16)
    if round_n == n and round_m == m:
        return x
    
    nw_x = cupy.zeros( (round_m, round_n), dtype=x.dtype )
    nw_x[:m, :n] = x
    logger.info("Round matrix (%d, %d) -> (%d, %d)", m, n, round_m, round_n)
    return nw_x


    

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
    device = a.device
    cublas_version = cublas.getVersion(device.cublas_handle)


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
    
    if out.dtype == cupy.int32:
        type_out = 10   # CUDA_R_32I
        if cublas_version >= 11000:
            ct = cublas.CUBLAS_COMPUTE_32I
        else:
            ct = 10
    else:
        raise TypeError("Unknown type %s for gemm" % out.dtype)

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
    
    one = np.array(1, dtype=np.int32)
    zero = np.array(0, dtype=np.int32)
    cublas.gemmEx(
        device.cublas_handle, 
        transA,
        transB,
        m, n, k,
        one.ctypes.data,
        a.data.ptr,
        type_in,
        lda,
        b.data.ptr,
        type_in,
        ldb,
        zero.ctypes.data,
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