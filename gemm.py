import cupy
import numpy as np
import time

import logging
logger = logging.getLogger(__name__)


def gemm(a, aT, b, bT, out):
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
        transA = cupy.cuda.cublas.CUBLAS_OP_N
    else:
        m, k1 = a.shape
        transA = cupy.cuda.cublas.CUBLAS_OP_T

    if bT:
        n, k2 = b.shape
        transB = cupy.cuda.cublas.CUBLAS_OP_N
    else:
        k2, n = b.shape
        transB = cupy.cuda.cublas.CUBLAS_OP_T
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
        ct = cupy.cuda.cublas.CUBLAS_COMPUTE_32F
    elif out.dtype == cupy.int32:
        type_out = 10
        ct = cupy.cuda.cublas.CUBLAS_COMPUTE_32I
    else:
        raise TypeError("Unknown type %s for gemm" % out.dtype)
    
    valpha = np.array(1, dtype=out.dtype)
    vbeta = np.array(0, dtype=out.dtype)

    device = a.device

    lda = m if aT else k
    ldb = k if bT else n
    ldc = m

    if m % 8 != 0:
        logger.warning("[WARN] gemm m % 8 != 0")
    if k % 8 != 0:
        logger.warning("[WARN] gemm k % 8 != 0")
    if not (bT or n % 8 == 0):
        logger.warning("[WARN] gemm n % 8 != 0 and bT == False")
    if a.data.ptr % 16 != 0:
        logger.warning("[WARN] gemm intptr_t(A) % 16 != 0")
    if b.data.ptr % 16 != 0:
        logger.warning("[WARN] gemm intptr_t(B) % 16 != 0")
    if out.data.ptr % 16 != 0:
        logger.warning("[WARN] gemm intptr_t(C) % 16 != 0")
    if lda % 16 != 0:
        logger.warning("[WARN] gemm lda % 16 != 0")
    if ldb % 16 != 0:
        logger.warning("[WARN] gemm ldb % 16 != 0")
    if ldc % 16 != 0:
        logger.warning("[WARN] gemm ldc % 16 != 0")

    cupy.cuda.cublas.gemmEx(
        device.cublas_handle, 
        transA,
        transB,
        m, n, k,
        valpha.ctypes.data,
        a.data.ptr,
        type_in,
        lda,
        b.data.ptr,
        type_in,
        ldb,
        vbeta.ctypes.data,
        out.data.ptr,
        type_out,
        ldc,
        ct,
        cupy.cuda.cublas.CUBLAS_GEMM_DEFAULT
    )

elementwise_copy_scale = cupy.core.create_ufunc('bms_scaled_copy', ('bf->f', 'be->e', 'ef->f', 'if->f'), 'out0 = in0 * in1')

def test():
    a = cupy.random.randn(512, 4096).astype(cupy.float32)
    b = cupy.random.randn(4096, 10240).astype(cupy.float32)

    scale_a = cupy.abs(a).max() / 120
    scale_b = cupy.abs(b).max() / 120

    qa = (a / scale_a).round().astype(cupy.int8)
    qb = (b / scale_b).round().astype(cupy.int8)

    device = qa.device

    qc = cupy.ndarray((10240, 512), dtype=cupy.int32)
    qd = cupy.ndarray((10240, 512), dtype=cupy.float32)
    loops = 640

    device.synchronize()
    print("Start")
    st = time.perf_counter()

    for i in range(loops):
        gemm(qa, False, qb, False, qc)
        elementwise_copy_scale(qc, cupy.float32(1.0), qd)
    qc = qd
    print("Wait")
    device.synchronize()

    total_time = time.perf_counter() - st
    print("total: %lf, loop: %lf" % (total_time, total_time / loops))

    ans = cupy.matmul(a, b).T
    out = qc * scale_a * scale_b

    diff = cupy.abs(ans - out).max()
    print(diff / ans.std())

test()