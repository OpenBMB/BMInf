import cupy
import numpy as np
import time

def test():
    a = cupy.random.randn(4096, 10240).astype(cupy.float32)
    b = cupy.random.randn(4096, 10240).astype(cupy.float32)

    scale_a = cupy.abs(a).max() / 120
    scale_b = cupy.abs(b).max() / 120

    qa = cupy.round(a / scale_a).astype(cupy.int8)
    qb = cupy.round(b / scale_b).astype(cupy.int8)

    v1 = np.array(1, dtype=np.int32).ctypes.data
    v0 = np.array(0, dtype=np.int32).ctypes.data

    device = qa.device

    qc = cupy.ndarray((4096, 4096), dtype=cupy.int32)

    loops = 100

    device.synchronize()
    print("Start")
    st = time.perf_counter()

    for _ in range(loops):
        cupy.cublas.cublas.gemmEx(
            device.cublas_handle, 
            cupy.cublas.cublas.CUBLAS_OP_T, 
            cupy.cublas.cublas.CUBLAS_OP_N,
            4096,
            4096,
            10240,
            v1,  # int32
            qa.data.ptr,
            cupy.cuda.runtime.CUDA_R_8I,
            10240,
            qb.data.ptr,
            cupy.cuda.runtime.CUDA_R_8I,
            4096,
            v0,
            qc.data.ptr,
            10,
            4096,
            cupy.cublas.cublas.CUBLAS_COMPUTE_32I,
            cupy.cublas.cublas.CUBLAS_GEMM_DEFAULT
        )
    print("Wait")
    device.synchronize()

    total_time = time.perf_counter() - st
    print("total: %lf, loop: %lf" % (total_time, total_time / loops))

    ans = cupy.matmul(a, b.T)

    out = qc.astype(cupy.float32) * scale_a * scale_b

    diff = cupy.abs(ans - out).max()
    print(diff)

test()