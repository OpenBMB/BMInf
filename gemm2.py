import cupy
from bigmodels.functions.gemm import sgemmBatched

def test():
    batch_k = cupy.random.randn(64, 64, 512).astype(cupy.float32)
    batch_q = cupy.random.randn(64, 64, 512).astype(cupy.float32)
    out = cupy.ndarray((64, 512, 512), dtype=cupy.float32)
    sgemmBatched(batch_k, True, batch_q, False, out)

    ans = cupy.ndarray((64, 512, 512), dtype=cupy.float32)
    for i in range(batch_k.shape[0]):
        ans[i] = cupy.matmul(batch_k[i].T, batch_q[i]).T
    
    diff = cupy.abs(out - ans).sum()
    print(diff)

test()