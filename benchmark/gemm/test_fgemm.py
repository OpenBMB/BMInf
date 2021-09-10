from bminference.allocator import SizeLimitedAllocator
from bminference.functions.gemm import fgemm
import cupy
import math
import random

def test(a, b, m, k, n):
    aT = random.random() < 0.5
    bT = random.random() < 0.5
    
    broad_b = random.random()

    ba = b
    bb = b
    if broad_b < 0.3:
        ba = 1
    elif broad_b < 0.6:
        bb = 1
    
    if aT:
        ma = cupy.random.randn(ba, m, k)
    else:
        ma = cupy.random.randn(ba, k, m)
    
    if bT:
        mb = cupy.random.randn(bb, k, n)
    else:
        mb = cupy.random.randn(bb, n, k)

    use_fp16 = random.random() < 0.5
    if use_fp16:
        dtype = cupy.float16
    else:
        dtype = cupy.float32
    ma = ma.astype(dtype) / math.sqrt(k)
    mb = mb.astype(dtype) / math.sqrt(k)

    mc = cupy.ndarray((b, n, m), dtype=dtype)
    fgemm(a, ma, aT, mb, bT, mc)
    ac = cupy.ndarray((b, n, m), dtype=dtype)
    for i in range(b):
        if ba == 1:
            m1 = ma[0]
        else:
            m1 = ma[i]
        
        if bb == 1:
            m2 = mb[0]
        else:
            m2 = mb[i]
        if not aT:
            m1 = m1.T
        if not bT:
            m2 = m2.T
        ac[i, :] = cupy.matmul(m1, m2).T
    diff = cupy.abs(ac - mc)
    amx = diff.argmax()
    mx = diff.max()
    print(mx)
    if mx > 5e-3:
        print("Test (%d, %d, %d, %d)" % (b, m, k, n))
        print("aT: %s, bT: %s, batch_a: %d, batch_b: %d" % (aT, bT, ba, bb))
        print (ac.reshape(-1)[amx], mc.reshape(-1)[amx])
        print(mx)
        print(diff)
        from IPython import embed; embed()

    
def main():
    allocator = SizeLimitedAllocator(1024 * 1024 * 1024 * 4)
    for i in range(3200):
        b = random.randint(1, 128)
        m = random.randint(1, 1024)
        k = random.randint(1, 1024)
        n = random.randint(1, 1024)
        test(allocator, b, m, k, n)

if __name__ == "__main__":
    main()