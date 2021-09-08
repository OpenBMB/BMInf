from bminference import allocator
from bminference.allocator import SizeLimitedAllocator
from bminference.functions.gemm import igemm
import cupy
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
        ma = (cupy.random.randn(ba, m, k) * 4).astype(cupy.int8)
    else:
        ma = (cupy.random.randn(ba, k, m) * 4).astype(cupy.int8)
    
    if bT:
        mb = (cupy.random.randn(bb, k, n) * 4).astype(cupy.int8)
    else:
        mb = (cupy.random.randn(bb, n, k) * 4).astype(cupy.int8)

    mc = cupy.ndarray((b, n, m), dtype=cupy.int32)
    igemm(a, ma, aT, mb, bT, mc)
    """
    ac = cupy.ndarray((b, n, m), dtype=cupy.int32)
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
        m1 = m1.astype(cupy.int32)
        if not bT:
            m2 = m2.T
        m2 = m2.astype(cupy.int32)
        ac[i, :] = cupy.matmul(m1, m2).T
    diff = cupy.abs(ac - mc)
    amx = diff.argmax()
    if amx > 0:
        print("Test (%d, %d, %d, %d)" % (b, m, k, n))
        print("aT: %s, bT: %s, batch_a: %d, batch_b: %d" % (aT, bT, ba, bb))
        print (ac.reshape(-1)[amx], mc.reshape(-1)[amx])
        print(diff.max())
        print(diff)
        from IPython import embed; embed()
    """
    
def main():
    allocator = SizeLimitedAllocator(1024 * 1024 * 1024 * 4)
    """
    for i in range(3200):
        b = random.randint(1, 128)
        m = random.randint(1, 1024)
        k = random.randint(1, 1024)
        n = random.randint(1, 1024)
        test(allocator, b, m, k, n)
        print(allocator._pool.used_bytes(), allocator._pool.total_bytes())
    """
    from tqdm import tqdm
    for i in tqdm(range(3200)):
        test(allocator, 128, 1023, 1023, 1023)

if __name__ == "__main__":
    main()