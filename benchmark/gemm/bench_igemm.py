from tqdm import tqdm
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
    for _ in tqdm(range(3200)):
        igemm(a, ma, aT, mb, bT, mc)
        mc.device.synchronize()
    
def main():
    allocator = SizeLimitedAllocator(1024 * 1024 * 1024 * 4)
    test(allocator, 128, 1023, 1023, 1023)

if __name__ == "__main__":
    main()