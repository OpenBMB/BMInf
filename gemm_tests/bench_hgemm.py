from tqdm import tqdm
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

    use_fp16 = True
    if use_fp16:
        dtype = cupy.float16
    else:
        dtype = cupy.float32
    ma = ma.astype(dtype) / math.sqrt(k)
    mb = mb.astype(dtype) / math.sqrt(k)

    mc = cupy.ndarray((b, n, m), dtype=dtype)
    for _ in tqdm(range(3200)):
        fgemm(a, ma, aT, mb, bT, mc)
        mc.device.synchronize()
    

    
def main():
    allocator = SizeLimitedAllocator(1024 * 1024 * 1024 * 4)
    test(allocator, 128, 1023, 1023, 1023)

if __name__ == "__main__":
    main()