from bigmodels.allocator.reused import ReusedAllocator
from bigmodels.models.cpm2 import CPM2, CPM2Configuration
from bigmodels.allocator.base import AllocatorConfig
from bigmodels.context import Context

import cupy
import time

def main():
    config = CPM2Configuration()
    st = time.perf_counter()
    cpm2 = CPM2(config)
    print( time.perf_counter() - st )

if __name__ == "__main__":
    main()