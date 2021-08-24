import numpy as np
from bigmodels.models.cpm2 import CPM2, CPM2Configuration
import logging, time
import cupy

def main():
    config = CPM2Configuration()
    cpm2 = CPM2(config)

    st = time.perf_counter()
    out = cpm2.encode(np.array([
        [1, 5971, 1215, 760, 762, 765, 10617, 1971, 1215, 26050]
    ]), [10])

    print(time.perf_counter() - st)
    for it in cpm2.decode(out, [512]):
        print(it)

if __name__ == "__main__":
    main()