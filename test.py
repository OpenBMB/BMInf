import numpy as np
from bigmodels.models.cpm2 import CPM2, CPM2Configuration
import logging, time
logging.basicConfig(level=logging.INFO)

def main():
    config = CPM2Configuration()
    cpm2 = CPM2(config)

    st = time.perf_counter()
    out = cpm2.forward(np.arange(512)[np.newaxis, :], [512])
    print(out.value.transpose((0, 2, 1)), out.scale)
    print(time.perf_counter() - st)
    

if __name__ == "__main__":
    main()