import numpy as np
import cupy
def greedy_sampler(x : cupy.ndarray) -> int:
    assert len(x.shape) == 1

    return cupy.asnumpy(x.argmax()).item()

def random_sampler(x : cupy.ndarray) -> int:
    assert len(x.shape) == 1

    x -= x.max()
    x = cupy.exp(x)
    sum_x = x.sum()
    x /= sum_x
    x = cupy.asnumpy(x)

    x[x.argsort()[:-20]] = 0
    x /= x.sum()
    
    return np.random.choice(x.shape[0], p=x)