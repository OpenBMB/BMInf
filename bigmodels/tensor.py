import cupy

class Tensor:
    def __init__(self, value : cupy.ndarray, scale : float = 1.0):
        self.value = value
        self.scale = scale