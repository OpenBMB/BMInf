import cupy

class Tensor:
    def __init__(self, value, scale = 1.0):
        self.value = value
        self.scale = scale