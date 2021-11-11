from .device import Device

class Memory:
    def __init__(self, ptr : int, nbytes : int, device : Device) -> None:
        self.ptr = ptr
        self.nbytes = nbytes
        self.device = device