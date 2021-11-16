from .device import Device
from .utils import get_traceback
from . import config

class Memory:
    def __init__(self, ptr : int, nbytes : int, device : Device) -> None:
        self.ptr = ptr
        self.nbytes = nbytes
        self.device = device

        if config.DEBUG:
            self.trace = get_traceback()
    
    def __str__(self):
        if hasattr(self, "trace"):
            return f"<Memory: {self.ptr} {self.nbytes}, at {self.trace}>"
        else:
            return f"<Memory: {self.ptr} {self.nbytes}>"
    
    def __repr__(self):
        return self.__str__()
    
    def __hash__(self) -> int:
        return self.ptr