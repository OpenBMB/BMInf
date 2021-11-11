from cpm_kernels.device import Device as CPMDevice
from cpm_kernels.device import num_devices
from cpm_kernels.library import cudart

class _Device:
    def __init__(self, idx : int) -> None:
        self.device = CPMDevice(idx)
        self.__idx = idx
        
        self.__device_stack = []
        self.__device_stack_skip = []
    
    def __enter__(self) -> None:
        curr_device = cudart.cudaGetDevice()
        if curr_device == self.__idx:
            self.__device_stack_skip.append(True)
            return self
        else:
            self.__device_stack_skip.append(False)
            self.__device_stack.append( curr_device )
            cudart.cudaSetDevice(self.__idx)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.__device_stack_skip.pop():
            return
        cudart.cudaSetDevice(self.__device_stack.pop())

    @property
    def idx(self):
        return self.__idx

DEVICE_LIST = [
    _Device(i) for i in range(num_devices())
]

class Device:
    def __init__(self, idx) -> None:
        self.__device = DEVICE_LIST[idx]
    
    def __enter__(self) -> None:
        self.__device.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.__device.__exit__(exc_type, exc_val, exc_tb)
        return 
    
    def __str__(self):
        return "<Device: %d>" % self.idx
    
    @property
    def idx(self):
        return self.__device.idx