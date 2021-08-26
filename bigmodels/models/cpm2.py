from typing import Optional, Union
from ..arch.t5 import T5Configuration, T5
import cupy

class CPM2Configuration(T5Configuration):
    MODEL_NAME = "cpm2"

class CPM2(T5):
    def __init__(self, device : Union[None, int, cupy.cuda.Device] = None, memory_limit : Optional[int] = None, config : Optional[CPM2Configuration] = None):
        if config is None:
            config = CPM2Configuration()

        if config.DEVICE is None:
            if device is None:
                device = 0
            if isinstance(device, int):
                device = cupy.cuda.Device(device)
            config.DEVICE = device
        
        # generate cublas handler
        config.DEVICE.cublas_handle

        if config.MEMORY_LIMIT is None:
            if memory_limit is None:
                # free - 100MB
                memory_limit = config.DEVICE.mem_info[0] - 100 * 1024 * 1024
            config.MEMORY_LIMIT = memory_limit
        
        if config.MEMORY_OVERLAP is None:
            config.MEMORY_OVERLAP = (config.MEMORY_LIMIT < 12 * 1024 * 1024)    # < 12GB

        if config.MEMORY_OVERLAP:
            if config.OVERLAP_LAYERS is None:
                max_layers = (config.MEMORY_LIMIT - config.DYNAMIC_MEMORY - 1235640320) // (226615296 * 4)
                if max_layers < 1:
                    raise RuntimeError("GPU memory is not enough")
                config.OVERLAP_LAYERS = max_layers
        
        super().__init__(config)
