from typing import Optional
import torch
from ..arch.gpt import TorchGPT2, GPTConfiguration

class CPM1Configuration(GPTConfiguration):
    ## Structure
    DIM_MODEL = 2560
    DIM_FF = 10240
    DIM_HEAD = 80
    NUM_HEADS = 32
    NUM_LAYERS = 32
    VOCAB_SIZE = 30000
    MAX_LENGTH = 1024
    EPS = 1e-5

    
    ## runtime
    DEVICE = None
    MEMORY_LIMIT = None
    MODEL_NAME = "file:///root/toolkit/cpm1-tmp"

SUPPORTED_VERSION = ["1.1"]

class CPM1(TorchGPT2):
    def __init__(self,
            memory_limit : Optional[int] = None,
            version : str = "1.1"
        ) -> None:
        if version not in SUPPORTED_VERSION:
            raise RuntimeError("CPM1 version %s is not supported (requires %s)" % (version, SUPPORTED_VERSION))
        # TODO: set model name here
        config = CPM1Configuration()

        device_idx = torch.cuda.current_device()

        config.DEVICE = device_idx
        config.MEMORY_LIMIT = memory_limit

        super().__init__(config)
