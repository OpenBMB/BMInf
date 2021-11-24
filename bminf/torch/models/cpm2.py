import torch
from typing import List, Optional, Tuple
from ..arch.t5 import T5Configuration
from ..arch.t5 import TorchT5

class CPM2Configuration(T5Configuration):
    ## structure
    DIM_MODEL = 4096
    DIM_FF = 10240
    DIM_HEAD = 64

    NUM_HEADS = 64
    NUM_ENCODER_LAYERS = 24
    NUM_DECODER_LAYERS = 24
    NUM_POSITION_BUCKETS = 32
    VOCAB_SIZE = 26240
    MAX_DISTANCE = 128

    ## runtime
    DEVICE = None
    MEMORY_LIMIT = None
    MODEL_NAME = "file:///root/toolkit/cpm-tmp"

SUPPORTED_VERSION = ["2.2"]


class CPM2(TorchT5):
    def __init__(self,
            memory_limit : Optional[int] = None,
            version : str = "2.2"
        ) -> None:
        if version not in SUPPORTED_VERSION:
            raise RuntimeError("CPM2 version %s is not supported (requires %s)" % (version, SUPPORTED_VERSION))
        # TODO: set model name here
        config = CPM2Configuration()

        device_idx = torch.cuda.current_device()

        config.DEVICE = device_idx
        config.MEMORY_LIMIT = memory_limit

        super().__init__(config)

        