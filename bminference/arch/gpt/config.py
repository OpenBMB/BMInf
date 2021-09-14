from ..base import Configuration

class GPTConfiguration(Configuration):
    ## structure
    DIM_MODEL = 2560
    DIM_FF = 10240
    DIM_KV = 64

    NUM_HEADS = 32
    NUM_LAYERS = 32
    
    VOCAB_SIZE = 26240
    
    MAX_LENGTH = 1024

    ENCODER_ONLY = False
    
    ## runtime
    MEMORY_OVERLAP = True
    DEVICE = None
    MEMORY_LIMIT = None
    OVERLAP_LAYERS = None
    DYNAMIC_MEMORY = 1024 * 1024 * 512
    MODEL_NAME = None
