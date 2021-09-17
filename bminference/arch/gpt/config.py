from ..base import Configuration

class GPTConfiguration(Configuration):
    ## structure
    DIM_MODEL = 2560
    DIM_FF = 10240
    DIM_KV = 80

    NUM_HEADS = 32
    NUM_LAYERS = 32
    
    VOCAB_SIZE = 30000
    
    MAX_LENGTH = 1024

    ENCODER_ONLY = False
    
    ## runtime
    DEVICE = None
    MEMORY_LIMIT = None
    DYNAMIC_MEMORY = 1024 * 1024 * 512
    MODEL_NAME = None
