from ..base import Configuration

class T5Configuration(Configuration):
    ## structure
    DIM_MODEL = 4096
    DIM_FF = 10240
    DIM_KV = 64

    NUM_HEADS = 64
    NUM_ENCODER_LAYERS = 24
    NUM_DECODER_LAYERS = 24
    NUM_POSITION_BUCKETS = 32
    VOCAB_SIZE = 26240
    MAX_DECODER_LENGTH = 256

    ENCODER_ONLY = False
    
    ## runtime
    MEMORY_OVERLAP = True
    DEVICE = None
    MEMORY_LIMIT = None
    OVERLAP_LAYERS = None
    DYNAMIC_MEMORY = 1024 * 1024 * 900
    MODEL_NAME = None
