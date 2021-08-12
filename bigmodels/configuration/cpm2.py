from .base import Configuration

class CPM2Configuration(Configuration):
    DIM_MODEL = 4096
    DIM_FF = 10240
    DIM_KV = 64

    NUM_HEADS = 64
    NUM_ENCODER_LAYERS = 24
    NUM_DECODER_LAYERS = 24
    NUM_POSITION_BUCKETS = 32
    VOCAB_SIZE = 26240

    MEMORY_OVERLAP = True
    OVERLAP_LAYERS = 2
    ENCODER_ONLY = False
    DEVICE = 0
    
    MEMORY_LIMIT = 1024 * 1024 * 1024 * 4 # total memory
    
    DYNAMIC_MEMORY = 1024 * 1024 * 512 # memory size for non-parameter variables

    MODEL_PATH = "./checkpoint.pt"