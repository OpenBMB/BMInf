class T5Configuration:
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

    ENCODER_ONLY = False
    
    ## runtime
    DEVICE = None
    MEMORY_LIMIT = None
    DYNAMIC_MEMORY = 1024 * 1024 * 512
    MODEL_NAME = None