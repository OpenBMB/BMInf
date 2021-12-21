class GPTConfiguration:
    ## structure
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
    MODEL_NAME = None