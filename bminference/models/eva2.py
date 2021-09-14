from typing import Optional, Union, List
import numpy as np
from ..arch.t5 import T5Configuration, T5
import cupy
from ..utils.sampler import GenerateSampler

import logging
logger = logging.getLogger(__name__)


class EVA2Configuration(T5Configuration):
    MODEL_NAME = "eva2"
    DIM_MODEL = 2048
    DIM_FF = 5120
    DIM_KV = 64

    NUM_HEADS = 32
    NUM_ENCODER_LAYERS = 24
    NUM_DECODER_LAYERS = 24
    NUM_POSITION_BUCKETS = 32
    VOCAB_SIZE = 30000
    MAX_DECODER_LENGTH = 256

class EVA2(T5):
    def __init__(self, device : Union[None, int, cupy.cuda.Device] = None, memory_limit : Optional[int] = None, config : Optional[EVA2Configuration] = None):
        if config is None:
            config = EVA2Configuration()

        if config.DEVICE is None:
            if device is None:
                device = 0
            if isinstance(device, int):
                device = cupy.cuda.Device(device)
            config.DEVICE = device

        if config.MEMORY_LIMIT is None:
            if memory_limit is None:
                # free - 100MB
                memory_limit = config.DEVICE.mem_info[0] - 100 * 1024 * 1024
            config.MEMORY_LIMIT = memory_limit
        
        if config.MEMORY_OVERLAP:
            if config.OVERLAP_LAYERS is None:
                max_overlap = max(config.NUM_ENCODER_LAYERS, config.NUM_DECODER_LAYERS)
                max_layers = (config.MEMORY_LIMIT - config.DYNAMIC_MEMORY - 1235640320) // 226615296

                logger.info("Auto overlap layers: (max_layers: %d, max_overlap: %d)", max_layers, max_overlap)
                if max_layers * 3 < max_overlap * 4:
                    config.OVERLAP_LAYERS = max_layers // 4
                elif max_layers < max_overlap * 2:
                    config.OVERLAP_LAYERS = max_layers - max_overlap
                else:
                    config.OVERLAP_LAYERS = max_overlap
                logger.info("Auto overlap layers: result %d", config.OVERLAP_LAYERS)
                if config.OVERLAP_LAYERS < 1:
                    raise ValueError("Memory is not enough")

        super().__init__(config)

    def dialogue(self, 
            context : List[str],
            max_tokens : int = 128,
            top_n : Optional[int] = None,
            top_p : Optional[float] = None,
            temperature : float = 0.9,
            frequency_penalty : float = 0,
            presence_penalty : float = 0,
            truncation_length : Optional[int] = 256
        ):
        
        idx = []
        for sentence in context:
            idx.extend( self.text_to_id(sentence) + [self.get_token_id("<sep>")] )
        idx += [ self.get_token_id("<s_0>") ]
        if truncation_length is not None and len(idx) > truncation_length:
            idx = idx[-truncation_length:]
        input_length = len(idx)
        ctx = self.encode(np.array([idx], dtype=np.int64), [input_length])
        self.init_decoder_context(ctx)

        decoder_ipts = self.get_token_id("<s_0>")
        sampler = GenerateSampler(
            idx, 
            self.tokenizer.vocab_size,
            self.device,
            max_tokens,
            top_n,
            top_p,
            temperature,
            frequency_penalty,
            presence_penalty
        )

        ret = []
        sep_id = self.get_token_id("<sep>")
        for _ in range(max_tokens):
            logits = self.decode_step(ctx, [decoder_ipts])[0]
            decoder_ipts = sampler.sample(logits)
            if decoder_ipts == sep_id:
                break
            ret.append(decoder_ipts)
            
        return self.id_to_text(ret)