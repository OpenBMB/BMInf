from typing import Optional, Union, List
from ..arch.t5 import T5Configuration, T5
import cupy
import numpy as np
from ..utils.sampler import GenerateSampler

import logging
logger = logging.getLogger(__name__)

SPAN_TOKEN = "<span>"

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

    def generate(self, 
            input_sentence : str,
            spans_position : Optional[List[int]] = None,
            max_tokens : int = 128,
            top_n : Optional[int] = None,
            top_p : Optional[float] = None,
            temperature : float = 0.9,
            frequency_penalty : float = 0,
            presence_penalty : float = 0,
        ):
        if spans_position is None:
            spans_position = []
            st = 0
            while True:
                nw_pos = input_sentence.find(SPAN_TOKEN, st)
                if nw_pos == -1:
                    break
                spans_position.append(nw_pos)
                st = nw_pos + len(SPAN_TOKEN)
        if len(spans_position) == 0:
            raise ValueError("No spans")
        if len(spans_position) > 16:
            raise ValueError("Too many spans")
        for pos in spans_position:
            if not input_sentence[pos:].startswith(SPAN_TOKEN):
                raise ValueError("Wrong span token at position %d" % pos)
        
        idx = []
        span_idx = 0
        last_pos = 0
        for pos in spans_position:
            idx += self.text_to_id(input_sentence[last_pos: pos])
            idx += [ self.tokenizer.get_span(span_idx) ]
            span_idx += 1
            last_pos = pos + len(SPAN_TOKEN)

        idx += self.text_to_id(input_sentence[last_pos:])
        input_length = len(idx)

        ctx = self.encode(np.array([idx], dtype=np.int64), [input_length])
        self.init_decoder_context(ctx)
        
        decoder_ipts = self.tokenizer.sod_id
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

        blanks = []
        next_span = 0

        for _ in range(max_tokens):
            logits = self.decode_step(ctx, [decoder_ipts])[0]
            decoder_ipts = sampler.sample(logits)
            if decoder_ipts == self.tokenizer.get_span(next_span):
                next_span += 1
                if next_span > len(spans_position):
                    break
                blanks.append([])
            else:
                blanks[-1].append(decoder_ipts)
        
        return [
            {
                "position": blank_pos,
                "text": self.id_to_text(blank_tokens)
            } 
            for blank_pos, blank_tokens in zip( spans_position, blanks )
        ]
