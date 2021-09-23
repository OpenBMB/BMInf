from typing import Optional, Union, List
from ..arch.gpt import GPTConfiguration, GPT
import cupy
import numpy as np
from ..utils.sampler import GenerateSampler

import logging
logger = logging.getLogger(__name__)


class CPM1Configuration(GPTConfiguration):
    MODEL_NAME = "cpm1"

class CPM1(GPT):
    def __init__(self, device : Union[None, int, cupy.cuda.Device] = None, memory_limit : Optional[int] = None, config : Optional[CPM1Configuration] = None):
        """Model CPM-1: A Large-scale Generative Chinese Pre-trained Language Model
        
        `[Repo] <https://github.com/TsinghuaAI/CPM-1-Generate>`__
        `[PDF] <https://arxiv.org/abs/2012.00413>`__

        Args:
            device: Index of CUDA device or ``None``.
            memory_limit: Total memory limit for this model in bytes.
            config: A CPM1 configuration object.

        """
        if config is None:
            config = CPM1Configuration()

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
        
        super().__init__(config)

    def generate(self, 
            input_sentence : str,
            max_tokens : int = 32,
            top_n : Optional[int] = None,
            top_p : Optional[float] = None,
            temperature : float = 0.9,
            frequency_penalty : float = 0,
            presence_penalty : float = 0,
        ):
        """Generate some words from the model.
        
        Args:
            input_sentence: Your input.
            max_tokens: Max number of tokens to generate.
            top_n: Only sampling from top n tokens in the result.
            top_p: Only sampling from tokens that comprising the top p probability in the result.
            temperature: Temperature for sampling. Higher values mean more diverse results. 
            frequency_penalty: A penalty used to avoid models generating the same content.
            presence_penalty: A penalty used to avoid models generating the same topic.

        Returns:
            The result sentence.
        """
        idx = self.text_to_id(input_sentence)
        input_length = len(idx)
        if input_length + max_tokens > self.max_length:
            idx = idx[ input_length + max_tokens - self.max_length:]
            input_length = self.max_length - max_tokens
            

        x, ctx = self.encode(np.array([idx], dtype=np.int64), [input_length])
        sampler = GenerateSampler(
            idx, 
            self.tokenizer.vocab_size,
            self.device,
            max_tokens,
            top_n,
            top_p,
            temperature,
            frequency_penalty,
            presence_penalty,
            no_penalty_tokens=[8],
            filter_tokens=[self.tokenizer.unk_id]
        )

        ret = []

        for _ in range(max_tokens):
            dec_inputs = sampler.sample(x[0])
            if dec_inputs == self.tokenizer.eod_id:
                break
            ret.append(dec_inputs)
            x = self.decode_step(ctx, [dec_inputs])
        
        return self.id_to_text(ret)