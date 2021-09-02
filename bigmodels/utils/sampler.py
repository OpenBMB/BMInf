from typing import List, Optional
import numpy as np
import cupy

class GenerateSampler:
    def __init__(self,
            prompt_text : List[int],
            vocab_size : int,
            device : cupy.cuda.Device,
            max_length : int = 128,
            top_n : Optional[int] = None,
            top_p : Optional[float] = None,
            temperature : float = 1,
            frequency_penalty : List[float] = 0.1,
            presence_penalty : List[float] = 0.1,
        ):
        self.max_length = max_length
        self.temperature = cupy.float16(temperature)
        self.frequency_penalty = cupy.float16(frequency_penalty)
        self.presence_penalty = cupy.float16(presence_penalty)
        self.vocab_size = vocab_size
        self.device = device
        self.top_n = top_n
        self.top_p = top_p

        if self.top_n is not None:
            if self.top_n > vocab_size:
                raise ValueError("top_n is larger than dictionary size")
            if self.top_n <= 0:
                raise ValueError("top_n <= 0")
        if self.top_p is not None:
            if self.top_p > 1:
                raise ValueError("top_p > 1")
            if self.top_p <= 0:
                raise ValueError("top_p <= 0")

        with device:
            self.frequency_count = cupy.zeros((vocab_size,), dtype=cupy.int32)
            for token in prompt_text:
                self.frequency_count[token] += 1

    def sample(self, logits : cupy.ndarray) -> int:
        assert logits.shape == (self.vocab_size,)
        assert logits.device == self.device
        with self.device:
            logits /= self.temperature
            logits -= self.frequency_penalty * self.frequency_count
            logits -= self.presence_penalty * (self.frequency_count > 1)

            logits -= logits.max()
            logits = cupy.exp(logits)
            logits /= logits.sum()
            cpu_probs = cupy.asnumpy(logits).astype(np.float32)

        idx = cpu_probs.argsort()
        cpu_probs.sort()

        cut_off = 0
        if self.top_n is not None:
            cut_off = max(cut_off, self.vocab_size - self.top_n)

        if self.top_p is not None:
            suffix_sum = 0
            suffix_pos = cpu_probs.shape[0]
            while suffix_pos > 0:
                suffix_pos -= 1
                suffix_sum += cpu_probs[suffix_pos]
                if suffix_sum > self.top_p:
                    break
            cut_off = max(cut_off, suffix_pos)
        
        cpu_probs[:cut_off] = 0
        
        cpu_probs /= cpu_probs.sum()
        ret = idx[np.random.choice(cpu_probs.shape[0], p=cpu_probs)].item()
        with self.device:
            self.frequency_count[ret] += 1
        return ret

