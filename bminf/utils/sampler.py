from ..core import Context, Tensor
from typing import List, Optional
import numpy as np
import cpm_kernels.kernels as ck

class GenerateSampler:
    def __init__(self,
            ctx : Context,
            prompt_text : List[int],
            vocab_size : int,
            top_n : Optional[int] = None,
            top_p : Optional[float] = None,
            temperature : float = 1,
            frequency_penalty : List[float] = 0.1,
            presence_penalty : List[float] = 0.1,
            no_penalty_tokens : List[int] = [],
            filter_tokens : List[int] = [],
        ):
        self.temperature = temperature
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.vocab_size = vocab_size
        self.top_n = top_n
        self.top_p = top_p
        self.no_penalty_tokens = set(no_penalty_tokens)


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

        frequency_cpu = np.zeros((vocab_size,), dtype=np.int32)
        for token in prompt_text:
            frequency_cpu[token] += 1
        self.frequency_count = Tensor.from_numpy(ctx, frequency_cpu)

        filter_mask = np.ones((vocab_size,), dtype=np.int8)
        for token in filter_tokens:
            filter_mask[token] = 0
        self.filter_mask = Tensor.from_numpy(ctx, filter_mask)

        self._ctx = ctx

    def sample(self, logits : Tensor) -> int:
        assert logits.shape == (self.vocab_size,)
        
        ck.utils.adjustify_logits(
            1, self.vocab_size,
            logits.ptr,
            self.temperature,
            self.frequency_penalty,
            self.presence_penalty,
            self.frequency_count.ptr,
            self._ctx.current_stream
        )
        ck.mask(
            1, 1, self.vocab_size,
            logits.ptr,
            self.filter_mask.ptr,
            float("-inf"),
            logits.ptr,
            self._ctx.current_stream
        )
        ck.softmax_step_inplace(
            1, self.vocab_size,
            logits.ptr,
            self._ctx.current_stream
        )
        logits_cpu = logits.numpy().astype(np.float32)
        idx = logits_cpu.argsort()
        logits_cpu.sort()
        cut_off = 0
        if self.top_n is not None:
            cut_off = max(cut_off, self.vocab_size - self.top_n)

        if self.top_p is not None:
            suffix_sum = 0
            suffix_pos = logits_cpu.shape[0]
            while suffix_pos > 0:
                suffix_pos -= 1
                suffix_sum += logits_cpu[suffix_pos]
                if suffix_sum > self.top_p:
                    break
            cut_off = max(cut_off, suffix_pos)
        logits_cpu[:cut_off] = 0
        logits_cpu /= logits_cpu.sum()
        ret = idx[np.random.choice(logits_cpu.shape[0], p=logits_cpu)].item()
        if ret not in self.no_penalty_tokens:
            ck.utils.array_add(
                self.frequency_count.ptr,
                ret,
                1,
                self._ctx.current_stream
            )
        return ret

    def free(self):
        self._ctx.free(self.frequency_count)
        self._ctx.free(self.filter_mask)