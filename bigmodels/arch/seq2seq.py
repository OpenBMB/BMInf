from typing import List, Optional, Union
from .base import Model, InferenceContext, Tokenizer
import numpy as np
import cupy
from ..utils.sampler import GenerateSampler

SPAN_TOKEN = "<span>"

class Seq2SeqTokenizer(Tokenizer):
    def get_span(self, span_id) -> int:
        raise NotImplementedError()

class Seq2SeqModel(Model):
    
    tokenizer : Seq2SeqTokenizer

    def encode(self, input_idx : np.ndarray, input_length : List[int]) -> InferenceContext:
        raise NotImplementedError()
    
    def init_decoder_context(self, ctx : InferenceContext):
        return self._init_decoder_context(ctx)

    def decode_step(self, ctx : InferenceContext, inputs : Union[List[int], np.ndarray]) -> cupy.ndarray:
        raise NotImplementedError()

    def text_to_id(self, sentence : str) -> List[int]:
        return self._text_to_id(sentence)

    def id_to_text(self, idx : List[int]) -> str:
        return self._id_to_text(idx)
    
    def get_token_id(self, token : str, use_unk : bool = True) -> Union[int, None]:
        return self._get_token_id(token, use_unk)
        
    def get_id_token(self, idx : int) -> str:
        return self.get_id_token(idx)

    def _text_to_id(self, sentence : str) -> List[int]:
        raise NotImplementedError()

    def _id_to_text(self, idx : List[int]) -> str:
        raise NotImplementedError()
    
    def _get_token_id(self, token : str, use_unk : bool) -> Union[int, None]:
        raise NotImplementedError()
        
    def _get_id_token(self, idx : int) -> str:
        raise NotImplementedError()
    
    def _init_decoder_context(self, ctx : InferenceContext):
        raise NotImplementedError()

    def generate(self, 
            input_sentence : str,
            spans_position : Optional[List[int]] = None,
            max_tokens : int = 128,
            top_n : Optional[int] = None,
            top_p : Optional[float] = None,
            temperature : float = 0.9,
            frequency_penalty : List[float] = 0,
            presence_penalty : List[float] = 0,
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
