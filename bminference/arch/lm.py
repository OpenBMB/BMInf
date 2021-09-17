from typing import List, Union
from .base import Model, InferenceContext, Tokenizer
import numpy as np
import cupy


class LMModelTokenizer(Tokenizer):
    def get_span(self, span_id) -> int:
        raise NotImplementedError()

class LMModel(Model):
    
    tokenizer : LMModelTokenizer

    def encode(self, input_idx : np.ndarray, input_length : List[int]) -> InferenceContext:
        raise NotImplementedError()

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

    