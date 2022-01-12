from typing import List, Optional, Tuple
from ..arch.t5 import T5Configuration, T5Model
from ..core.allocators.cuda import CUDAAllocator
from ..core.allocators.sizelimited import SizeLimitedAllocator
from ..core import Context, Device
from ..utils.sampler import GenerateSampler
from cpm_kernels.library import cudart
import cpm_kernels.kernels as ck
import numpy as np

SPAN_TOKEN = "<span>"

class CPM2Configuration(T5Configuration):
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
    EPS = 1e-6


SUPPORTED_VERSION = ["cpm2.1-new"]
LATEST_VERSION =  SUPPORTED_VERSION[-1]

class CPM2:
    def __init__(self,
            device_idx : Optional[int] = None,
            dynamic_memory : int = 512 * 1024 * 1024,   # 512MB
            memory_limit : Optional[int] = None,
            version : Optional[str] = None
        ) -> None:
        if version is None:
            version = LATEST_VERSION
        if version not in SUPPORTED_VERSION and not version.startswith("file://"):
            raise RuntimeError("CPM2 version %s is not supported (requires %s)" % (version, SUPPORTED_VERSION))
        config = CPM2Configuration()
        config.MODEL_NAME = version

        if device_idx is None:
            device_idx = cudart.cudaGetDevice()
        
        config.DEVICE = device_idx
        config.MEMORY_LIMIT = memory_limit

        self.device = Device(config.DEVICE)

        self._cudaAlloc = CUDAAllocator(config.DEVICE)
        self._ctx = Context([config.DEVICE], [
            SizeLimitedAllocator( self._cudaAlloc.allocate( dynamic_memory ))
        ])
        self._model = T5Model(config)
        self._config = config

    def _pre_processing(self,
            input_sentence : str,
            spans_position : Optional[List[int]] = None,
            start_span_idx : int = 0
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
        span_idx = start_span_idx
        last_pos = 0
        for pos in spans_position:
            idx += self._model.tokenizer.encode(input_sentence[last_pos: pos])
            idx += [ self._model.tokenizer.get_span(span_idx) ]
            span_idx += 1
            last_pos = pos + len(SPAN_TOKEN)

        idx += self._model.tokenizer.encode(input_sentence[last_pos:])
        input_length = len(idx)

        while len(idx) % 4 != 0:
            idx.append(0)

        return idx, input_length, spans_position

    def _gen_iter(self, 
            idx : List[int],
            input_length : int,
            max_length : int,
            start_token : int,
            top_n : Optional[int] = None,
            top_p : Optional[float] = None,
            temperature : float = 0.9,
            frequency_penalty : float = 0,
            presence_penalty : float = 0,
            no_penalty_tokens : List[int] = [],
            filter_tokens : List[int] = []
        ):
        self.free()

        with self.device:
            sampler = GenerateSampler(
                self._ctx,
                idx,
                self._model.tokenizer.vocab_size,
                top_n,
                top_p,
                temperature,
                frequency_penalty,
                presence_penalty,
                no_penalty_tokens,
                filter_tokens
            )
            hidden_enc = self._ctx.allocate((1, self._config.DIM_MODEL, len(idx)), np.float16)
            self._model.embedding(
                self._ctx,
                np.array([idx], dtype=np.int32),
                hidden_enc
            )
            mask_enc = (np.arange(len(idx)) < input_length)[np.newaxis, :]
            self._model.encode(
                self._ctx,
                hidden_enc,
                mask_enc
            )

            buffer_len = 0
            dec_pos = 0
            buffer_k_self = None
            buffer_v_self = None
            buffer_k_cros = self._model.allocate_decode_buffer(self._ctx, 1, hidden_enc.shape[-1])
            buffer_v_cros = self._model.allocate_decode_buffer(self._ctx, 1, hidden_enc.shape[-1])

        last_ipt = start_token
        while max_length is None or dec_pos < max_length:
            with self.device:
                if dec_pos >= buffer_len:
                    # need new buffer
                    nw_buffer_len = buffer_len + 64
                    nw_buffer_k_self = self._model.allocate_decode_buffer(self._ctx, 1, nw_buffer_len)
                    if buffer_k_self is not None:
                        for old, nw in zip(buffer_k_self, nw_buffer_k_self):
                            ck.utils.copy_extend_buffer(
                                nw.shape[1], old.shape[2] * old.shape[3], nw.shape[2] * nw.shape[3],
                                old.ptr,
                                nw.ptr,
                                self._ctx.current_stream
                            )
                            self._ctx.free(old)
                    buffer_k_self = nw_buffer_k_self

                    nw_buffer_v_self = self._model.allocate_decode_buffer(self._ctx, 1, nw_buffer_len)
                    if buffer_v_self is not None:
                        for old, nw in zip(buffer_v_self, nw_buffer_v_self):
                            ck.utils.copy_extend_buffer(
                                nw.shape[1], old.shape[2] * old.shape[3], nw.shape[2] * nw.shape[3],
                                old.ptr,
                                nw.ptr,
                                self._ctx.current_stream
                            )
                            self._ctx.free(old)
                    buffer_v_self = nw_buffer_v_self
                    buffer_len = nw_buffer_len
                
                hidden_dec = self._ctx.allocate((1, self._config.DIM_MODEL), np.float16)
                self._model.embedding_step(self._ctx, np.array([last_ipt], dtype=np.int32), hidden_dec)
                
                logits = self._ctx.allocate((1, self._config.VOCAB_SIZE), np.float16)
                self._model.decode_step(
                    self._ctx,
                    hidden_dec,
                    hidden_enc if dec_pos == 0 else None, 
                    mask_enc,
                    buffer_k_self,
                    buffer_v_self,
                    buffer_k_cros,
                    buffer_v_cros,
                    dec_pos
                )
                self._model.projection_step(self._ctx, hidden_dec, logits)
                if dec_pos == 0:
                    self._ctx.free(hidden_enc)
                self._ctx.free(hidden_dec)
                dec_pos += 1

                logits.reshape((self._config.VOCAB_SIZE,))
                last_ipt = sampler.sample(logits)
                self._ctx.free(logits)
            yield last_ipt

    def fill_blank(self, 
            input_sentence : str,
            spans_position : Optional[List[int]] = None,
            max_tokens : int = 128,
            top_n : Optional[int] = None,
            top_p : Optional[float] = None,
            temperature : float = 0.9,
            frequency_penalty : float = 0,
            presence_penalty : float = 0,
        ):
        """Generate spans from input sentence.

        Args:
            input_sentence: Input sentence with "<span>" tokens.
            spans_position: List of span positions. If ``None``, the positions of span are automatically detected.
            max_tokens: Maximum number of tokens to generate.
            top_n: Only sampling from top n tokens in the result.
            top_p: Only sampling from tokens that comprising the top p probability in the result.
            temperature: Temperature for sampling. Higher values mean more diverse results. 
            frequency_penalty: A penalty used to avoid models generating the same content.
            presence_penalty: A penalty used to avoid models generating the same topic.
        
        Returns:
            A list of generated spans, including positions and contents.
        """
        idx, input_length, spans_position = self._pre_processing(input_sentence, spans_position, 0)
        res = self._gen_iter(
            idx,
            input_length,
            max_tokens,
            self._model.tokenizer.sod_id,
            top_n,
            top_p,
            temperature,
            frequency_penalty,
            presence_penalty,
            filter_tokens=[self._model.tokenizer.unk_id]
        )
        next_span = 0
        blanks = []
        for token in res:
            if token == self._model.tokenizer.get_span(next_span):
                blanks.append([])
                next_span += 1
                if next_span > len(spans_position):
                    break
            elif next_span == 0:
                raise RuntimeError("Unexpected model output: %d" % token)
            else:
                blanks[-1].append(token)
        self.free()
        return [
            {
                "position": blank_pos,
                "text": self._model.tokenizer.decode(blank_tokens)
            } 
            for blank_pos, blank_tokens in zip( spans_position, blanks )
        ]
    
    def generate(self,
            input_sentence : str,
            max_tokens : int = 128,
            top_n : Optional[int] = None,
            top_p : Optional[float] = None,
            temperature : float = 0.9,
            frequency_penalty : float = 0,
            presence_penalty : float = 0,
            stop_tokens : Optional[List[str]] = None,
        ) -> Tuple[str, bool]:
        """Generate some words from the model.
        
        Args:
            input_sentence: Your input.
            max_tokens: Maximum number of tokens to generate.
            top_n: Only sampling from top n tokens in the result.
            top_p: Only sampling from tokens that comprising the top p probability in the result.
            temperature: Temperature for sampling. Higher values mean more diverse results. 
            frequency_penalty: A penalty used to avoid models generating the same content.
            presence_penalty: A penalty used to avoid models generating the same topic.
            stop_tokens: A list of tokens that will stop the generation.
        
        Returns:
            The result sentence and a boolean indicating whether stop_tokens has been generated.
        """
        if stop_tokens is None:
            stop_tokens = []
        else:
            stop_tokens = [
                self._model.tokenizer.encoder.get(word, self._model.tokenizer.unk_token) 
                    for word in stop_tokens
            ]
        if not self._model.tokenizer.eod_id in stop_tokens:
            stop_tokens.append(self._model.tokenizer.eod_id)
        idx, input_length, _ = self._pre_processing(
            input_sentence + SPAN_TOKEN,
            [len(input_sentence)],
            189 # start from 189 span
        )
        res = self._gen_iter(
            idx,
            input_length,
            max_tokens,
            self._model.tokenizer.sod_id,
            top_n,
            top_p,
            temperature,
            frequency_penalty,
            presence_penalty,
            filter_tokens=[self._model.tokenizer.unk_id]
        )
        next(res)   # skip first span token
        blanks = []
        stoped = False
        
        sentinels = set(self._model.tokenizer.sentinel_list)
        for token in res:
            if token in stop_tokens or token in sentinels:
                stoped = True
                break
            blanks.append(token)
            if len(blanks) >= max_tokens:
                break
        self.free()
        return self._model.tokenizer.decode(blanks), stoped

    def free(self):
        self._ctx.free_all()

        

