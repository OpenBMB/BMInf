from typing import List, Optional, Tuple
from ..arch.gpt import GPTConfiguration, GPT2Model
from ..core.allocators.cuda import CUDAAllocator
from ..core.allocators.sizelimited import SizeLimitedAllocator
from ..core import Context, Device
from ..utils.sampler import GenerateSampler
from cpm_kernels.library import cudart
import cpm_kernels.kernels as ck
import numpy as np

class CPM1Configuration(GPTConfiguration):
    ## Structure
    DIM_MODEL = 2560
    DIM_FF = 10240
    DIM_HEAD = 80
    NUM_HEADS = 32
    NUM_LAYERS = 32
    VOCAB_SIZE = 30000
    MAX_LENGTH = 1024
    EPS = 1e-5


SUPPORTED_VERSION = ["cpm1-new"]
LATEST_VERSION = SUPPORTED_VERSION[-1]

class CPM1:
    def __init__(self,
            device_idx : Optional[int] = None,
            dynamic_memory : int = 512 * 1024 * 1024,
            memory_limit : Optional[int] = None,
            version : Optional[str] = None
        ) -> None:
        if version is None:
            version = LATEST_VERSION
        if version not in SUPPORTED_VERSION and not version.startswith("file://"):
            raise RuntimeError("CPM1 version %s is not supported (requires %s)" % (version, SUPPORTED_VERSION))
        config = CPM1Configuration()
        config.MODEL_NAME = version

        if device_idx is None:
            device_idx = cudart.cudaGetDevice()
        
        config.DEVICE = device_idx
        config.MEMORY_LIMIT = memory_limit

        self.device = Device(config.DEVICE)
        
        self._cudaAlloc = CUDAAllocator(config.DEVICE)
        self._ctx = Context([config.DEVICE], [
            SizeLimitedAllocator(self._cudaAlloc.allocate(dynamic_memory))
        ])
        self._model = GPT2Model(config)
        self._config = config
        self._chunk_size = 64
    
    def _pre_processing(self,
            input_sentence : str,
        ):
        
        idx = self._model.tokenizer.encode(input_sentence)
        input_length = len(idx)
        
        while len(idx) % self._chunk_size != 0:
            idx.append(0)

        return idx, input_length
    
    def _gen_iter(self,
            idx : List[int],
            input_length : int,
            max_length : int,
            top_n : Optional[int] = None,
            top_p : Optional[float] = None,
            temperature : float = 0.9,
            frequency_penalty : float = 0,
            presence_penalty : float = 0,
            no_penalty_tokens : List[int] = [8],
            filter_tokens : List[int] = []
        ):
        self.free()

        buffer_len = len(idx)
        with self.device:

            buffer_k_self = self._model.allocate_decode_buffer(
                self._ctx,
                1,
                buffer_len
            )
            buffer_v_self = self._model.allocate_decode_buffer(
                self._ctx,
                1,
                buffer_len
            )

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
            hidden_enc = self._ctx.allocate((1, self._config.DIM_MODEL, len(idx)), dtype=np.float16)
            self._model.embedding(
                self._ctx,
                np.array([idx], dtype=np.int32),
                np.arange(len(idx), dtype=np.int32)[np.newaxis, :],
                hidden_enc
            )
            mask_enc = (np.arange(len(idx)) < input_length)[np.newaxis, :]
            
            logits = self._ctx.allocate((1, self._config.VOCAB_SIZE), dtype=np.float16)
            self._model.encode(
                self._ctx,
                hidden_enc,
                mask_enc,
                buffer_k_self,
                buffer_v_self
            )
            self._model.projection(
                self._ctx,
                hidden_enc,
                logits,
                output_one=input_length - 1
            )
            self._ctx.free(hidden_enc)
            logits.reshape((self._config.VOCAB_SIZE,))
            last_ipt = sampler.sample(logits)
            self._ctx.free(logits)
        yield last_ipt

        dec_pos = input_length
        if max_length is None:
            max_length = self._config.MAX_LENGTH
        else:
            max_length = min(max_length + input_length, self._config.MAX_LENGTH)

        while dec_pos < max_length:
            with self.device:
                if dec_pos >= buffer_len:
                    nw_buffer_len = buffer_len + self._chunk_size

                    nw_buffer_k_self = self._model.allocate_decode_buffer(self._ctx, 1, nw_buffer_len)
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
                self._model.embedding_step(self._ctx, 
                    np.array([last_ipt], dtype=np.int32),
                    np.array([dec_pos], dtype=np.int32),
                    hidden_dec
                )

                logits = self._ctx.allocate((1, self._config.VOCAB_SIZE), np.float16)
                self._model.step(
                    self._ctx,
                    hidden_dec,
                    buffer_k_self,
                    buffer_v_self,
                    dec_pos
                )
                self._model.projection_step(self._ctx, hidden_dec, logits)
                self._ctx.free(hidden_dec)
                dec_pos += 1

                logits.reshape((self._config.VOCAB_SIZE,))
                last_ipt = sampler.sample(logits)
                self._ctx.free(logits)
            yield last_ipt

    def generate(self,
            input_sentence : str,
            max_tokens : int = 128,
            top_n : Optional[int] = None,
            top_p : Optional[float] = None,
            temperature : float = 0.9,
            frequency_penalty : float = 0,
            presence_penalty : float = 0,
            stop_tokens : Optional[List[str]] = None,
        ):
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
        
        idx, input_length = self._pre_processing(input_sentence)

        res = self._gen_iter(
            idx,
            input_length,
            max_tokens,
            top_n,
            top_p,
            temperature,
            frequency_penalty,
            presence_penalty,
        )

        blanks = []
        stoped = False

        for token in res:
            if token in stop_tokens:
                stoped = True
                break
            blanks.append(token)
            if len(blanks) >= max_tokens:
                break
        self.free()
        return self._model.tokenizer.decode(blanks), stoped

    def free(self):
        self._ctx.free_all()

