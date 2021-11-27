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

class EVAConfiguration(T5Configuration):
    ## structure
    DIM_MODEL = 2048
    DIM_FF = 5120
    DIM_HEAD = 64

    NUM_HEADS = 32
    NUM_ENCODER_LAYERS = 24
    NUM_DECODER_LAYERS = 24
    NUM_POSITION_BUCKETS = 32
    VOCAB_SIZE = 30000
    MAX_DISTANCE = 256

    ## runtime
    DEVICE = None
    MEMORY_LIMIT = None
    MODEL_NAME = None

SUPPORTED_VERSION = ["eva-int8-new"]
class EVA:
    def __init__(self,
            device_idx : Optional[int] = None,
            dynamic_memory : int = 512 * 1024 * 1024,   # 512MB
            memory_limit : Optional[int] = None,
            version : str = "eva-int8-new"
        ) -> None:
        if version not in SUPPORTED_VERSION:
            raise RuntimeError("EVA version %s is not supported (requires %s)" % (version, SUPPORTED_VERSION))
        config = EVAConfiguration()
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
            context : List[str],
            truncation_length : Optional[int] = 256
        ):
        idx = []
        sep_idx = self._model.tokenizer.convert_tokens_to_ids(["<sep>"])[0]
        for sentence in context:
            idx.extend(self._model.tokenizer.encode(sentence) + [sep_idx])
        idx.append( self._model.tokenizer.get_span(0) )

        if truncation_length is not None and len(idx) > truncation_length:
            idx = idx[-truncation_length:]

        input_length = len(idx)

        while len(idx) % 4 != 0:
            idx.append(0)

        return idx, input_length

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

    def dialogue(self,
            context : List[str],
            max_tokens : int = 128,
            top_n : Optional[int] = 10,
            top_p : Optional[float] = None,
            temperature : float = 0.85,
            frequency_penalty : float = 0,
            presence_penalty : float = 0,
            truncation_length : Optional[int] = 256
        ) -> Tuple[str, bool]:
        """Generate dialogue based on context.
        Args:
            context: Context of the dialogue.
            max_tokens: Maximum tokens to generate.
            top_n: Only sampling from top n tokens in the result.
            top_p: Only sampling from tokens that comprising the top p probability in the result.
            temperature: Temperature for sampling. Higher values mean more diverse results. 
            frequency_penalty: A penalty used to avoid models generating the same content.
            presence_penalty: A penalty used to avoid models generating the same topic.
        
        Returns:
            A response generated by the model.
        """
        idx, input_length = self._pre_processing(context, truncation_length)
        res = self._gen_iter(
            idx,
            input_length,
            max_tokens,
            self._model.tokenizer.get_span(0),
            top_n,
            top_p,
            temperature,
            frequency_penalty,
            presence_penalty,
            filter_tokens=[self._model.tokenizer.unk_id]
        )
        
        blanks = []
        stoped = False

        sep_idx = self._model.tokenizer.convert_tokens_to_ids(["<sep>"])[0]

        for token in res:
            if token == sep_idx:
                stoped = True
                break
            blanks.append(token)
            if len(blanks) >= max_tokens:
                break
        self.free()
        return self._model.tokenizer.decode(blanks), stoped

    def free(self):
        self._ctx.free_all()

        

