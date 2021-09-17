from typing import List, Tuple, Union
import cupy
from ..lm import LMModel
from ...layers.lm_head import LMHead
from ...layers.transformer_block import TransformerBlockGPT
from ...layers.embedding import Embedding
from ...layers.layer_norm import GPTLayerNorm
from ...layers.mask import InputMask
from ...layers.layer_list import LayerList
from .config import GPTConfiguration
from .tokenizer import GPT2Tokenizer
from .context import GPTInferenceContext
from ...allocator import ReusedAllocator, SizeLimitedAllocator
import numpy as np
import logging
from ... import data

logger = logging.getLogger(__name__)

class GPT(LMModel):
    def __init__(self, config : GPTConfiguration):
        # Build Model
        logger.info("Building model")
        
        self.max_overlap_layers = config.NUM_LAYERS
        self.max_length = config.MAX_LENGTH
        self.dim_model = config.DIM_MODEL

        logger.info("============ GPT ==============")
        logger.info("MAX_LENGTH: %s", self.max_length)

        self.input_embedding = Embedding(config.VOCAB_SIZE, config.DIM_MODEL)
        self.position_embedding = Embedding(config.MAX_LENGTH, config.DIM_MODEL)
        self.input_mask = InputMask(is_decoder=True)

        self._lmhead = LMHead(config.VOCAB_SIZE, config.DIM_MODEL)
        self._lmhead.weight = self.input_embedding.weight   # share parameter

        self.num_layers = config.NUM_LAYERS
        self.layers = LayerList([
            TransformerBlockGPT(config.DIM_MODEL, config.DIM_FF, config.DIM_KV, config.NUM_HEADS)
            for _ in range(self.num_layers)
        ])
        
        self.encoder_final_layer_nrom = GPTLayerNorm(config.DIM_MODEL)
        self.num_heads = config.NUM_HEADS
        self.dim_qkv = config.DIM_KV

        if config.MODEL_NAME is not None:
            # init parameter

            model_path = data.ensure_file(config.MODEL_NAME, "checkpoint.pt")
            vocab_path = data.ensure_file(config.MODEL_NAME, "vocab.txt")

            self.tokenizer = GPT2Tokenizer(vocab_path)

            self.device = config.DEVICE
            with self.device:
                logger.info("Start loading parameters from disk to cpu")
                self.load( open(model_path, "rb") )

                logger.info("Start loading parameters from cpu to gpu")
                
                load_stream = cupy.cuda.Stream()
                if self.nbytes + config.DYNAMIC_MEMORY > config.MEMORY_LIMIT:
                    raise ValueError("memory limit not enough, at least %d bytes, but got %d bytes" % (self.nbytes + config.DYNAMIC_MEMORY, config.MEMORY_LIMIT))
                
                logger.info("Using static loader: total: %d, dynamic_memory %d, memory_limit %d", self.nbytes, config.DYNAMIC_MEMORY, config.MEMORY_LIMIT)
                self.parameter_allocator = ReusedAllocator(self.nbytes)
                self.variable_allocator = SizeLimitedAllocator(config.MEMORY_LIMIT - self.nbytes)

                self.to_device(self.parameter_allocator, load_stream)
                
                self.device.synchronize()
                self.calc_stream = cupy.cuda.Stream(non_blocking=True)
                with self.calc_stream:
                    self.variable_allocator.alloc(config.DYNAMIC_MEMORY) # preallocate
                self.device.synchronize()

            logger.info("Cleaning useless parameters on cpu")
            self._remove_data()
            logger.info("End of model initialization")

    def encode(self, input_idx : np.ndarray, input_length : List[int]) -> Tuple[cupy.ndarray, GPTInferenceContext]:
        with self.device:
            calc_stream = self.calc_stream

            batch_size, seq_len = input_idx.shape
            with calc_stream:
                x = self.input_embedding.forward(self.variable_allocator, input_idx)
                encoder_attn_mask = self.input_mask.forward(self.variable_allocator, input_length, seq_len)
                x = x.transpose((0, 2, 1))      # (batch_size, dim_model, seq_len)
                assert x.dtype == cupy.float16

                x_pos = self.position_embedding.forward(self.variable_allocator, list(range(seq_len)))  # (seq_len, dim_model)
                x_pos = x_pos.T[cupy.newaxis]   # (1#batch_size, dim_model, seq_len)
                assert x_pos.shape == (1, self.dim_model, seq_len)
                assert x_pos.dtype == cupy.float16

                x += x_pos

                past_kv = self.variable_allocator.alloc_array((self.num_layers, 2, batch_size, self.num_heads, self.dim_qkv, self.max_length), dtype=cupy.float16)
                past_kv[:] = 0

            for i in range(self.num_layers):
                logger.info("Calc encoder layer %d", i)
                with calc_stream:
                    x = self.layers[i].forward(
                        self.variable_allocator, 
                        x,
                        encoder_attn_mask,
                        past_kv[i],
                        True
                    )
            with calc_stream:
                x = self.encoder_final_layer_nrom.forward(self.variable_allocator, x)
                last_out = []
                for i, pos in enumerate(input_length):
                    last_out.append( x[i, :, pos - 1] )
                last_out = cupy.stack(last_out)
                x = self._lmhead.forward(self.variable_allocator, last_out)
            calc_stream.synchronize()
            return x, GPTInferenceContext(past_kv, input_length)    # (batch, dim_model, seq_len)
    
    def decode_step(self,
            ctx : GPTInferenceContext,
            inputs : Union[List[int], np.ndarray]
        ) -> cupy.ndarray:
        past_kv = ctx.past_kv
        input_length = ctx.input_length
        step_input = inputs

        with self.device:
            calc_stream = self.calc_stream

            with calc_stream:
                x = self.input_embedding.forward(self.variable_allocator, step_input)    # (batch, dim_model)
                past_kv_mask = cupy.repeat(cupy.arange(self.max_length)[cupy.newaxis], len(inputs), axis=0)
                past_kv_mask = (past_kv_mask <= cupy.array(input_length, dtype=cupy.int64)[:, cupy.newaxis]) # (batch, max_len)
                x_pos = self.position_embedding.forward(self.variable_allocator, input_length)  # (batch, dim_model)
                x += x_pos
            for i in range(self.num_layers):
                logger.info("Calc decoder layer %d", i)

                with calc_stream:
                    x = self.layers[i].forward_partial(
                        self.variable_allocator,
                        x,                          # (batch, dim_model)
                        input_length,               # List[int]
                        past_kv[i],                 # (2, batch, num_heads, dim_qkv, max_length)
                        past_kv_mask,               # (batch, max_length)
                        True
                    )
            with calc_stream:
                x = self.encoder_final_layer_nrom.forward(self.variable_allocator, x[:, :, cupy.newaxis])[:, :, 0]
                x = self._lmhead.forward(self.variable_allocator, x)
        ctx.input_length = [ x + 1 for x in input_length ]
        calc_stream.synchronize()
        return x
    
    def _text_to_id(self, sentence):
        return self.tokenizer.encode(sentence)

    def _id_to_text(self, idx : List[int]):
        return self.tokenizer.decode(idx)
    
    def _get_token_id(self, token, use_unk):
        token = token.translate(self.tokenizer.translator_enc)
        if use_unk:
            return self.tokenizer.encoder.get(token, self.tokenizer.unk_id)
        else:
            return self.tokenizer.encoder.get(token, None)
    
    def _get_id_token(self, idx):
        return self.tokenizer.decoder[idx].translate(self.tokenizer.translator_dec)
    