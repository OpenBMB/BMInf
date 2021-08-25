from typing import Callable, Generator, List, Union
import cupy
from ..base import Model
from ...layers.transformer_block import TransformerBlockDecoder, TransformerBlockEncoder
from ...layers.encoder_kv import EncoderKeyValueProjection
from ...layers.position_bias import PositionBias
from ...layers.embedding import Embedding
from ...layers.layer_norm import LayerNorm
from ...layers.mask import InputMask
from ...layers.lm_head import LMHead
from ...layers.layer_list import LayerList
from .config import T5Configuration
from .tokenizer import T5Tokenizer
from ...allocator import ReusedAllocator, SizeLimitedAllocator
import numpy as np
import logging
from ... import data
from ...utils import round_up

logger = logging.getLogger(__name__)

class T5(Model):
    def __init__(self, config : T5Configuration):
        # Build Model
        logger.info("Building model")

        self.memory_overlap = config.MEMORY_OVERLAP
        if self.memory_overlap:
            self.overlap_layers = config.OVERLAP_LAYERS
        else:
            self.overlap_layers = max(config.NUM_ENCODER_LAYERS, config.NUM_DECODER_LAYERS)
        self.encoder_only = config.ENCODER_ONLY
        self.max_decoder_length = config.MAX_DECODER_LENGTH

        self.input_embedding = Embedding(config.VOCAB_SIZE, config.DIM_MODEL)
        self.input_mask = InputMask(is_decoder=False)

        self.encoder_position_bias = PositionBias(config.NUM_POSITION_BUCKETS, config.NUM_HEADS, is_decoder=False)
        self.num_encoder = config.NUM_ENCODER_LAYERS
        self.encoder = LayerList([
            TransformerBlockEncoder(config.DIM_MODEL, config.DIM_FF, config.DIM_KV, config.NUM_HEADS)
                for _ in range(config.NUM_ENCODER_LAYERS)
        ])
        self.encoder_final_layer_nrom = LayerNorm(config.DIM_MODEL)
        self.num_heads = config.NUM_HEADS
        self.dim_qkv = config.DIM_KV

        if not self.encoder_only:
            self.decoder_position_bias = PositionBias(config.NUM_POSITION_BUCKETS, config.NUM_HEADS, is_decoder=True)
            self.encoder_kv = EncoderKeyValueProjection(config.NUM_DECODER_LAYERS, config.DIM_MODEL, config.DIM_KV, config.NUM_HEADS)
            self.lm_head = LMHead(config.VOCAB_SIZE, config.DIM_MODEL)
            self.num_decoder = config.NUM_DECODER_LAYERS
            self.decoder = LayerList([
                TransformerBlockDecoder(config.DIM_MODEL, config.DIM_FF, config.DIM_KV, config.NUM_HEADS)
                    for _ in range(config.NUM_DECODER_LAYERS)
            ])
            self.decoder_final_layer_nrom = LayerNorm(config.DIM_MODEL)

        if config.MODEL_NAME is not None:
            # init parameter

            model_path = data.ensure_file(config.MODEL_NAME, "checkpoint.pt")
            vocab_path = data.ensure_file(config.MODEL_NAME, "vocab.txt")

            self.tokenizer = T5Tokenizer(vocab_path)

            self.device = config.DEVICE
            with self.device:
                logger.info("Start loading parameters from disk to cpu")
                self.load( open(model_path, "rb") )

                logger.info("Start loading parameters from cpu to gpu")
                
                load_stream = cupy.cuda.Stream()
                
                if self.memory_overlap:
                    mx_size = 0
                    for i in range(config.NUM_ENCODER_LAYERS):
                        mx_size = max(self.encoder[i].nbytes, mx_size)
                    for i in range(config.NUM_DECODER_LAYERS):
                        mx_size = max(self.decoder[i].nbytes, mx_size)

                    overlap_size = mx_size * self.overlap_layers * 4
                    other_size = self.nbytes - self.encoder.nbytes - self.decoder.nbytes

                    logger.info("Using overlap loader: overlap_size %d, other_size: %d, dynamic_memory %d, memory_limit %d", overlap_size, other_size, config.DYNAMIC_MEMORY, config.MEMORY_LIMIT)
                    if overlap_size + other_size + config.DYNAMIC_MEMORY > config.MEMORY_LIMIT:
                        raise ValueError("memory limit not enough, at least %d bytes, bug got %d bytes" % (overlap_size + other_size + config.DYNAMIC_MEMORY, config.MEMORY_LIMIT))
                    self.parameter_allocator = ReusedAllocator(other_size + (overlap_size // 2))
                    self.overlap_allocator = [ReusedAllocator(overlap_size // 4), ReusedAllocator(overlap_size // 4)]

                    self.variable_allocator = SizeLimitedAllocator(config.MEMORY_LIMIT - other_size - overlap_size)

                    self.variable_allocator.alloc(config.DYNAMIC_MEMORY) # preallocate

                    for name, layer in self._sub_layers.items():
                        if name in ["encoder", "decoder"]:
                            # move first overlap_size layers to device
                            for i in range(min(self.overlap_layers, len(layer))):
                                layer[i].to_device( self.parameter_allocator, load_stream )
                        else:
                            layer.to_device( self.parameter_allocator, load_stream  )
                else:
                    if self.nbytes + config.DYNAMIC_MEMORY < config.MEMORY_LIMIT:
                        raise ValueError("memory limit not enough, at least %d bytes, bug got %d bytes" % (self.nbytes + config.DYNAMIC_MEMORY, config.MEMORY_LIMIT))
                    
                    logger.info("Using static loader: total: %d, dynamic_memory %d, memory_limit %d", self.nbytes, config.DYNAMIC_MEMORY, config.MEMORY_LIMIT)
                    self.parameter_allocator = ReusedAllocator(self.nbytes)
                    self.variable_allocator = SizeLimitedAllocator(config.MEMORY_LIMIT - self.nbytes)

                    self.variable_allocator.alloc(config.DYNAMIC_MEMORY) # preallocate

                    self.to_device(self.parameter_allocator, load_stream)
                
                self.load_stream = cupy.cuda.Stream()
                self.device.synchronize()

            logger.info("Cleaning useless parameters on cpu")
            if self.memory_overlap:
                for name, layer in self._sub_layers.items():
                    if name in ["encoder", "decoder"]:
                        # move first overlap_size layers to device
                        for i in range(self.overlap_layers):
                            layer[i]._remove_data()
                    else:
                        layer._remove_data()
            else:
                self._remove_data()
            logger.info("End of model initialization")


    def encode(self, input_idx : np.ndarray, input_length : List[int]):
        with self.device:
            load_stream = self.load_stream
            calc_stream = cupy.cuda.get_current_stream()
            load_event = load_stream.record()
            calc_event = calc_stream.record()

            batch_size, seq_len = input_idx.shape

            if seq_len % 16 != 0:
                nw_seq_len = round_up(seq_len, 16)  # round up
                nw_input_idx = np.zeros((batch_size, nw_seq_len), dtype=np.int64)
                nw_input_idx[:, :seq_len] = input_idx
                seq_len = nw_seq_len
                input_idx = nw_input_idx
                del nw_seq_len
                del nw_input_idx

            x = self.input_embedding.forward(self.variable_allocator, input_idx)
            encoder_attn_mask = self.input_mask.forward(self.variable_allocator, input_length, seq_len)
            x = x.transpose((0, 2, 1))
            assert x.dtype == cupy.float16

            x_pos = self.encoder_position_bias.forward(self.variable_allocator, seq_len, seq_len)
            assert x_pos.shape == (1, self.num_heads, seq_len, seq_len)
            assert x_pos.dtype == cupy.float16

            for i in range(self.num_encoder):
                if i % self.overlap_layers == 0:
                    calc_stream.wait_event(load_event)
                logger.info("Calc encoder layer %d", i)
                x = self.encoder[i].forward(
                    self.variable_allocator, 
                    x,
                    encoder_attn_mask,
                    x_pos,
                    True
                )
                if i % self.overlap_layers == self.overlap_layers - 1 and i + 1 < self.num_encoder:
                    overlap_idx = ((i + 1) // self.overlap_layers) % 2
                    olp_allocator = self.overlap_allocator[overlap_idx]
                    olp_allocator.reset()
                    load_stream.wait_event(calc_event)
                    for j in range(i + 1, min(i + self.overlap_layers + 1, self.num_encoder)):
                        logger.info("Load encoder layer %d", j)
                        self.encoder[j].to_device(olp_allocator, load_stream)
                    
                    calc_event = calc_stream.record()
                    load_event = load_stream.record()
            x = self.encoder_final_layer_nrom.forward(self.variable_allocator, x)
            return x    # (batch, dim_model, seq_len)

    def decode(self, hidden_state, input_length, sampler : Union[str, Callable[[cupy.ndarray], int] ] = "random") -> Generator[int, None, None]:
        if self.encoder_only:
            raise ValueError("T5-encoder only")

        if isinstance(sampler, str):
            if sampler == "greedy":
                from ...utils import greedy_sampler
                sampler = greedy_sampler
            elif sampler == "random":
                from ...utils import random_sampler
                sampler = random_sampler
            else:
                raise ValueError("Unknown sampler type %s" % sampler)
        
        with self.device:
            batch_size, _, seq_ipt_len = hidden_state.shape

            # (batch, num_decoder, 2, num_heads, dim_kv, seq_ipt_len),
            encoder_layers_kv = self.encoder_kv.forward(self.variable_allocator, hidden_state)

            # (1, num_heads, max_decoder_length, max_decoder_length)
            dec_pos = self.decoder_position_bias.forward(
                self.variable_allocator,
                self.max_decoder_length,
                self.max_decoder_length
            )

            past_kv = self.variable_allocator.alloc_array((self.num_decoder, batch_size, 2, self.num_heads, self.dim_qkv, self.max_decoder_length), dtype=cupy.float32)
            past_kv[:] = 0
            
            encoder_mask = self.input_mask.forward(self.variable_allocator, input_length, seq_ipt_len)[:, :, 0]

            last_ipt = [1] * batch_size

        for i in range(self.max_decoder_length):
            with self.device:
                x = self.decode_step(
                    past_kv,
                    encoder_layers_kv,
                    dec_pos,
                    encoder_mask,
                    last_ipt,
                    i,
                )
                last_ipt = [
                    sampler(x[i]) for i in range(batch_size)
                ]
                yield last_ipt
    
    def decode_step(self, 
            past_kv : cupy.ndarray,                     # (num_decoder, batch, 2, num_heads, dim_kv, max_decoder_length)
            encoder_layers_kv : cupy.ndarray,           # (batch, num_decoder, 2, num_heads, dim_kv, seq_ipt_len)
            dec_position_bias : cupy.ndarray,           # (1, num_heads, max_decoder_length, max_decoder_length)
            encoder_mask : cupy.ndarray,                # (batch, seq_ipt_len)
            step_input : Union[List[int], np.ndarray],  # (batch,)
            step_pos : int
        ):
        with self.device:
            load_stream = self.load_stream
            calc_stream = cupy.cuda.get_current_stream()
            load_event = load_stream.record()
            calc_event = calc_stream.record()

            x = self.input_embedding.forward(self.variable_allocator, step_input)    # (batch, dim_model)
            for i in range(self.num_decoder):
                if i % self.overlap_layers == 0:
                    calc_stream.wait_event(load_event)
                logger.info("Calc decoder layer %d", i)

                x = self.decoder[i].forward(
                    self.variable_allocator,
                    x,                          # (batch, dim_model)
                    past_kv[i],                 # (batch, 2, num_heads, dim_kv, max_decoder_length)
                    step_pos,                   # 1
                    encoder_mask,               # (batch, seq_ipt_len)
                    encoder_layers_kv[:, i],    # (batch, 2, num_heads, dim_kv, seq_ipt_len)
                    dec_position_bias,          # (1, num_heads, max_decoder_length, max_decoder_length)
                    True
                )
                if i % self.overlap_layers == self.overlap_layers - 1 and i + 1 < self.num_decoder:
                    overlap_idx = ((i + 1) // self.overlap_layers) % 2
                    olp_allocator = self.overlap_allocator[overlap_idx]
                    olp_allocator.reset()
                    load_stream.wait_event(calc_event)
                    for j in range(i + 1, min(i + self.overlap_layers + 1, self.num_decoder)):
                        logger.info("Load decoder layer %d", j)
                        self.decoder[j].to_device(olp_allocator, load_stream)
                    
                    calc_event = calc_stream.record()
                    load_event = load_stream.record()
            x = self.decoder_final_layer_nrom.forward(self.variable_allocator, x[:, :, cupy.newaxis])[:, :, 0]
            x = self.lm_head.forward(self.variable_allocator, x)
            return x
    
    def text_to_id(self, sentence):
        return self.tokenizer.encode(sentence)

    def id_to_text(self, idx : List[int]):
        return self.tokenizer.decode(idx)
    
    def get_token_id(self, token : str, use_unk : bool = True) -> Union[int, None]:
        token = token.translate(self.tokenizer.translator_enc)
        if use_unk:
            return self.tokenizer.encoder.get(token, self.tokenizer.unk_id)
        else:
            return self.tokenizer.encoder.get(token, None)
    
    def get_id_token(self, idx : int) -> str:
        return self.tokenizer.decoder[idx].translate(self.tokenizer.translator_dec)