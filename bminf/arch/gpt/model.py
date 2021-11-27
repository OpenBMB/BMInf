import logging
from typing import List, Optional
from .config import GPTConfiguration
from ...core import Model, Layer, Context, Tensor, Device
from ...core.allocators.cuda import CUDAAllocator
from ...layers import Embedding, LayerList, Layernorm, DecoderBlock
from ..scheduler import LayerScheduler, calc_fixed_layers
from .tokenizer import GPT2Tokenizer
import cpm_kernels.kernels as ck
import numpy as np
import math
from ... import data

logger = logging.getLogger(__name__)

class GPT2Model(Model):
    def __init__(self, config : GPTConfiguration):
        super().__init__()

        self.config = config
        if self.config.DEVICE is None:
            device = Device.current()
        else:
            device = Device(self.config.DEVICE)
        self.device = device
        self.allocator = CUDAAllocator(self.device.idx)
        self.num_layers = self.config.NUM_LAYERS

        self.token_embedding = Embedding(config.VOCAB_SIZE, config.DIM_MODEL)
        self.position_embedding = Embedding(config.MAX_LENGTH, config.DIM_MODEL)
        self.layernorm = Layernorm(config.DIM_MODEL, config.EPS, bias=True)

        self.layers = LayerList([
            DecoderBlock(config.DIM_MODEL, config.NUM_HEADS, config.DIM_HEAD, config.DIM_FF, config.EPS, bias=True, gated=False, attn_scale=1.0/math.sqrt(config.DIM_HEAD))
                for _ in range(self.num_layers)
        ], offset=False)
        self.max_block_nbytes = self.layers[0].nbytes
        fixed_layers : List[Layer] = [ self.token_embedding, self.position_embedding, self.layernorm ]

        if config.MODEL_NAME is not None:
            fixed_size = sum([x.nbytes for x in fixed_layers])
            if config.MEMORY_LIMIT is None:
                config.MEMORY_LIMIT = min(device.free_memory, self.nbytes)
            layers_memory_limit = config.MEMORY_LIMIT - fixed_size
            max_layers_in_memory = layers_memory_limit // self.max_block_nbytes

            if max_layers_in_memory < 2:
                raise RuntimeError("CUDA Error: out of memory: at least %d" % (self.max_block_nbytes * 2))
            swap_buffers = 2
            if max_layers_in_memory >= self.num_layers:
                swap_buffers = 0
            num_fixed_layers = max_layers_in_memory - swap_buffers
            
            logger.info("========= GPT =========")
            logger.info("Total layers: %d" , self.num_layers)
            logger.info("OnDev layers: %d", max_layers_in_memory)
            logger.info("Fixed layers: %s", calc_fixed_layers(self.num_layers, num_fixed_layers) )

            for layer_id in calc_fixed_layers(self.num_layers, num_fixed_layers):
                self.layers[layer_id].is_fixed = True
            for layer in fixed_layers:
                layer.init_data(pinned=False)
            for i in range(self.num_layers):
                if self.layers[i].is_fixed:
                    self.layers[i].init_data(pinned=False)
                else:
                    self.layers[i].init_data(pinned=True)
                self.layers[i].locked = False
                self.layers[i].on_device = False
                self.layers[i].loader_event = device.create_event()

            model_path = data.ensure_file(config.MODEL_NAME, "checkpoint.pt")
            vocab_path = data.ensure_file(config.MODEL_NAME, "vocab.txt")

            self.tokenizer = GPT2Tokenizer(vocab_path)
            self.load(open(model_path, "rb"))

            for layer in fixed_layers:
                layer.is_fixed = True
                layer.locked = False
                layer.on_device = True
                with device:
                    device_ptr = self.allocator.allocate(layer.nbytes)
                    layer._to_device(device_ptr)
                layer.data = None
            for i in range(self.num_layers):
                if self.layers[i].is_fixed:
                    self.layers[i].locked = False
                    self.layers[i].on_device = True
                    with device:
                        device_ptr = self.allocator.allocate(self.layers[i].nbytes)
                        self.layers[i]._to_device(device_ptr)
                    self.layers[i].data = None
            with device:
                self.loader_stream = device.create_stream()
                self.scheduler = LayerScheduler(self.allocator, 2, self.max_block_nbytes, self.loader_stream)
            
        else:
            for layer in fixed_layers:
                layer.init_data()
            for i in range(self.num_layers):
                self.layers[i].init_data()
    
    def allocate_decode_buffer(self, ctx : Context, batch : int, length : int) -> List[Tensor]:
        return [ 
            ctx.allocate((batch, self.config.NUM_HEADS, length, self.config.DIM_HEAD), dtype=np.float16)
                for _ in range(self.num_layers)
        ]

    def embedding(self, 
            ctx : Context, 
            ids : np.ndarray,       # (batch, seq_len)  int32
            position : np.ndarray,  # (batch, seq_len)  int32
            x_out : Tensor          # (batch, dim_model, seq_len)
        ):
        batch, dim_model, seq_len = x_out.shape
        tensor_ids = Tensor.from_numpy(ctx, ids)
        position_ids = Tensor.from_numpy(ctx, position)
        tmp = ctx.allocate(x_out.shape, np.float16)
        self.token_embedding.embedding_forward(ctx, tensor_ids, x_out)
        self.position_embedding.embedding_forward(ctx, position_ids, tmp)
        ck.arith_element_add(
            batch, seq_len * dim_model,
            x_out.ptr, tmp.ptr,
            x_out.ptr,
            ctx.current_stream
        )
        ctx.free(tmp)
        ctx.free(position_ids)
        ctx.free(tensor_ids)
    
    def embedding_step(self,
            ctx : Context,
            ids : np.ndarray,       # (batch,)  int32
            pos : np.ndarray,       # (batch,)  int32
            x_out : Tensor          # (batch, dim_model)
        ):
        batch, dim_model = x_out.shape
        tensor_ids = Tensor.from_numpy(ctx, ids)
        pos_ids = Tensor.from_numpy(ctx, pos)
        tmp = ctx.allocate(x_out.shape, np.float16)
        self.token_embedding.embedding_step(ctx, tensor_ids, x_out)
        self.position_embedding.embedding_step(ctx, pos_ids, tmp)
        ck.arith_element_add(
            batch, dim_model,
            x_out.ptr, tmp.ptr,
            x_out.ptr,
            ctx.current_stream
        )
        ctx.free(tmp)
        ctx.free(pos_ids)
        ctx.free(tensor_ids)
    
    def encode(self, 
            ctx : Context, 
            x : Tensor,                             # (batch. dim_model, seq_len)
            x_mask : np.ndarray,                    # (batch, seq_len)
            key_out : Optional[List[Tensor]] = None,      # (batch, num_head, seq_len, dim_head)
            value_out : Optional[List[Tensor]] = None,    # (batch, num_head, seq_len, dim_head)
        ):
        batch, dim_model, seq_len = x.shape
        assert x_mask.shape == (batch, seq_len)
        self_attn_mask = x_mask[:, :, np.newaxis] & x_mask[:, np.newaxis, :] & (np.arange(seq_len)[:, np.newaxis] <= np.arange(seq_len)[np.newaxis, :])
        tensor_mask = Tensor.from_numpy(ctx, self_attn_mask)

        layer_order = list(range(self.num_layers))
        for i, layer in zip(layer_order, self.scheduler.loop_layers(ctx, self.layers, layer_order)):
            layer.forward(
                ctx,
                x,
                tensor_mask,
                None,   # no position bias
                x,
                key_out[i] if key_out is not None else None,
                value_out[i] if key_out is not None else None
            )

        self.layernorm.forward(ctx, x, x)
        ctx.free(tensor_mask)
    
    def projection(self, ctx : Context, hidden : Tensor, logits_out : Tensor, output_one : Optional[int] = None):
        batch, dim_model, seq_len = hidden.shape
        if output_one is not None:
            x_pos = ctx.allocate((batch, dim_model), np.float16)
            ck.utils.copy_pos_hidden(
                batch, dim_model, seq_len, output_one,
                hidden.ptr, x_pos.ptr,
                ctx.current_stream
            )
            self.token_embedding.projection_step(
                ctx,
                x_pos,
                logits_out
            )
            ctx.free(x_pos)
        else:
            self.token_embedding.projection_forward(ctx, hidden, logits_out)
        
        
    def step(self,
            ctx : Context,
            step_input : Tensor,        # (batch, dim_model)
            buffer_k : List[Tensor],    # List[(batch, num_heads, buffer_len, dim_head)]
            buffer_v : List[Tensor],    # List[(batch, num_heads, buffer_len, dim_head)]
            step_pos : int,             # int
        ):
        batch = step_input.shape[0]
        buffer_len = buffer_k[0].shape[2]

        self_attn_mask = (np.arange(buffer_len) <= step_pos)[np.newaxis].repeat(batch, axis=0)
        tensor_mask_self = Tensor.from_numpy(ctx, self_attn_mask)
        layer_order = list(range(self.num_layers))
        for i, layer in zip(layer_order, self.scheduler.loop_layers(ctx, self.layers, layer_order)):
            layer.step(
                ctx,
                step_input,
                tensor_mask_self,
                None,   # no position bias
                buffer_k[i],
                buffer_v[i],
                step_pos,
                step_input
            )
        self.layernorm.step(ctx, step_input, step_input)
        ctx.free(tensor_mask_self)
    
    def projection_step(self, ctx : Context, x : Tensor, logits_out : Tensor):
        self.token_embedding.projection_step(ctx, x, logits_out)
    
    def encode_requires_grad(self,
            ctx : Context,
            x : Tensor,                             # (batch. dim_model, seq_len)
            x_mask : np.ndarray,                    # (batch, seq_len)
            hidden_list : List[Tensor]              # List[(batch, dim_model, seq_len)]
        ):
        batch, dim_model, seq_len = x.shape
        assert x_mask.shape == (batch, seq_len)
        self_attn_mask = x_mask[:, :, np.newaxis] & x_mask[:, np.newaxis, :] & (np.arange(seq_len)[:, np.newaxis] <= np.arange(seq_len)[np.newaxis, :])
        tensor_mask = Tensor.from_numpy(ctx, self_attn_mask)

        for hidden_buffer, layer in zip(hidden_list, self.scheduler.loop_layers(ctx, self.layers, list(range(self.num_layers)))):
            layer.forward(
                ctx,
                x,
                tensor_mask,
                None,   # no position bias
                x,
            )
            hidden_buffer.copy_(ctx, x)
        
        self.layernorm.forward(ctx, x, x)
        ctx.free(tensor_mask)
    
    def encode_backward(self,
            ctx : Context,
            x : Tensor,                 # (batch. dim_model, seq_len)
            x_mask : np.ndarray,        # (batch, seq_len)
            hidden_list : List[Tensor], # List[(batch, dim_model, seq_len)]
            grad : Tensor               # (batch. dim_model, seq_len)
        ):
        batch, dim_model, seq_len = grad.shape
        assert len(hidden_list) == self.num_layers
        layer_inputs = [x] + hidden_list[:-1]
        layer_output = hidden_list[-1]

        self_attn_mask = x_mask[:, :, np.newaxis] & x_mask[:, np.newaxis, :] & (np.arange(seq_len)[:, np.newaxis] <= np.arange(seq_len)[np.newaxis, :])
        tensor_mask = Tensor.from_numpy(ctx, self_attn_mask)

        tmp_grad = ctx.allocate(grad.shape, np.float16)
        tmp_grad.zero_(ctx)
        self.layernorm.backward(ctx, layer_output, grad, tmp_grad)
        grad.copy_(ctx, tmp_grad)
        ctx.free(tmp_grad)

        for layer, layer_input in zip(self.scheduler.loop_layers(
            ctx,
            self.layers,
            list(reversed(range(self.num_layers)))), layer_inputs[::-1]
        ):
            layer.backward(
                ctx,
                layer_input,
                tensor_mask,
                None,
                grad
            )
        ctx.free(tensor_mask)
    
    def projection_backward(self,
            ctx : Context, 
            grad_output : Tensor,   # (batch, seq_len, vocab_size)
            grad : Tensor           # (batch, dim_model, seq_len)
        ):
        self.token_embedding.projection_backward(ctx, grad_output, grad)