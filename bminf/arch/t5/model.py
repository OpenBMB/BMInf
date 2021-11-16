from typing import List, Optional, Tuple
from cpm_kernels.library import cudart
from .config import T5Configuration
from ...core import Model, Layer, Context, Tensor, Device
from ...core.allocators.cuda import CUDAAllocator
from ...layers import LayerList, EncoderBlock, DecoderBlockWithCrossAttention
from ...layers import  Embedding, Layernorm, PositionEmbedding, OutputLogits
from ... import data
from .tokenizer import T5Tokenizer
from ..scheduler import LayerScheduler
import numpy as np

class T5Model(Model):
    def __init__(self, config : T5Configuration):
        super().__init__()

        self.config = config
        if self.config.DEVICE is None:
            self.config.DEVICE = cudart.cudaGetDevice()

        device = Device(self.config.DEVICE)
        self.allocator = CUDAAllocator(self.config.DEVICE)

        self.num_enc = config.NUM_ENCODER_LAYERS
        self.num_dec = config.NUM_DECODER_LAYERS
        self.eps = 1e-5
        
        self.enc_layers = LayerList([
            EncoderBlock(config.DIM_MODEL, config.NUM_HEADS, config.DIM_HEAD, config.DIM_FF, self.eps, bias=False, gated=True) for _ in range(self.num_enc)
        ], offset=False)

        self.dec_layers = LayerList([
            DecoderBlockWithCrossAttention(config.DIM_MODEL, config.NUM_HEADS, config.DIM_HEAD, config.DIM_FF, self.eps, bias=False, gated=True) for _ in range(self.num_enc)
        ], offset=False)
        self.max_block_nbytes = max( self.enc_layers[0].nbytes, self.dec_layers[0].nbytes)
        self.ln_enc = Layernorm(config.DIM_MODEL, self.eps, bias=False)
        self.ln_dec = Layernorm(config.DIM_MODEL, self.eps, bias=False)

        self.input_embedding = Embedding(config.VOCAB_SIZE, config.DIM_MODEL)
        self.output_embedding = OutputLogits(config.VOCAB_SIZE, config.DIM_MODEL)

        self.position_bias_enc = PositionEmbedding(config.NUM_HEADS, config.NUM_POSITION_BUCKETS, config.MAX_DISTANCE, bidirectional=True)
        self.position_bias_dec = PositionEmbedding(config.NUM_HEADS, config.NUM_POSITION_BUCKETS, config.MAX_DISTANCE, bidirectional=False)

        fixed_layers : List[Layer] = [self.ln_enc, self.ln_dec, self.input_embedding, self.output_embedding, self.position_bias_enc, self.position_bias_dec]

        if config.MODEL_NAME is not None:
            

            fixed_size = sum([x.nbytes for x in fixed_layers])

            if config.MEMORY_LIMIT is None:
                with device:
                    config.MEMORY_LIMIT = min(cudart.cudaMemGetInfo()[0], self.nbytes)
            
            layers_memory_limit = config.MEMORY_LIMIT - fixed_size
            max_layers_in_memory = layers_memory_limit // self.max_block_nbytes

            if max_layers_in_memory < 2:
                raise RuntimeError("CUDA out of memory: at least %d" % (self.max_block_nbytes * 2))
            
            on_device_enc_layers = min((max_layers_in_memory - 2) // 2, self.num_enc)
            on_device_dec_layers = min(max_layers_in_memory - 2 - on_device_enc_layers, self.num_dec)

            for layer in fixed_layers:
                layer.init_data(pinned=False)
            
            for i in range(self.num_enc):
                if i < on_device_enc_layers:
                    self.enc_layers[i].init_data(pinned=False)
                    self.enc_layers[i].is_fixed = True
                else:
                    self.enc_layers[i].init_data(pinned=True)
                    self.enc_layers[i].is_fixed = False
                self.enc_layers[i].locked = False
                self.enc_layers[i].on_device = False
                self.enc_layers[i].loader_event = cudart.cudaEventCreate()
            
            for i in range(self.num_dec):
                if i < on_device_dec_layers:
                    self.dec_layers[i].init_data(pinned=False)
                    self.dec_layers[i].is_fixed = True
                else:
                    self.dec_layers[i].init_data(pinned=True)
                    self.dec_layers[i].is_fixed = False
                self.dec_layers[i].locked = False
                self.dec_layers[i].on_device = False
                self.dec_layers[i].loader_event = cudart.cudaEventCreate()
            
            model_path = data.ensure_file(config.MODEL_NAME, "checkpoint.pt")
            vocab_path = data.ensure_file(config.MODEL_NAME, "vocab.txt")

            self.tokenizer = T5Tokenizer(vocab_path)
            self.load( open(model_path, "rb") )

            for layer in fixed_layers:
                device_ptr = self.allocator.allocate(layer.nbytes)
                layer.is_fixed = True
                layer.locked = False
                layer.on_device = True

                layer._to_device(device_ptr)
                layer.data = None
                
            for i in range(on_device_enc_layers):
                device_ptr = self.allocator.allocate(self.enc_layers[i].nbytes)
                self.enc_layers[i].is_fixed = True
                self.enc_layers[i].locked = False
                self.enc_layers[i].on_device = True

                self.enc_layers[i]._to_device(device_ptr)
                self.enc_layers[i].data = None
            
            for i in range(on_device_dec_layers):
                device_ptr = self.allocator.allocate(self.dec_layers[i].nbytes)
                self.dec_layers[i].is_fixed = True
                self.dec_layers[i].locked = False
                self.dec_layers[i].on_device = True

                self.dec_layers[i]._to_device(device_ptr)
                self.dec_layers[i].data = None
            
            
            self.loader_stream = cudart.cudaStreamCreate()
            self.scheduler = LayerScheduler(self.allocator, 2, self.max_block_nbytes, self.loader_stream)
        else:
            # only init data
            for layer in fixed_layers:
                layer.init_data(pinned=False)
            for i in range(self.num_enc):
                self.enc_layers[i].init_data(pinned=False)
            for i in range(self.num_dec):
                self.dec_layers[i].init_data(pinned=False)
            
        
    def embedding(self, 
            ctx : Context, 
            ids : np.ndarray,       # (batch, seq_len)
            x_out : Tensor          # (batch, dim_model, seq_len)
        ):
        tensor_ids = Tensor.from_numpy(ctx, ids)

        self.input_embedding.forward(ctx, tensor_ids, x_out)
        ctx.free(tensor_ids)
    
    def embedding_step(self,
            ctx : Context,
            ids : np.ndarray,       # (batch,)  int32
            x_out : Tensor          # (batch, dim_model)
        ):
        tensor_ids = Tensor.from_numpy(ctx, ids)

        self.input_embedding.step(ctx, tensor_ids, x_out)
        ctx.free(tensor_ids)


    def encode(self, 
            ctx : Context, 
            x : Tensor,             # (batch. dim_model, seq_len)
            x_mask : np.ndarray     # (batch, seq_len)
        ):
        batch, dim_model, seq_len = x.shape
        assert x_mask.shape == (batch, seq_len)
        self_attn_mask = x_mask[:, :, np.newaxis] & x_mask[:, np.newaxis, :]

        tensor_mask = Tensor.from_numpy(ctx, self_attn_mask)

        position_bias = ctx.allocate((self.config.NUM_HEADS, seq_len, seq_len), dtype=np.float16)
        self.position_bias_enc.forward(ctx, seq_len, seq_len, position_bias)

        for layer in self.scheduler.loop_layers(ctx, self.enc_layers, list(range(self.num_enc))):
            layer.forward(
                ctx,
                x,
                position_bias,
                tensor_mask,
                x
            )
        
        self.ln_enc.forward(ctx, x, x)
        ctx.free(tensor_mask)
        ctx.free(position_bias)
    
    def decode(self,
            ctx : Context,
            decoder_input : Tensor,         # (batch, dim_model, seq_q)
            encoder_output : Tensor,        # (batch, dim_model, seq_k)
            decoder_mask : np.ndarray,      # (batch, dim_model, seq_q)
            encoder_mask : np.ndarray,      # (batch, dim_model, seq_k)
            decoder_output : Tensor         # (batch, seq_q, vocab_size)
        ):
        assert decoder_input.shape[:2] == encoder_output.shape[:2]
        batch, dim_model, seq_q = decoder_input.shape
        batch, dim_model, seq_k = encoder_output.shape
        assert decoder_mask.shape == (batch, seq_q)
        assert encoder_mask.shape == (batch, seq_k)

        self_attn_mask = decoder_mask[:, :, np.newaxis] & decoder_mask[:, np.newaxis, :] & (np.arange(seq_q)[:, np.newaxis] <= np.arange(seq_q)[np.newaxis, :])
        cross_attn_mask = encoder_mask[:, :, np.newaxis] & decoder_mask[:, np.newaxis, :]
        tensor_mask_self_attn = Tensor.from_numpy(ctx, self_attn_mask)
        tensor_mask_cross_attn = Tensor.from_numpy(ctx, cross_attn_mask)

        position_bias = ctx.allocate((self.config.NUM_HEADS, seq_q, seq_q), dtype=np.float16)
        self.position_bias_dec.forward(ctx, seq_q, seq_q, position_bias)

        for layer in self.scheduler.loop_layers(ctx, self.dec_layers, list(range(self.num_dec))):
            layer.forward(
                ctx,
                decoder_input,
                encoder_output,
                tensor_mask_self_attn,
                tensor_mask_cross_attn,
                position_bias,
                None,
                decoder_input
            )
        
        self.ln_dec.forward(ctx, decoder_input, decoder_input)
        self.output_embedding.forward(ctx, decoder_input, decoder_output)
        ctx.free(tensor_mask_self_attn)
        ctx.free(tensor_mask_cross_attn)
        ctx.free(position_bias)

    def allocate_decode_buffer(self, ctx : Context, batch : int, length : int) -> List[Tensor]:
        return [ 
            ctx.allocate((batch, self.config.NUM_HEADS, length, self.config.DIM_HEAD), dtype=np.float16)
                for _ in range(self.num_dec)
        ]

    def decode_step(self,
            ctx : Context,
            step_input : Tensor,                # (batch, dim_model)
            encoder_output : Optional[Tensor],  # (batch, dim_model, seq_k) # needed for step_pos == 0
            # mask_x is not needed
            mask_encoder : Tensor,              # (batch, seq_k)

            buffer_k_self : List[Tensor],       # List[(batch, num_heads, buffer_len, dim_head)]
            buffer_v_self : List[Tensor],       # List[(batch, num_heads, buffer_len, dim_head)]
            buffer_k_cross : List[Tensor],      # List[(batch, num_heads, seq_k, dim_head)]
            buffer_v_cross : List[Tensor],      # List[(batch, num_heads, seq_k, dim_head)]

            step_pos : int,
            step_out : Tensor,                  # (batch, vocab_size)
        ):
        batch = step_input.shape[0]
        buffer_len = buffer_k_self[0].shape[2]

        tensor_mask_cross_attn = Tensor.from_numpy(ctx, mask_encoder)

        self_attn_mask = (np.arange(buffer_len) <= step_pos)[np.newaxis].repeat(batch, axis=0)
        tensor_mask_self_attn = Tensor.from_numpy(ctx, self_attn_mask)

        position_bias = ctx.allocate((self.config.NUM_HEADS, buffer_len), dtype=np.float16)
        self.position_bias_dec.step(ctx, buffer_len, step_pos, position_bias)

        for i, layer in enumerate(self.scheduler.loop_layers(ctx, self.dec_layers, list(range(self.num_dec)))):
            layer.step(
                ctx,
                step_input,
                encoder_output,
                tensor_mask_self_attn,
                tensor_mask_cross_attn,
                position_bias,
                None,
                buffer_k_self[i],
                buffer_v_self[i],
                buffer_k_cross[i],
                buffer_v_cross[i],
                step_pos,
                step_input
            )
        self.ln_dec.step(ctx, step_input, step_input)
        self.output_embedding.step(ctx, step_input, step_out)
        ctx.free(tensor_mask_self_attn)
        ctx.free(tensor_mask_cross_attn)
        ctx.free(position_bias)
    
    def encode_requires_grad(self,
            ctx : Context,
            x : Tensor,                 # (batch. dim_model, seq_len)
            x_mask : np.ndarray,        # (batch, seq_len)
            hidden_list : List[Tensor]
        ):
        batch, dim_model, seq_len = x.shape
        assert x_mask.shape == (batch, seq_len)
        self_attn_mask = x_mask[:, :, np.newaxis] & x_mask[:, np.newaxis, :]

        assert len(hidden_list) == self.num_enc
        for i in range(self.num_enc):
            assert hidden_list[i].shape == (batch, dim_model, seq_len) and hidden_list[i].dtype == np.float16

        tensor_mask = Tensor.from_numpy(ctx, self_attn_mask)

        position_bias = ctx.allocate((self.config.NUM_HEADS, seq_len, seq_len), dtype=np.float16)
        self.position_bias_enc.forward(ctx, seq_len, seq_len, position_bias)

        for hidden_buffer, layer in zip(hidden_list, self.scheduler.loop_layers(ctx, self.enc_layers, list(range(self.num_enc)))):
            layer.forward(
                ctx,
                x,
                position_bias,
                tensor_mask,
                x
            )
            cudart.cudaMemcpyAsync(hidden_buffer.ptr, x.ptr, x.nbytes, cudart.cudaMemcpyDeviceToDevice, ctx.current_stream)
        
        self.ln_enc.forward(ctx, x, x)
        ctx.free(tensor_mask)
        ctx.free(position_bias)

    def decode_requires_grad(self,
            ctx : Context,
            decoder_input : Tensor,         # (batch, dim_model, seq_q)
            encoder_output : Tensor,        # (batch, dim_model, seq_k)
            decoder_mask : np.ndarray,      # (batch, dim_model, seq_q)
            encoder_mask : np.ndarray,      # (batch, dim_model, seq_k)
            decoder_output : Tensor,        # (batch, seq_q, vocab_size)
            hidden_list : List[Tensor]
        ):
        assert decoder_input.shape[:2] == encoder_output.shape[:2]
        batch, dim_model, seq_q = decoder_input.shape
        batch, dim_model, seq_k = encoder_output.shape
        assert decoder_mask.shape == (batch, seq_q)
        assert encoder_mask.shape == (batch, seq_k)

        assert len(hidden_list) == self.num_dec
        for i in range(self.num_dec):
            assert hidden_list[i].shape == (batch, dim_model, seq_q) and hidden_list[i].dtype == np.float16

        self_attn_mask = decoder_mask[:, :, np.newaxis] & decoder_mask[:, np.newaxis, :] & (np.arange(seq_q)[:, np.newaxis] <= np.arange(seq_q)[np.newaxis, :])
        cross_attn_mask = encoder_mask[:, :, np.newaxis] & decoder_mask[:, np.newaxis, :]
        tensor_mask_self_attn = Tensor.from_numpy(ctx, self_attn_mask)
        tensor_mask_cross_attn = Tensor.from_numpy(ctx, cross_attn_mask)

        position_bias = ctx.allocate((self.config.NUM_HEADS, seq_q, seq_q), dtype=np.float16)
        self.position_bias_dec.forward(ctx, seq_q, seq_q, position_bias)

        for hidden_buffer, layer in zip(hidden_list, self.scheduler.loop_layers(ctx, self.dec_layers, list(range(self.num_dec)))):
            layer.forward(
                ctx,
                decoder_input,
                encoder_output,
                tensor_mask_self_attn,
                tensor_mask_cross_attn,
                position_bias,
                None,
                decoder_input
            )
            cudart.cudaMemcpyAsync(hidden_buffer.ptr, decoder_input.ptr, decoder_input.nbytes, cudart.cudaMemcpyDeviceToDevice, ctx.current_stream)
        
        self.ln_dec.forward(ctx, decoder_input, decoder_input)
        self.output_embedding.forward(ctx, decoder_input, decoder_output)
        ctx.free(tensor_mask_self_attn)
        ctx.free(tensor_mask_cross_attn)
        ctx.free(position_bias)

    def encode_backward(self,
            ctx : Context,
            x : Tensor,                 # (batch. dim_model, seq_len)
            x_mask : np.ndarray,        # (batch, seq_len)
            hidden_list : List[Tensor],
            grad_output : Tensor,        # (batch, dim_model, seq_len)
            grad : Tensor
        ):
        batch, dim_model, seq_len = x.shape
        assert len(hidden_list) == self.num_enc
        layer_inputs = [x] + hidden_list[:-1]
        layer_output = hidden_list[-1]

        self_attn_mask = x_mask[:, :, np.newaxis] & x_mask[:, np.newaxis, :]
        tensor_mask = Tensor.from_numpy(ctx, self_attn_mask)

        position_bias = ctx.allocate((self.config.NUM_HEADS, seq_len, seq_len), dtype=np.float16)
        self.position_bias_enc.forward(ctx, seq_len, seq_len, position_bias)

        grad.zero_(ctx)
        self.ln_enc.backward(ctx, layer_output, grad_output, grad)

        for layer, layer_input in zip(self.scheduler.loop_layers(
            ctx,
            self.enc_layers,
            list(reversed(range(self.num_enc)))), layer_inputs[::-1]
        ):
            layer.backward(
                ctx,
                layer_input,
                position_bias,
                tensor_mask,
                grad
            )
        ctx.free(position_bias)
        ctx.free(tensor_mask)
    
    def decode_backward(self,
            ctx : Context,
            decoder_input : Tensor,         # (batch, dim_model, seq_q)
            encoder_output : Tensor,        # (batch, dim_model, seq_k)
            decoder_mask : np.ndarray,      # (batch, dim_model, seq_q)
            encoder_mask : np.ndarray,      # (batch, dim_model, seq_k)
            hidden_list : List[Tensor],
            grad_output : Tensor,           # (batch, seq_q, vocab_size)
            grad_encoder : Tensor,          # (batch, dim_model, seq_k)
            grad : Tensor                   # (batch, dim_model, seq_q)
        ):
        assert decoder_input.shape[:2] == encoder_output.shape[:2]
        batch, dim_model, seq_q = decoder_input.shape
        batch, dim_model, seq_k = encoder_output.shape
        assert decoder_mask.shape == (batch, seq_q)
        assert encoder_mask.shape == (batch, seq_k)
        assert len(hidden_list) == self.num_dec

        self_attn_mask = decoder_mask[:, :, np.newaxis] & decoder_mask[:, np.newaxis, :] & (np.arange(seq_q)[:, np.newaxis] <= np.arange(seq_q)[np.newaxis, :])
        cross_attn_mask = encoder_mask[:, :, np.newaxis] & decoder_mask[:, np.newaxis, :]
        tensor_mask_self_attn = Tensor.from_numpy(ctx, self_attn_mask)
        tensor_mask_cross_attn = Tensor.from_numpy(ctx, cross_attn_mask)

        position_bias = ctx.allocate((self.config.NUM_HEADS, seq_q, seq_q), dtype=np.float16)
        self.position_bias_dec.forward(ctx, seq_q, seq_q, position_bias)

        grad_encoder.zero_(ctx)
        grad.zero_(ctx)

        layer_inputs = [decoder_input] + hidden_list[:-1]
        layer_output = hidden_list[-1]
        tmp_grad = ctx.allocate((batch, dim_model, seq_q), dtype=np.float16)
        tmp_grad.zero_(ctx)
        self.output_embedding.backward(ctx, grad_output, tmp_grad)
        self.ln_dec.backward(ctx, layer_output, tmp_grad, grad)
        ctx.free(tmp_grad)

        for layer, layer_input in zip(self.scheduler.loop_layers(
            ctx,
            self.dec_layers,
            list(reversed(range(self.num_dec)))), layer_inputs[::-1]
        ):
            layer.backward(
                ctx,
                layer_input,
                encoder_output,
                tensor_mask_self_attn,
                tensor_mask_cross_attn,
                position_bias,
                None,
                grad, grad_encoder
            )
        ctx.free(position_bias)
        ctx.free(tensor_mask_self_attn)
        ctx.free(tensor_mask_cross_attn)

        



            
                    
        





        
