from typing import List
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
        if not ids.flags["C_CONTIGUOUS"]:
            ids = ids.copy()
        ids = ids.astype(np.int32)

        tensor_ids = ctx.allocate(ids.shape, dtype=np.int32)
        cudart.cudaMemcpy(tensor_ids.ptr, ids.ctypes.data, ids.nbytes, cudart.cudaMemcpyHostToDevice)

        self.input_embedding.forward(ctx, tensor_ids, x_out)
        ctx.free(tensor_ids)

    def encode(self, 
            ctx : Context, 
            x : Tensor, 
            x_mask : np.ndarray
        ):
        batch, dim_model, seq_len = x.shape
        assert x_mask.shape == (batch, seq_len)
        self_attn_mask = x_mask[:, :, np.newaxis] & x_mask[:, np.newaxis, :]
        assert self_attn_mask.flags["C_CONTIGUOUS"]

        tensor_mask = ctx.allocate(self_attn_mask.shape, dtype=np.int8)
        cudart.cudaMemcpy(tensor_mask.ptr, self_attn_mask.ctypes.data, self_attn_mask.nbytes, cudart.cudaMemcpyHostToDevice)

        position_bias = ctx.allocate((self.config.NUM_HEADS, seq_len, seq_len), dtype=np.float16)
        self.position_bias_enc.forward(ctx, seq_len, seq_len, position_bias)

        for i in range(self.num_enc):
            for j in range(i, self.num_enc):
                if not self.enc_layers[j].on_device:
                    # try to load this layer
                    if not self.scheduler.load(self.enc_layers[j]):
                        break
            assert self.enc_layers[i].on_device
            # wait for loader stream
            cudart.cudaStreamWaitEvent(ctx.current_stream, self.enc_layers[i].loader_event)
            self.enc_layers[i].forward(
                ctx,
                x,
                position_bias,
                tensor_mask,
                x
            )
            self.scheduler.release(ctx, self.enc_layers[i])
        
        self.ln_enc.forward(ctx, x, x)
        ctx.free(tensor_mask)
        ctx.free(position_bias)
    
    def decode(self,
            ctx : Context,
            decoder_input : Tensor,
            encoder_output : Tensor,
            decoder_mask : np.ndarray,
            encoder_mask : np.ndarray,
            decoder_output : Tensor
        ):
        assert decoder_input.shape[:2] == encoder_output.shape[:2]
        batch, dim_model, seq_q = decoder_input.shape
        batch, dim_model, seq_k = encoder_output.shape
        assert decoder_mask.shape == (batch, seq_q)
        assert encoder_mask.shape == (batch, seq_k)

        self_attn_mask = decoder_mask[:, :, np.newaxis] & decoder_mask[:, np.newaxis, :] & (np.arange(seq_q)[:, np.newaxis] <= np.arange(seq_q)[np.newaxis, :])
        cross_attn_mask = encoder_mask[:, :, np.newaxis] & decoder_mask[:, np.newaxis, :]
        tensor_mask_self_attn = ctx.allocate(self_attn_mask.shape, dtype=np.int8)
        cudart.cudaMemcpy(tensor_mask_self_attn.ptr, self_attn_mask.ctypes.data, self_attn_mask.nbytes, cudart.cudaMemcpyHostToDevice)

        tensor_mask_cross_attn = ctx.allocate(cross_attn_mask.shape, dtype=np.int8)
        cudart.cudaMemcpy(tensor_mask_cross_attn.ptr, cross_attn_mask.ctypes.data, cross_attn_mask.nbytes, cudart.cudaMemcpyHostToDevice)

        position_bias = ctx.allocate((self.config.NUM_HEADS, seq_q, seq_q), dtype=np.float16)
        self.position_bias_dec.forward(ctx, seq_q, seq_q, position_bias)

        for i in range(self.num_dec):
            for j in range(i, self.num_dec):
                if not self.dec_layers[j].on_device:
                    # try to load this layer
                    if not self.scheduler.load(self.dec_layers[j]):
                        break
            assert self.dec_layers[i].on_device
            # wait for loader stream
            cudart.cudaStreamWaitEvent(ctx.current_stream, self.dec_layers[i].loader_event)

            self.dec_layers[i].forward(
                ctx,
                decoder_input,
                encoder_output,
                tensor_mask_self_attn,
                tensor_mask_cross_attn,
                position_bias,
                None,
                decoder_input
            )
            self.scheduler.release(ctx, self.dec_layers[i])
        
        self.ln_dec.forward(ctx, decoder_input, decoder_input)
        self.output_embedding.forward(ctx, decoder_input, decoder_output)
        ctx.free(tensor_mask_self_attn)
        ctx.free(tensor_mask_cross_attn)
        ctx.free(position_bias)
        cudart.cudaStreamSynchronize(ctx.current_stream)




    





            
                    
        





        
