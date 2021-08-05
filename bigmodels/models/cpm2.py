import cupy
from bigmodels.parameter import Parameter
from .base import Model
from ..layers.transformer_block import TransformerBlock
from ..layers.encoder_kv import EncoderKeyValueProjection
from ..layers.position_bias import PositionBias
from ..layers.embedding import Embedding
from ..layers.layer_norm import LayerNorm
from ..layers.lm_head import LMHead
from ..layers.layer_list import LayerList
from ..configuration import CPM2Configuration
from ..allocator import AllocatorConfig, ReusedAllocator, SizeLimitedAllocator
from ..context import Context
import logging

logger = logging.getLogger(__name__)

class CPM2(Model):
    def __init__(self, config : CPM2Configuration):
        # Build Model
        logger.info("Building model")
        if config.BIT_SIZE == 8:
            ltype = TransformerBlock.TYPE_I8
        elif config.BIT_SIZE == 16:
            ltype = TransformerBlock.TYPE_F16
        elif config.BIT_SIZE == 32:
            ltype = TransformerBlock.TYPE_F32
        else:
            raise ValueError("BIT_SIZE must be 8, 16 or 32")
        
        self.memory_overlap = config.MEMORY_OVERLAP
        self.overlap_layers = config.OVERLAP_LAYERS
        self.encoder_only = config.ENCODER_ONLY

        self.input_embedding = Embedding(config.VOCAB_SIZE, config.DIM_MODEL, ltype=ltype)

        self.encoder_position_bias = PositionBias(config.NUM_POSITION_BUCKETS, config.NUM_HEADS, PositionBias.TYPE_F32)
        self.encoder = LayerList([
            TransformerBlock(False, config.DIM_MODEL, config.DIM_FF, config.DIM_KV, config.NUM_HEADS, ltype)
                for _ in range(config.NUM_ENCODER_LAYERS)
        ])
        self.encoder_final_layer_nrom = LayerNorm(config.DIM_MODEL, ltype=LayerNorm.TYPE_F32)

        if not self.encoder_only:
            self.decoder_position_bias = PositionBias(config.NUM_POSITION_BUCKETS, config.NUM_HEADS, PositionBias.TYPE_F32)
            self.encoder_kv = EncoderKeyValueProjection(config.NUM_DECODER_LAYERS, config.DIM_MODEL, config.DIM_KV, config.NUM_HEADS, ltype)
            self.lm_head = LMHead(config.VOCAB_SIZE, config.DIM_MODEL, ltype=ltype)
            
            self.decoder = LayerList([
                TransformerBlock(True, config.DIM_MODEL, config.DIM_FF, config.DIM_KV, config.NUM_HEADS, ltype)
                    for _ in range(config.NUM_DECODER_LAYERS)
            ])
            self.decoder_final_layer_nrom = LayerNorm(config.DIM_MODEL, ltype=LayerNorm.TYPE_F32)

        # init parameter
        if not isinstance(config.DEVICES, list):
            devices_ids = [ config.DEVICES ]
        else:
            devices_ids = config.DEVICES

        assert len(devices_ids) == 1 # multi device not supported

        devices = [ cupy.cuda.Device(idx) for idx in devices_ids ]

        logger.info("Start assign devices")
        self.assign_device([ cupy.cuda.Device(device_id) for device_id in devices ])
        
        logger.info("Start loading parameters from disk to cpu")
        self.load( open(config.MODEL_PATH, "rb") )

        logger.info("Start loading parameters from cpu to gpu")
        init_ctx = Context(devices)
        if self.memory_overlap:
            mx_size = 0
            for i in range(config.NUM_ENCODER_LAYERS):
                mx_size = max(self.encoder[i].nbytes, mx_size)
            for i in range(config.NUM_DECODER_LAYERS):
                mx_size = max(self.decoder[i].nbytes, mx_size)

            temp_size = self._get_preapre_buffer_size()
            overlap_size = mx_size * self.overlap_layers * 4
            other_size = self.nbytes - self.encoder.nbytes - self.decoder.nbytes

            if overlap_size + other_size + temp_size * 2 + config.DYNAMIC_MEMORY < config.MEMORY_LIMIT:
                raise ValueError("memory limit not enough, at least %d bytes, bug got %d bytes" % (overlap_size + other_size + temp_size * 2 + config.DYNAMIC_MEMORY, config.MEMORY_LIMIT))
            logger.info("Using overlap loader: overlap_size %d, temp_size %d, other_size: %d, dynamic_memory %d, memory_limit %d", overlap_size, temp_size, other_size, config.DYNAMIC_MEMORY, config.MEMORY_LIMIT)
            self.parameter_allocator = ReusedAllocator([ 
                AllocatorConfig(devices[0], other_size + (overlap_size // 2), temp_size)
            ]) # FIXME: multi device not supported now.
            self.overlap_allocator = ReusedAllocator([
                AllocatorConfig(devices[0], overlap_size // 2, temp_size)
            ])
            self.variable_allocator = SizeLimitedAllocator([
                AllocatorConfig(devices[0], config.MEMORY_LIMIT - other_size - overlap_size - temp_size * 2, 0)
            ])

            for name, layer in self._sub_layers.items():
                if name in ["encoder", "decoder"]:
                    # move first overlap_size layers to device
                    for i in range(self.overlap_layers):
                        layer[i].to_device( self.parameter_allocator, init_ctx )
                else:
                    layer.to_device( self.parameter_allocator, init_ctx )
        else:
            if self.nbytes + config.DYNAMIC_MEMORY < config.MEMORY_LIMIT:
                raise ValueError("memory limit not enough, at least %d bytes, bug got %d bytes" % (self.nbytes + config.DYNAMIC_MEMORY, config.MEMORY_LIMIT))
            
            temp_size = self._get_preapre_buffer_size()
            logger.info("Using static loader: total: %d, temp_size %d, dynamic_memory %d, memory_limit %d", self.nbytes, temp_size, config.DYNAMIC_MEMORY, config.MEMORY_LIMIT)
            self.parameter_allocator = ReusedAllocator([
                AllocatorConfig(devices[0], self.nbytes, temp_size)
            ])
            self.variable_allocator = SizeLimitedAllocator([
                AllocatorConfig(devices[0], config.MEMORY_LIMIT - self.nbytes, 0)
            ])
            self.to_device(self.parameter_allocator, init_ctx)
        
        default_stream = cupy.cuda.get_current_stream()
        init_ctx.sync_load(default_stream)
        default_stream.synchronize()

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

    def forward(self):
        pass