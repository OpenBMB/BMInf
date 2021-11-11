
from typing import List, Union
from ..core import Layer, Allocator, Memory, Context
from cpm_kernels.library import cudart

class LayerScheduler:
    def __init__(self, allocator : Allocator, pool_size : int, layer_size : int, stream) -> None:
        base_ptr = allocator.allocate(pool_size * layer_size)
        
        self.pool = [ Memory(base_ptr.ptr + i * layer_size, layer_size, base_ptr.device) for i in range(pool_size)]
        
        self.pool_size = pool_size
        self.pool_items : List[Union[None, Layer]] = [None for _ in range(pool_size)]

        self.stream = stream
        self.calc_event = cudart.cudaEventCreate()
    
    def load(self, layer : Layer):
        if layer.on_device:
            return

        for i in range(self.pool_size):
            if self.pool_items[i] is None:
                self.pool_items[i] = layer
                layer.locked = True
                layer.on_device = True
                layer._to_device(self.pool[i], self.stream)
                cudart.cudaEventRecord(layer.loader_event, self.stream)
                return True
        
        for i in range(self.pool_size):
            if isinstance(self.pool_items[i], Layer) and \
                (not self.pool_items[i].is_fixed) and (not self.pool_items[i].locked):
                self.pool_items[i].on_device = False
                self.pool_items[i] = layer
                layer.locked = True
                layer.on_device = True
                layer._to_device(self.pool[i], self.stream)
                cudart.cudaEventRecord(layer.loader_event, self.stream)
                return True
        return False
        
    def release(self, ctx : Context, layer : Layer):
        cudart.cudaEventRecord(self.calc_event, ctx.current_stream) # release after calc this layer
        cudart.cudaStreamWaitEvent(self.stream, self.calc_event)    # wait until finished this layer
        layer.locked = False