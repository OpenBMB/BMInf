import cupy
import logging

logger = logging.getLogger(__name__)
class AllocatorConfig:
    def __init__(self, device, size, temp_size):
        assert isinstance(device, cupy.cuda.Device)
        self.device = device
        self.size = size
        self.temp_size = temp_size

class Allocator:
    def __init__(self):
        pass

    def alloc(self, size):
        logger.info("Allocate %d" % size)
        return self._alloc(size)

    def _alloc(self, size):
        raise NotImplementedError()

    @property
    def temp_ptr(self):
        raise NotImplementedError()