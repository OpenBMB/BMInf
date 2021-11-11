from .memory import Memory

class Allocator:
    def allocate(self, nbytes : int) -> Memory: ...

    def free(self, mem : Memory) -> None: ...

