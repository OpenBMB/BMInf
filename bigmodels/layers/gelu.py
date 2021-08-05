from .base import Layer

class GeLU(Layer):
    TYPE_FLOAT = 1
    TYPE_I32 = 2

    def __init__(self, ltype):
        if ltype not in [self.TYPE_FLOAT, self.TYPE_I32]:
            raise TypeError("Unknown type for %s (%s)" % (self.__class__.__name__, ltype))
        self.ltype = ltype