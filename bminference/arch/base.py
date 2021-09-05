from ..layers.base import Layer
import cupy

class Model(Layer):
    device : cupy.cuda.Device
    pass
            

class Configuration:
    def __init__(self) -> None:
        self._kws = {}
    
    def __getattribute__(self, name: str):
        if name.startswith("_"):
            return super().__getattribute__(name)
        if name in self._kws:
            return self._kws[name]
        return super().__getattribute__(name)
    
    def __setattr__(self, name: str, value) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        self._kws[name] = value

class InferenceContext:
    pass


class Tokenizer(object):

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError()

    def __len__(self) -> int:
        return self.vocab_size

    @property
    def sod_id(self) -> int:
        raise NotImplementedError()

    @property
    def eod_id(self) -> int:
        raise NotImplementedError()
    
    @property
    def unk_id(self) -> int:
        raise NotImplementedError()

    @property
    def sod_token(self):
        return "<s>"

    @property
    def eod_token(self):
        return '<eod>'
    
    @property
    def unk_token(self):
        return "<unk>"
