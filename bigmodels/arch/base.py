from ..layers.base import Layer
class Model(Layer):
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