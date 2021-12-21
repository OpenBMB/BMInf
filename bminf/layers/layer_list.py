from typing import Generic, TypeVar, List
from ..core import Layer

T = TypeVar("T", bound=Layer)
class LayerList(Layer, Generic[T]):
    def __init__(self, layers : List[T], offset = True):
        super().__init__()
        self.__offset = offset    
        self._layers = layers
        for i, it in enumerate(self._layers):
            self._add_sublayer(f"{i}", it)
        
    def _add_sublayer(self, name, layer : 'Layer'):
        if self._parent is not None:
            raise RuntimeError("Adding parameter after intialization")
        if self.__offset:
            layer._set_parent(self, self._parameter_bytes)
        self._sub_layers[name] = layer
        self._parameter_bytes += layer.nbytes
    
    def __getitem__(self, key) -> T:
        return self._layers[key]
    
    def __iter__(self):
        return iter(self._layers)
    
    def __len__(self):
        return len(self._layers)