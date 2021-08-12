from .base import Layer
class LayerList(Layer):
    def __init__(self, layers):
        self._layers = layers
        for i, it in enumerate(self._layers):
            self._add_sublayer(f"{i}", it)
    
    def __getitem__(self, key):
        return self._layers[key]
    
    def __iter__(self):
        return iter(self._layers)
    
    def __len__(self):
        return len(self._layers)