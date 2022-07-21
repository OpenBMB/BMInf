from typing import Tuple
import torch
from .scheduler import TransformerBlockList
from .quantization import QuantizedLinear
import warnings

def _wrapper(model : torch.nn.Module, quantization, is_in_blocklist : bool) -> Tuple[torch.nn.Module, bool]:
    if not is_in_blocklist:
        # move to cuda
        for kw, param in model._parameters.items():
            if param is None:
                continue
            model._parameters[kw] = param.cuda()
    else:
        # keeps on cpu if in transformer block
        pass
    
    if quantization and isinstance(model, torch.nn.Linear):
        return QuantizedLinear(model), True
    
    found_linear = False
    
    is_module_list = isinstance(model, torch.nn.ModuleList)
    if is_module_list and is_in_blocklist:
        raise ValueError("nested `torch.nn.ModuleList` is not supported.")


    for kw in model._modules.keys():
        model._modules[kw], fd = _wrapper(model._modules[kw], quantization, is_module_list or is_in_blocklist)
        found_linear = found_linear or fd

    if is_module_list and len(model) > 0:
        model = TransformerBlockList([
            layer for layer in model
        ], [torch.cuda.current_device()])
    return model, found_linear

def wrapper(model : torch.nn.Module, quantization : bool = True) -> torch.nn.Module:
    model, found_linear = _wrapper(model, quantization, False)
    if quantization and not found_linear:
        warnings.warn("`quantization` is set to true but `torch.nn.Linear` is not found in your model.")
    return model
