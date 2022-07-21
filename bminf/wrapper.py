from typing import Tuple
import torch
from .scheduler import TransformerBlockList
from .quantization import QuantizedLinear
import warnings
from cpm_kernels.library import cudart

def _wrapper(model : torch.nn.Module, quantization, is_in_blocklist : bool, memory_limit : int) -> Tuple[torch.nn.Module, bool]:
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
        model._modules[kw], fd = _wrapper(model._modules[kw], quantization, is_module_list or is_in_blocklist, memory_limit)
        found_linear = found_linear or fd

    if is_module_list and len(model) > 0:
        model = TransformerBlockList([
            layer for layer in model
        ], [(torch.cuda.current_device(), memory_limit)])
    return model, found_linear

def wrapper(model : torch.nn.Module, quantization : bool = True, memory_limit = None) -> torch.nn.Module:
    if memory_limit is None:
        memory_limit = cudart.cudaMemGetInfo()[0] * 0.9
    model, found_linear = _wrapper(model, quantization, False, memory_limit)
    if quantization and not found_linear:
        warnings.warn("`quantization` is set to true but `torch.nn.Linear` is not found in your model.")
    return model
