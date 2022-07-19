import torch
import torch.nn.functional as F

class Linear(torch.nn.Module):
    def __init__(self, in_features : int, out_features : int, bias: bool = True,
                 dtype=None) -> None:
        super().__init__()
        torch.nn.Linear
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype

        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), dtype=dtype))
        if bias:
            if self.dtype == torch.int8:
                self.bias = torch.nn.Parameter(torch.empty(out_features, dtype=torch.half))
            else:
                self.bias = torch.nn.Parameter(torch.empty(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.dtype == torch.int8:
            raise RuntimeError("INT8 not supported now")
        return F.linear(input, self.weight, self.bias)
