import torch
from .linear import Linear

class FeedForward(torch.nn.Module):
    def __init__(
            self,
            dim_model : int,
            dim_ff : int,
            gated : bool,
            gelu : bool,
            bias : bool,
            dtype : torch.dtype
        ) -> None:
        super().__init__()

        self.gated = gated
        self.gelu = gelu

        self.w_in = Linear(dim_model, dim_ff, bias=bias, dtype=dtype)
        self.w_out = Linear(dim_ff, dim_model, bias=bias, dtype=dtype)

        if self.gelu:
            self.activation = torch.nn.GELU()
        else:
            self.activation = torch.nn.ReLU()
        
        if self.gated:
            self.w_gate = Linear(dim_model, dim_ff, bias=bias, dtype=dtype)
    
    def forward(self, x : torch.Tensor):
        x_in = self.activation(self.w_in(x))
        if self.gated:
            x_in = x_in * self.w_gate(x)
        return self.w_out(x_in)
