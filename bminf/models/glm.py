import torch
from ..scheduler import TransformerBlockList
from ..layers import GLMBlock

class GLM130B(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.input_device = torch.device(0)

        self.token_embedding = torch.nn.Embedding(150528, 12288, dtype=torch.half, device=torch.device(self.input_device))

        self.layers = TransformerBlockList([
            GLMBlock(
                12288,
                96,
                128,
                32768,
                True,
                True,
                (2 * 70) ** 0.5,
                1e-5,
                torch.half
            )
            for _ in range(1)
        ], [0])
    
    def forward(self, ids : torch.LongTensor, position : torch.LongTensor, mask : torch.BoolTensor):
        with torch.cuda.device(self.input_device):
            hidden_state = self.token_embedding(ids.cuda())
        for layer in self.layers:
            position = position.cuda()
            hidden_state = hidden_state.cuda()
            mask = mask.cuda()
            hidden_state = layer(hidden_state, mask, position)
        return hidden_state
    
