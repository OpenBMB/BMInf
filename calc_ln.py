import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        # convert into float16 if necessary
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float16)
        return self.weight * hidden_states

def main():
    model = torch.load("../merge.pt")

    ln_layer = LayerNorm(4096)
    ln_layer.load_state_dict({'weight': model['encoder.blocks.0.self_attn.layer_norm.weight']})

    input_emb = model['word_embeds.weight'][:512]
    out = ln_layer(input_emb)
    print(out[1, 300])

if __name__ == "__main__":
    main()