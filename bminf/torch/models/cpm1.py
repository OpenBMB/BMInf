from typing import Optional
import numpy as np
import torch
from ..arch.gpt import TorchGPT2
from ..utils import ResultClass
from ...models.cpm1 import CPM1Configuration, SUPPORTED_VERSION, LATEST_VERSION

class CPM1(TorchGPT2):
    def __init__(self,
            memory_limit : Optional[int] = None,
            version : Optional[str] = None
        ) -> None:
        if version is None:
            version = LATEST_VERSION
        if version not in SUPPORTED_VERSION:
            raise RuntimeError("CPM1 version %s is not supported (requires %s)" % (version, SUPPORTED_VERSION))
        
        config = CPM1Configuration()
        config.MODEL_NAME = version

        device_idx = torch.cuda.current_device()

        config.DEVICE = device_idx
        config.MEMORY_LIMIT = memory_limit

        super().__init__(config)

    def forward(self,
            input_ids : Optional[torch.LongTensor] = None,                  # (batch_size, enc_len)
            inputs_embeds : Optional[torch.FloatTensor] = None,             # (batch_size, enc_len, embed_dim)
            position_ids : Optional[torch.LongTensor] = None,               # (batch_size, enc_len)
            attention_mask : Optional[torch.FloatTensor] = None,            # (batch_size, enc_len)
            output_attentions : bool = False,
            output_hidden_states : bool = False
        ):
        if output_attentions:
            raise ValueError("output attentions is not supported")
        if inputs_embeds is None:
            if position_ids is None:
                position_ids = torch.arange(input_ids.size(1)).unsqueeze(0).expand_as(input_ids)
            position_ids = position_ids.cpu().numpy().astype(np.int32)
            input_ids = input_ids.cpu().numpy().astype(np.int32)
            inputs_embeds = self.embedding(input_ids, position_ids)
        else:
            inputs_embeds = inputs_embeds.permute(0, 2, 1) # (batch_size, embed_dim, enc_len)
            inputs_embeds = inputs_embeds.half()
        if attention_mask is None:
            attention_mask = torch.ones((inputs_embeds.shape[0], inputs_embeds.shape[2]))
        hidden_state = self.encode(
            inputs_embeds,
            attention_mask.cpu().numpy() > 0.5,
        )
        logits = self.project(hidden_state)
        return ResultClass(
            logits=logits,
            hidden_states=[hidden_state.transpose(1, 2)] if output_hidden_states else None
        )