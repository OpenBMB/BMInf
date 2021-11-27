import numpy as np
import torch
from typing import Optional
from ..arch.t5 import TorchT5
from ..utils import ResultClass
from ...models.cpm2 import CPM2Configuration, SUPPORTED_VERSION

class CPM2(TorchT5):
    def __init__(self,
            memory_limit : Optional[int] = None,
            version : str = "2.2"
        ) -> None:
        if version not in SUPPORTED_VERSION:
            raise RuntimeError("CPM2 version %s is not supported (requires %s)" % (version, SUPPORTED_VERSION))
        # TODO: set model name here
        config = CPM2Configuration()

        device_idx = torch.cuda.current_device()

        config.DEVICE = device_idx
        config.MEMORY_LIMIT = memory_limit

        super().__init__(config)

    def forward(self,
            input_ids : torch.LongTensor, 
            attention_mask : torch.FloatTensor,
            decoder_input_ids : torch.LongTensor,
            decoder_attention_mask : torch.FloatTensor,
            encoder_outputs : Optional[torch.FloatTensor] = None,
            inputs_embeds : Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds : Optional[torch.FloatTensor] = None,
            output_attentions : bool = False,
            output_hidden_states : bool = False
        ):
        if output_attentions:
            raise ValueError("output attentions is not supported")
        if encoder_outputs is None:
            if inputs_embeds is None:
                inputs_embeds = self.embedding(input_ids.cpu().numpy().astype(np.int32))
            encoder_outputs = self.encode(
                inputs_embeds,
                attention_mask.cpu().numpy() > 0.5,
            )
        if decoder_inputs_embeds is None:
            decoder_inputs_embeds = self.embedding(decoder_input_ids.cpu().numpy().astype(np.int32))
        decoder_outputs = self.decode(
            decoder_inputs_embeds,
            decoder_attention_mask.cpu().numpy() > 0.5,
            encoder_outputs,
            attention_mask.cpu().numpy() > 0.5,
        )
        logits = self.project(decoder_outputs)
        return ResultClass(
            logits=logits,
            decoder_hidden_states=[decoder_outputs.transpose(1, 2)] if output_hidden_states else None,
        )