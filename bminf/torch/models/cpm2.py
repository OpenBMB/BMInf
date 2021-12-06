import numpy as np
import torch
from typing import Optional
from ..arch.t5 import TorchT5
from ..utils import ResultClass
from ...models.cpm2 import CPM2Configuration, SUPPORTED_VERSION, LATEST_VERSION

class CPM2(TorchT5):
    def __init__(self,
            memory_limit : Optional[int] = None,
            version : Optional[str] = None
        ) -> None:
        if version is None:
            version = LATEST_VERSION
        if version not in SUPPORTED_VERSION:
            raise RuntimeError("CPM2 version %s is not supported (requires %s)" % (version, SUPPORTED_VERSION))
        
        config = CPM2Configuration()
        config.MODEL_NAME = version

        device_idx = torch.cuda.current_device()

        config.DEVICE = device_idx
        config.MEMORY_LIMIT = memory_limit

        super().__init__(config)

    def forward(self,
            input_ids : Optional[torch.LongTensor] = None,                  # (batch_size, enc_len)
            inputs_embeds : Optional[torch.FloatTensor] = None,             # (batch_size, enc_len, embed_dim)
            attention_mask : Optional[torch.FloatTensor] = None,            # (batch_size, enc_len)
            decoder_input_ids : Optional[torch.LongTensor] = None,          # (batch_size, dec_len)
            decoder_inputs_embeds : Optional[torch.FloatTensor] = None,     # (batch_size, dec_len, embed_dim)
            decoder_attention_mask : Optional[torch.FloatTensor] = None,    # (batch_size, dec_len)
            encoder_outputs : Optional[torch.FloatTensor] = None,
            output_attentions : bool = False,
            output_hidden_states : bool = False
        ):
        if output_attentions:
            raise ValueError("output attentions is not supported")
        if encoder_outputs is None:
            if inputs_embeds is None:
                inputs_embeds = self.embedding(input_ids.cpu().numpy().astype(np.int32))
            else:
                inputs_embeds = inputs_embeds.permute(0, 2, 1) # (batch_size, embed_dim, enc_len)
                inputs_embeds = inputs_embeds.half()
            if attention_mask is None:
                attention_mask = torch.ones((inputs_embeds.shape[0], inputs_embeds.shape[2]))
            encoder_outputs = self.encode(
                inputs_embeds,
                attention_mask.cpu().numpy() > 0.5,
            )
        if decoder_inputs_embeds is None:
            decoder_inputs_embeds = self.embedding(decoder_input_ids.cpu().numpy().astype(np.int32))
            if decoder_attention_mask is None:
                decoder_attention_mask = decoder_input_ids != self.tokenizer.pad_token_id
        else:
            decoder_inputs_embeds = decoder_inputs_embeds.permute(0, 2, 1) # (batch_size, embed_dim, dec_len)
            decoder_inputs_embeds = decoder_inputs_embeds.half()
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