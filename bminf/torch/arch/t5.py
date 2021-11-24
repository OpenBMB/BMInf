import torch
from typing import List, Optional, Tuple, Union
from ...arch.t5 import T5Configuration, T5Model
from ..allocator import TorchAllocator
from ...core import Context, Device
from ..utils import torch_to_tensor, wait_stream
import numpy as np
import torch.nn.functional as F

def clone_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.size(-1) % 4 != 0:
        return F.pad(x, (0, 4 - (x.size(-1) % 4)), "constant", 0).contiguous()
    else:
        return x.clone(memory_format=torch.contiguous_format)

def align_mask(last_dim, mask : np.ndarray):
    assert mask.ndim == 2
    assert last_dim >= mask.shape[-1]
    
    if mask.shape[1] != last_dim:
        nw_mask = np.zeros(mask.shape[:-1] + (last_dim,), dtype=np.int8)
        nw_mask[:, :mask.shape[-1]] = mask
        return nw_mask
    else:
        return mask

class OpT5Encode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inf_ctx : Context, input_hidden : torch.Tensor, input_mask : np.ndarray, model : T5Model):
        orig_input = input_hidden
        input_hidden = clone_tensor(input_hidden)
        input_mask = align_mask(input_hidden.size(-1), input_mask)

        enc_buffer = [
            torch.empty(input_hidden.size(), device=input_hidden.device, dtype=torch.half) 
            for _ in range(model.num_enc)
        ]
        
        wait_stream(torch.cuda.current_stream().cuda_stream, inf_ctx.current_stream)
        model.encode_requires_grad(
            inf_ctx,
            torch_to_tensor(input_hidden),
            input_mask,
            [ torch_to_tensor(x) for x in enc_buffer ]
        )
        wait_stream(inf_ctx.current_stream, torch.cuda.current_stream().cuda_stream)

        ctx.save_for_backward(orig_input, *enc_buffer)
        
        ctx.inf_ctx = inf_ctx
        ctx.input_mask = input_mask
        ctx.model = model
        return input_hidden[:, :, :orig_input.size(-1)]

    @staticmethod
    def backward(ctx, grad_hidden : torch.Tensor):
        grad_hidden = clone_tensor(grad_hidden)

        model : T5Model = ctx.model
        input_hidden = clone_tensor(ctx.saved_tensors[0])
        enc_buffer = list(ctx.saved_tensors[1:])

        grad_out = torch.empty(grad_hidden.size(), device=grad_hidden.device, dtype=torch.half)

        wait_stream(torch.cuda.current_stream().cuda_stream, ctx.inf_ctx.current_stream)
        model.encode_backward(
            ctx.inf_ctx,
            input_hidden,
            ctx.input_mask,
            [ torch_to_tensor(x) for x in enc_buffer ],
            torch_to_tensor(grad_hidden),
            torch_to_tensor(grad_out)
        )
        wait_stream(ctx.inf_ctx.current_stream, torch.cuda.current_stream().cuda_stream)

        return None, grad_out[:, :, :ctx.saved_tensors[0].size(-1)], None, None

class OpT5Decode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
            inf_ctx : Context,
            dec_hidden : torch.Tensor, dec_mask : np.ndarray,
            enc_hidden : torch.Tensor, enc_mask : np.ndarray,
            model : T5Model
        ) -> torch.Tensor:
        orig_dec = dec_hidden
        orig_enc = enc_hidden
        dec_hidden = clone_tensor(dec_hidden)
        enc_hidden = clone_tensor(enc_hidden)
        dec_mask = align_mask(dec_hidden.size(-1), dec_mask)
        enc_mask = align_mask(enc_hidden.size(-1), enc_mask)
        
        out = torch.empty(
            dec_hidden.size(0), dec_hidden.size(2), model.config.VOCAB_SIZE,
            dtype=torch.half,
            device=dec_hidden.device
        )
        dec_buffer = [
            torch.empty(dec_hidden.size(), device=dec_hidden.device, dtype=torch.half)
                for _ in range(model.num_dec)
        ]
        
        wait_stream(torch.cuda.current_stream().cuda_stream, inf_ctx.current_stream)
        model.decode_requires_grad(
            inf_ctx,
            torch_to_tensor(dec_hidden), torch_to_tensor(enc_hidden),
            dec_mask, enc_mask,
            torch_to_tensor(out),
            [ torch_to_tensor(x) for x in dec_buffer ]
        )
        wait_stream(inf_ctx.current_stream, torch.cuda.current_stream().cuda_stream)

        ctx.save_for_backward(orig_dec, orig_enc, *dec_buffer)
        ctx.inf_ctx = inf_ctx
        ctx.dec_mask = dec_mask
        ctx.enc_mask = enc_mask
        ctx.model = model
        
        return out[:, :orig_dec.size(-1), :]
    
    @staticmethod
    def backward(ctx, grad_hidden : torch.Tensor):
        if grad_hidden.size(1) % 4 != 0:
            grad_hidden = F.pad(grad_hidden, (0, 0, 0, 4 - (grad_hidden.size(1) % 4)), "constant", 0).contiguous()
        else:
            grad_hidden = clone_tensor(grad_hidden)

        model : T5Model = ctx.model
        dec_hidden : torch.Tensor = clone_tensor(ctx.saved_tensors[0])
        enc_hidden : torch.Tensor = clone_tensor(ctx.saved_tensors[1])
        dec_buffer : List[torch.Tensor] = list(ctx.saved_tensors[2:])

        grad_dec = torch.empty(
            dec_hidden.size(), device=dec_hidden.device, dtype=torch.half
        )
        grad_enc = torch.empty(
            enc_hidden.size(), device=enc_hidden.device, dtype=torch.half
        )

        wait_stream(torch.cuda.current_stream().cuda_stream, ctx.inf_ctx.current_stream)
        model.decode_backward(
            ctx.inf_ctx,
            torch_to_tensor(dec_hidden), torch_to_tensor(enc_hidden),
            ctx.dec_mask, ctx.enc_mask,
            [ torch_to_tensor(x) for x in dec_buffer ],
            torch_to_tensor(grad_hidden), torch_to_tensor(grad_enc), torch_to_tensor(grad_dec)
        )
        wait_stream(ctx.inf_ctx.current_stream, torch.cuda.current_stream().cuda_stream)

        return None, grad_dec[:, :, :ctx.saved_tensors[0].size(-1)], None, grad_enc[:, :, :ctx.saved_tensors[1].size(-1)], None, None


class TorchT5:
    def __init__(self,
            config : T5Configuration
        ) -> None:
        config.DEVICE = torch.cuda.current_device()

        self.device = Device(config.DEVICE)
        self._torch_device = torch.device("cuda:%d" % config.DEVICE)

        self._ctx = Context([config.DEVICE], [
            TorchAllocator()
        ])
        self._model = T5Model(config)
        self._config = config
    
    def tokenize(self, text : Union[str, List[str]]) -> torch.Tensor:
        if isinstance(text, str):
            return torch.LongTensor(self._model.tokenizer.encode(text))
        elif isinstance(text, list):
            return torch.LongTensor([self._model.tokenizer.encode(t) for t in text])
        else:
            raise TypeError("text must be str or list[str]")
    
    def detokenize(self, ids : torch.Tensor) -> Union[str, List[str]]:
        assert ids.dtype == torch.int64 or ids.dtype == torch.int32
        if ids.ndim == 1:
            return self._model.tokenizer.decode(ids.tolist()) 
        elif ids.ndim == 2:
            return [
                self._model.tokenizer.decode(s)
                for s in ids.tolist()
            ]
        else:
            raise ValueError("ids must be 1D or 2D")


    def embedding(self, input_idx : np.ndarray) -> torch.Tensor:
        """
        Args:
            input_idx: (batch_size, seq_len)            on cpu
        Returns:
            hidden: (batch_size, embed_dim, seq_len)    on gpu
        """

        assert input_idx.ndim == 2, "input_idx must be 2-dimensional"
        batch, seq_len = input_idx.shape

        input_idx = input_idx.astype(np.int32)
        
        embedding_out = torch.empty(batch, self._config.DIM_MODEL, seq_len, dtype=torch.float16, device=self._torch_device)
        self._model.embedding(
            self._ctx, 
            input_idx,
            torch_to_tensor(embedding_out)
        )
        return embedding_out
    
    def encode(self, input_hidden : torch.Tensor, input_mask : np.ndarray) -> torch.Tensor:
        """
        Args:
            input_hidden: (batch_size, embed_dim, seq_len)      on gpu
            input_mask:   (batch_size, seq_len)                 on cpu
        Returns:
            hidden: (batch_size, embed_dim, seq_len)            on gpu
        """
        batch, dim_model, seq_len = input_hidden.size()
        assert dim_model == self._config.DIM_MODEL, "input_hidden must have dim_model=%d" % self._config.DIM_MODEL
        assert input_hidden.is_cuda, "input_hidden must be on gpu"
        assert input_hidden.dtype == torch.float16, "input_hidden must be float16"
        assert input_mask.dtype == np.bool8, "input_mask must be bool"
        assert input_mask.shape == (batch, seq_len), "input_mask must have same size as input_hidden"

        return OpT5Encode.apply(self._ctx, input_hidden, input_mask.astype(np.int8), self._model)
    
    def decode(self, dec_hidden : torch.Tensor, dec_mask : np.ndarray, enc_hidden : torch.Tensor, enc_mask : np.ndarray) -> torch.Tensor:
        """
        Args:
            dec_hidden: (batch_size, embed_dim, seq_q)        on gpu
            dec_mask:   (batch_size, seq_q)                   on cpu
            enc_hidden: (batch_size, embed_dim, seq_k)        on gpu
            enc_mask:   (batch_size, seq_k)                   on cpu
        Returns:
            logits:     (batch_size, seq_q, vocab_size)       on gpu
        """
        assert dec_hidden.ndim == 3 and enc_hidden.ndim == 3, "dec_hidden and enc_hidden must be 3-dimensional"
        batch, dim_model, seq_q = dec_hidden.size()
        seq_k = enc_hidden.size(2)

        assert enc_hidden.size() == (batch, dim_model, seq_k)
        assert dec_hidden.is_cuda, "dec_hidden must be on gpu"
        assert dec_hidden.dtype == torch.float16, "dec_hidden must be float16"
        assert dec_mask.dtype == np.bool8, "dec_mask must be bool"
        assert dec_mask.shape == (batch, seq_q)
        assert enc_mask.dtype == np.bool8, "enc_mask must be bool"
        assert enc_mask.shape == (batch, seq_k)
        assert enc_hidden.is_cuda, "enc_hidden must be on gpu"
        assert enc_hidden.dtype == torch.float16, "enc_hidden must be float16"
        assert enc_hidden.device == dec_hidden.device, "enc_hidden and dec_hidden must be on the same device"

        return OpT5Decode.apply(self._ctx, dec_hidden, dec_mask.astype(np.int8), enc_hidden, enc_mask.astype(np.int8), self._model)

