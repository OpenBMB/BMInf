import torch
from typing import List, Optional, Tuple, Union
from ...arch.gpt import GPTConfiguration, GPT2Model
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

class OpGPTEncode(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
            inf_ctx : Context,
            input_hidden : torch.Tensor, input_mask : np.ndarray,
            model : GPT2Model
        ):
        orig_hidden = input_hidden
        input_hidden = clone_tensor(input_hidden)
        input_mask = align_mask(input_hidden.size(-1), input_mask)

        out = torch.empty(input_hidden.size(0), input_hidden.size(2), model.config.VOCAB_SIZE, dtype=torch.half, device=input_hidden.device)
        enc_buffer = [
            torch.empty(input_hidden.size(), dtype=torch.half, device=input_hidden.device)
            for _ in range(model.num_layers)
        ]
        
        wait_stream(torch.cuda.current_stream().cuda_stream, inf_ctx.current_stream)
        model.encode_requires_grad(
            inf_ctx,
            torch_to_tensor(input_hidden),
            input_mask,
            torch_to_tensor(out),
            [ torch_to_tensor(x) for x in enc_buffer ]
        )
        wait_stream(inf_ctx.current_stream, torch.cuda.current_stream().cuda_stream)

        ctx.save_for_backward(orig_hidden, *enc_buffer)
        ctx.inf_ctx = inf_ctx
        ctx.model = model
        ctx.input_mask = input_mask
        return out[:, :orig_hidden.size(-1), :]
    
    @staticmethod
    def backward(ctx, grad_hidden : torch.Tensor):
        if grad_hidden.size(1) % 4 != 0:
            grad_hidden = F.pad(grad_hidden, (0, 0, 0, 4 - (grad_hidden.size(1) % 4)), "constant", 0).contiguous()
        else:
            grad_hidden = clone_tensor(grad_hidden)

        orig_hidden = clone_tensor(ctx.saved_tensors[0])
        enc_buffer = list(ctx.saved_tensors[1:])

        model : GPT2Model = ctx.model
        inf_ctx : Context = ctx.inf_ctx
        grad = torch.empty(orig_hidden.size(), dtype=torch.half, device=orig_hidden.device)

        wait_stream(torch.cuda.current_stream().cuda_stream, inf_ctx.current_stream)
        model.encode_backward(
            inf_ctx,
            torch_to_tensor(orig_hidden),
            ctx.input_mask,
            [ torch_to_tensor(x) for x in enc_buffer ],
            torch_to_tensor(grad_hidden),
            torch_to_tensor(grad)
        )
        wait_stream(inf_ctx.current_stream, torch.cuda.current_stream().cuda_stream)

        return None, grad[:, :, :ctx.saved_tensors[0].size(-1)], None, None

class TorchGPT2:
    def __init__(self,
            config : GPTConfiguration
        ) -> None:
        config.DEVICE = torch.cuda.current_device()

        self.device = Device(config.DEVICE)
        self._torch_device = torch.device("cuda:%d" % config.DEVICE)

        self._ctx = Context([config.DEVICE], [
            TorchAllocator()
        ])
        self._model = GPT2Model(config)
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


    def embedding(self, input_idx : np.ndarray, position : np.ndarray) -> torch.Tensor:
        """
        Args:
            input_idx: (batch_size, seq_len)            on cpu
        Returns:
            hidden: (batch_size, embed_dim, seq_len)    on gpu
        """

        assert input_idx.ndim == 2, "input_idx must be 2-dimensional"
        batch, seq_len = input_idx.shape

        input_idx = input_idx.astype(np.int32)
        position = position.astype(np.int32)
        
        embedding_out = torch.empty(batch, self._config.DIM_MODEL, seq_len, dtype=torch.float16, device=self._torch_device)
        self._model.embedding(
            self._ctx, 
            input_idx,
            position,
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

        return OpGPTEncode.apply(self._ctx, input_hidden, input_mask.astype(np.int8), self._model)
    
    