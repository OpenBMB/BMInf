import torch
from typing import List, Union, overload
from ...arch.gpt import GPTConfiguration, GPT2Model, GPT2Tokenizer
from ..allocator import TorchAllocator
from ...core import Context, Device
from ..utils import torch_to_tensor, wait_stream, clone_tensor, align_mask
import numpy as np
import torch.nn.functional as F
from cpm_kernels.library import cudart

class OpGPTProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
            inf_ctx : Context,
            hidden : torch.Tensor,
            model : GPT2Model
        ):
        seq_len = hidden.size(2)    # original seq_len
        hidden = clone_tensor(hidden)   # seq_len aligned to 4
        out = torch.empty(hidden.size(0), hidden.size(2), model.config.VOCAB_SIZE, dtype=hidden.dtype, device=hidden.device)
        model.projection(
            inf_ctx,
            torch_to_tensor(hidden),
            torch_to_tensor(out)
        )
        ctx.inf_ctx = inf_ctx
        ctx.model = model
        return out[:, :seq_len, :]
    
    @staticmethod
    def backward(ctx,
            grad_output : torch.Tensor,
        ):
        seq_len = grad_output.size(1)
        if grad_output.size(1) % 4 != 0:
            grad_output = F.pad(grad_output, (0, 0, 0, 4 - (grad_output.size(1) % 4)), "constant", 0).contiguous()

        model : GPT2Model = ctx.model

        grad = torch.empty(grad_output.size(0), model.config.DIM_MODEL, grad_output.size(1), dtype=grad_output.dtype, device=grad_output.device)
        model.projection_backward(
            ctx.inf_ctx,
            torch_to_tensor(grad_output),
            torch_to_tensor(grad)
        )
        return None, grad[:, :, :seq_len], None

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

        enc_buffer = [
            torch.empty(input_hidden.size(), dtype=torch.half, device=input_hidden.device)
            for _ in range(model.num_layers)
        ]
        
        wait_stream(torch.cuda.current_stream().cuda_stream, inf_ctx.current_stream)
        model.encode_requires_grad(
            inf_ctx,
            torch_to_tensor(input_hidden),
            input_mask,
            [ torch_to_tensor(x) for x in enc_buffer ]
        )
        wait_stream(inf_ctx.current_stream, torch.cuda.current_stream().cuda_stream)

        ctx.save_for_backward(orig_hidden, *enc_buffer)
        ctx.inf_ctx = inf_ctx
        ctx.model = model
        ctx.input_mask = input_mask
        return input_hidden[:, :, :orig_hidden.size(-1)]
    
    @staticmethod
    def backward(ctx, grad_hidden : torch.Tensor):
        grad_hidden = clone_tensor(grad_hidden)
        orig_hidden = clone_tensor(ctx.saved_tensors[0])
        enc_buffer = list(ctx.saved_tensors[1:])

        model : GPT2Model = ctx.model
        inf_ctx : Context = ctx.inf_ctx

        wait_stream(torch.cuda.current_stream().cuda_stream, inf_ctx.current_stream)
        model.encode_backward(
            inf_ctx,
            torch_to_tensor(orig_hidden),
            ctx.input_mask,
            [ torch_to_tensor(x) for x in enc_buffer ],
            torch_to_tensor(grad_hidden)
        )
        wait_stream(inf_ctx.current_stream, torch.cuda.current_stream().cuda_stream)

        return None, grad_hidden[:, :, :ctx.saved_tensors[0].size(-1)], None, None


class TorchGPTTokenizer:
    def __init__(self, tokenizer : GPT2Tokenizer) -> None:
        self.tokenizer = tokenizer
    
    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eod_id
    
    @overload
    def convert_tokens_to_ids(self, tokens : List[str]) -> List[int]: ...

    @overload
    def convert_tokens_to_ids(self, tokens : str) -> int: ...

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, list):
            return self.tokenizer.convert_tokens_to_ids(tokens)
        else:
            return self.tokenizer.convert_tokens_to_ids([tokens])[0]

    @overload
    def convert_ids_to_tokens(self, ids : List[int]) -> List[str]: ...

    @overload
    def convert_ids_to_tokens(self, ids : int) -> str: ...

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, list):
            return self.tokenizer.convert_ids_to_tokens(ids)
        else:
            return self.tokenizer.convert_ids_to_tokens([ids])[0]
    
    def convert_tokens_to_string(self, tokens : List[str]) -> str:
        text = ''.join(tokens)
        text = text.translate(self.tokenizer.translator_dec)
        return text
    
    def tokenize(self, text : str) -> List[str]:
        return self.tokenizer.tokenize(text)


class TorchGPT2(torch.nn.Module):
    def __init__(self,
            config : GPTConfiguration
        ) -> None:
        super().__init__()
        
        config.DEVICE = torch.cuda.current_device()

        self.device = Device(config.DEVICE)
        self._torch_device = torch.device("cuda:%d" % config.DEVICE)

        self._ctx = Context([config.DEVICE], [
            TorchAllocator()
        ])
        self._model = GPT2Model(config)
        self._config = config
    
    @property
    def tokenizer(self):
        return TorchGPTTokenizer(self._model.tokenizer)
    
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
        
        with torch.cuda.device(self._torch_device):
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

        with torch.cuda.device(self._torch_device):
            return OpGPTEncode.apply(self._ctx, input_hidden, input_mask.astype(np.int8), self._model)
    
    def project(self, hidden_state : torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: (batch_size, embed_dim, seq_len)    on gpu
        Returns:
            logits: (batch_size, seq_len, vocab_size)         on gpu
        """
        return OpGPTProjection.apply(self._ctx, hidden_state, self._model)
    
    def get_input_embeddings(self) -> torch.Tensor:
        out = torch.empty(self._model.config.VOCAB_SIZE, self._model.config.DIM_MODEL, dtype=torch.half, device=self._torch_device)
        with torch.cuda.device(self._torch_device):
            cudart.cudaMemcpyAsync(
                out.data_ptr(), self._model.token_embedding.weight.value.ptr, self._model.token_embedding.weight.nbytes, cudart.cudaMemcpyDeviceToDevice, torch.cuda.current_stream().cuda_stream
            )
        return out