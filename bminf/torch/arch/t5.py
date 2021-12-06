import torch
from typing import List, Union, overload
from ...arch.t5 import T5Configuration, T5Model, T5Tokenizer
from ..allocator import TorchAllocator
from ...core import Context, Device
from ..utils import torch_to_tensor, wait_stream, clone_tensor, align_mask
import numpy as np
import torch.nn.functional as F
from cpm_kernels.library import cudart

class OpT5Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inf_ctx : Context, hidden : torch.Tensor, model : T5Model):
        seq_len = hidden.size(2)
        hidden = clone_tensor(hidden)
        out = torch.empty(hidden.size(0), hidden.size(2), model.config.VOCAB_SIZE, dtype=hidden.dtype, device=hidden.device)
        wait_stream(torch.cuda.current_stream().cuda_stream, inf_ctx.current_stream)
        model.projection(
            inf_ctx,
            torch_to_tensor(hidden),
            torch_to_tensor(out)
        )
        wait_stream(inf_ctx.current_stream, torch.cuda.current_stream().cuda_stream)
        ctx.inf_ctx = inf_ctx
        ctx.model = model
        return out[:, :seq_len, :]
    
    @staticmethod
    def backward(ctx, grad_output : torch.Tensor):
        model : T5Model = ctx.model
        inf_ctx : Context = ctx.inf_ctx
        seq_len = grad_output.size(1)
        if grad_output.size(1) % 4 != 0:
            grad_output = F.pad(grad_output, (0, 0, 0, 4 - (grad_output.size(1) % 4)), "constant", 0).contiguous()
        else:
            grad_output = clone_tensor(grad_output)

        out = torch.empty(grad_output.size(0), model.config.DIM_MODEL, grad_output.size(1), dtype=grad_output.dtype, device=grad_output.device)
        wait_stream(torch.cuda.current_stream().cuda_stream, inf_ctx.current_stream)
        model.projection_backward(
            inf_ctx,
            torch_to_tensor(grad_output),
            torch_to_tensor(out)
        )
        wait_stream(inf_ctx.current_stream, torch.cuda.current_stream().cuda_stream)
        return None, out[:, :, :seq_len], None


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

        wait_stream(torch.cuda.current_stream().cuda_stream, ctx.inf_ctx.current_stream)
        model.encode_backward(
            ctx.inf_ctx,
            input_hidden,
            ctx.input_mask,
            [ torch_to_tensor(x) for x in enc_buffer ],
            torch_to_tensor(grad_hidden)
        )
        wait_stream(ctx.inf_ctx.current_stream, torch.cuda.current_stream().cuda_stream)

        return None, grad_hidden[:, :, :ctx.saved_tensors[0].size(-1)], None, None

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
        
        dec_buffer = [
            torch.empty(dec_hidden.size(), device=dec_hidden.device, dtype=torch.half)
                for _ in range(model.num_dec)
        ]
        
        wait_stream(torch.cuda.current_stream().cuda_stream, inf_ctx.current_stream)
        model.decode_requires_grad(
            inf_ctx,
            torch_to_tensor(dec_hidden), torch_to_tensor(enc_hidden),
            dec_mask, enc_mask,
            [ torch_to_tensor(x) for x in dec_buffer ]
        )
        wait_stream(inf_ctx.current_stream, torch.cuda.current_stream().cuda_stream)

        ctx.save_for_backward(orig_dec, orig_enc, *dec_buffer)
        ctx.inf_ctx = inf_ctx
        ctx.dec_mask = dec_mask
        ctx.enc_mask = enc_mask
        ctx.model = model
        
        return dec_hidden[:, :, :orig_dec.size(-1)]
    
    @staticmethod
    def backward(ctx, grad_hidden : torch.Tensor):
        model : T5Model = ctx.model
        dec_hidden : torch.Tensor = clone_tensor(ctx.saved_tensors[0])
        enc_hidden : torch.Tensor = clone_tensor(ctx.saved_tensors[1])
        dec_buffer : List[torch.Tensor] = list(ctx.saved_tensors[2:])
        grad_hidden = clone_tensor(grad_hidden)

        grad_enc = torch.empty(
            enc_hidden.size(), device=enc_hidden.device, dtype=torch.half
        )

        wait_stream(torch.cuda.current_stream().cuda_stream, ctx.inf_ctx.current_stream)
        model.decode_backward(
            ctx.inf_ctx,
            torch_to_tensor(dec_hidden), torch_to_tensor(enc_hidden),
            ctx.dec_mask, ctx.enc_mask,
            [ torch_to_tensor(x) for x in dec_buffer ],
            torch_to_tensor(grad_enc), torch_to_tensor(grad_hidden)
        )
        wait_stream(ctx.inf_ctx.current_stream, torch.cuda.current_stream().cuda_stream)

        return None, grad_hidden[:, :, :ctx.saved_tensors[0].size(-1)], None, grad_enc[:, :, :ctx.saved_tensors[1].size(-1)], None, None

class TorchT5Tokenizer:
    def __init__(self, tokenizer : T5Tokenizer) -> None:
        self.tokenizer = tokenizer

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    def __len__(self):
        return len(self.tokenizer)
    
    @property
    def sod_token_id(self) -> int:
        return self.tokenizer.sod_id
    
    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eod_id

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.encoder['<pad>']

    def num_special_tokens_to_add(self) -> int:
        return 0
    
    def get_special_tokens_mask(self, ids : List[int]) -> List[int]:
        return [0 for _ in ids]
    
    def build_inputs_with_special_tokens(self, ids : List[int]) -> List[int]:
        return [x for x in ids]

    @property
    def additional_special_tokens_ids(self) -> List[int]:
        return self.tokenizer.sentinel_list
    
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
    
    def encode(self, text : Union[str, List[str]], **kwargs) -> List[int]:
        if isinstance(text, str):
            return self.tokenizer.encode(text)
        elif isinstance(text, list):
            return [self.tokenizer.encode(t) for t in text]
        else:
            raise TypeError("text must be str or list[str]")
    
    def decode(self, ids : torch.Tensor) -> Union[str, List[str]]:
        assert ids.dtype == torch.int64 or ids.dtype == torch.int32
        if ids.ndim == 1:
            return self.tokenizer.decode(ids.tolist()) 
        elif ids.ndim == 2:
            return [
                self.tokenizer.decode(s)
                for s in ids.tolist()
            ]
        else:
            raise ValueError("ids must be 1D or 2D")


class TorchT5(torch.nn.Module):
    def __init__(self,
            config : T5Configuration
        ) -> None:
        super().__init__()
        config.DEVICE = torch.cuda.current_device()

        self.device = Device(config.DEVICE)
        self._torch_device = torch.device("cuda:%d" % config.DEVICE)

        self._ctx = Context([config.DEVICE], [
            TorchAllocator()
        ])
        self._model = T5Model(config)
        self._config = config
    
    @property
    def tokenizer(self):
        return TorchT5Tokenizer(self._model.tokenizer)

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
        
        with torch.cuda.device(self._torch_device):
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
        with torch.cuda.device(self._torch_device):
            return OpT5Encode.apply(self._ctx, input_hidden, input_mask.astype(np.int8), self._model)
    
    def decode(self, dec_hidden : torch.Tensor, dec_mask : np.ndarray, enc_hidden : torch.Tensor, enc_mask : np.ndarray) -> torch.Tensor:
        """
        Args:
            dec_hidden: (batch_size, embed_dim, seq_q)        on gpu
            dec_mask:   (batch_size, seq_q)                   on cpu
            enc_hidden: (batch_size, embed_dim, seq_k)        on gpu
            enc_mask:   (batch_size, seq_k)                   on cpu
        Returns:
            hidden:     (batch_size, embed_dim, seq_q)        on gpu
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
        with torch.cuda.device(self._torch_device):
            return OpT5Decode.apply(self._ctx, dec_hidden, dec_mask.astype(np.int8), enc_hidden, enc_mask.astype(np.int8), self._model)
    
    def project(self, hidden : torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: (batch_size, embed_dim, seq_len)        on gpu
        Returns:
            logits: (batch_size, seq_len, vocab_size)       on gpu
        """
        with torch.cuda.device(self._torch_device):
            return OpT5Projection.apply(self._ctx, hidden, self._model)
    
    def get_input_embeddings(self) -> torch.Tensor:
        out = torch.empty(self._model.config.VOCAB_SIZE, self._model.config.DIM_MODEL, dtype=torch.half, device=self._torch_device)
        with torch.cuda.device(self._torch_device):
            cudart.cudaMemcpyAsync(
                out.data_ptr(), self._model.input_embedding.weight.value.ptr, self._model.input_embedding.weight.nbytes, cudart.cudaMemcpyDeviceToDevice, torch.cuda.current_stream().cuda_stream
            )
        # return out
        module = torch.nn.Embedding(self._model.config.VOCAB_SIZE, self._model.config.DIM_MODEL)
        with torch.no_grad():
            module.weight.copy_(out.detach())
        return module
            

