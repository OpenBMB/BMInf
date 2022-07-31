import torch
from cpm_kernels.kernels import gemm_int8
from ..kernels import gemm_calc_scale, gemm_round, gemm_scale, gemm_scale_round

class OpLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp : torch.Tensor, weight : torch.Tensor, weight_scale : torch.Tensor):
        """
        Args:
            inp:    (..., in_features)          dtype
            weight: (out_features, in_features) int8
            scale:  (out_featues,)              dtype
        Returns:
            out:    (..., out_features)
        """
        assert inp.dtype == weight_scale.dtype
        assert inp.device == weight.device
        assert inp.is_cuda

        ctx.inp_shape = inp.size()
        x = inp.contiguous().view(-1, weight.size(1))
        scale_x = gemm_calc_scale(x)        # (...,)
        quant_x = gemm_round(x, scale_x)    # (..., in_features)    int8

        M = quant_x.size(0)
        K = quant_x.size(1)
        N = weight.size(0)

        with torch.cuda.device(inp.device):
            out = torch.empty(M, N, dtype=torch.int32, device="cuda")   # (..., out_features)
            gemm_int8(
                N, K, M,
                1, 1,
                True, False,
                weight.data_ptr(), quant_x.data_ptr(),
                out.data_ptr(),
                torch.cuda.current_stream().cuda_stream
            )
        out = gemm_scale(out, scale_x, weight_scale, inp.dtype)

        ctx.save_for_backward(weight, weight_scale)
        return out.view( *(ctx.inp_shape[:-1] + (N,)) )
    
    @staticmethod
    def backward(ctx, grad_f : torch.Tensor):
        weight, weight_scale = ctx.saved_tensors
        grad_f = grad_f.contiguous().view(-1, weight.size(0))   # (..., out_features)
        
        quant_grad, scale_grad = gemm_scale_round(grad_f, weight_scale)

        with torch.cuda.device(quant_grad.device):
            out = torch.empty(grad_f.size(0), weight.size(1), dtype=torch.int32, device="cuda")
            gemm_int8(
                weight.size(1), weight.size(0), grad_f.size(0),
                1, 1,
                False, False,
                weight.data_ptr(), quant_grad.data_ptr(),
                out.data_ptr(),
                torch.cuda.current_stream().cuda_stream
            )
        out = gemm_scale(out, scale_grad, None, grad_f.dtype)
        return out.view(ctx.inp_shape), None, None

class QuantizedLinear(torch.nn.Module):
    def __init__(self, linear : torch.nn.Linear):
        super().__init__()

        if not linear.weight.is_cuda:
            w = linear.weight.cuda()
            self.weight_scale = gemm_calc_scale(w)
            self.weight_quant = torch.nn.Parameter(gemm_round(w, self.weight_scale).cpu(), requires_grad=False)
            self.weight_scale = torch.nn.Parameter(self.weight_scale.cpu(), requires_grad=False)
        else:
            self.weight_scale = torch.nn.Parameter(gemm_calc_scale(linear.weight), requires_grad=False)
            self.weight_quant = torch.nn.Parameter(gemm_round(linear.weight, self.weight_scale), requires_grad=False)
        self.bias = linear.bias
        self.in_features = linear.in_features
        self.out_features = linear.out_features
    
    def forward(self, x):
        out = OpLinear.apply(x, self.weight_quant, self.weight_scale)
        if self.bias is not None:
            out = out + self.bias
        return out
        

