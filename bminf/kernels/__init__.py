from typing import List, Optional
from cpm_kernels.kernels.base import LazyKernelCModule, KernelFunction, DevicePointer, CUDAStream, round_up
import pkg_resources
import torch
import ctypes

RESOURCE_PACKAGE_NAME = __name__

class Kernel:
    def __init__(self, filename : str, function_names : List[str]):
        filename = filename + ".fatbin"
        if not pkg_resources.resource_exists(RESOURCE_PACKAGE_NAME, filename):
            raise RuntimeError("File `%s` not found in `%s`" % (filename, RESOURCE_PACKAGE_NAME))
        self.filename = filename
        self.code = pkg_resources.resource_string(RESOURCE_PACKAGE_NAME, filename)
        self._function_names = function_names
        self._cmodule = LazyKernelCModule(self.code)

        for name in self._function_names:
            setattr(self, name, KernelFunction(self._cmodule, name))

scale_kernel = Kernel(
    "scale",
    [
        "bminf_linear_calc_scale_half",
        "bminf_linear_calc_scale_float",
        "bminf_linear_round_half",
        "bminf_linear_round_float",
        "bminf_linear_scale_half",
        "bminf_linear_scale_float",
        "bminf_linear_scale_round_half",
        "bminf_linear_scale_round_float"
    ]
)

def _calc_scale(
        dtype : torch.dtype, n : int, m : int,
        mat : DevicePointer,    # (n, m)
        out : DevicePointer,    # (n)
        stream : CUDAStream
    ):
    gridDim = (n, 1, 1)
    blockDim = (min(round_up(m, 32), 1024), 1, 1)
    func = scale_kernel.bminf_linear_calc_scale_float
    if dtype == torch.half:
        func = scale_kernel.bminf_linear_calc_scale_half
    func(
        gridDim, blockDim, 0, stream, [
            ctypes.c_int32(m),
            ctypes.c_void_p(mat),
            ctypes.c_void_p(out)
        ]
    )

def _round(
        dtype : torch.dtype, n : int, m : int,
        mat : DevicePointer,    # (n, m)
        scale : DevicePointer,  # (n,)
        out : DevicePointer,    # (n, m)    int8
        stream : CUDAStream
    ):
    threads = min(round_up(m, 32), 1024)
    gridDim = (n, round_up(m, threads) // threads, 1)
    blockDim = (threads, 1, 1)
    func = scale_kernel.bminf_linear_round_float
    if dtype == torch.half:
        func = scale_kernel.bminf_linear_round_half
    func(
        gridDim, blockDim, 0, stream, [
            ctypes.c_int32(m),
            ctypes.c_void_p(mat),
            ctypes.c_void_p(scale),
            ctypes.c_void_p(out)
        ]
    )

def _scale(
        dtype : torch.dtype, n : int, m : int,
        mat : DevicePointer,        # (n, m)    int8
        scale_x : DevicePointer,    # (n,)
        scale_y : DevicePointer,    # (n,)
        out : DevicePointer,        # (n, m)
        stream : CUDAStream
    ):
    threads = min(round_up(m, 32), 1024)
    gridDim = (n, round_up(m, threads) // threads, 1)
    blockDim = (threads, 1, 1)
    
    func = scale_kernel.bminf_linear_scale_float
    if dtype == torch.half:
        func = scale_kernel.bminf_linear_scale_half
    func(
        gridDim, blockDim, 0, stream, [
            ctypes.c_int32(m),
            ctypes.c_void_p(mat),
            ctypes.c_void_p(scale_x),
            ctypes.c_void_p(scale_y),
            ctypes.c_void_p(out),
        ]
    )

def _scale_round(
        dtype : torch.dtype,
        n : int, m : int,
        mat : DevicePointer,        # (n, m)    dtype
        scale_y : DevicePointer,    # (m,)      dtype
        scale_x : DevicePointer,    # (n,)      dtype
        out : DevicePointer,        # (n, m)    int8
        stream : CUDAStream
    ):
    gridDim = (n, 1, 1)
    blockDim = (min(round_up(m, 32), 1024), 1, 1)

    func = scale_kernel.bminf_linear_scale_round_float
    if dtype == torch.half:
        func = scale_kernel.bminf_linear_scale_round_half
    func(
        gridDim, blockDim, 0, stream, [
            ctypes.c_int32(m),
            ctypes.c_void_p(mat),
            ctypes.c_void_p(scale_y),
            ctypes.c_void_p(scale_x),
            ctypes.c_void_p(out),
        ]
    )

def gemm_calc_scale(
        mat : torch.Tensor      # (n, m)
    ) -> torch.Tensor:  # (n,)
    with torch.cuda.device(mat.device):
        out = torch.empty(mat.size(0), dtype=mat.dtype, device="cuda")
        stream = torch.cuda.current_stream()
        _calc_scale(mat.dtype, mat.size(0), mat.size(1), mat.data_ptr(), out.data_ptr(), stream.cuda_stream)
        return out

def gemm_round(
        mat : torch.Tensor,     # (n, m)
        scale : torch.Tensor,   # (n,)
    ):
    with torch.cuda.device(mat.device):
        out = torch.empty(mat.size(0), mat.size(1), dtype=torch.int8, device="cuda")
        stream = torch.cuda.current_stream()
        _round(mat.dtype, mat.size(0), mat.size(1), mat.data_ptr(), scale.data_ptr(), out.data_ptr(), stream)
        return out

def gemm_scale(
        mat : torch.Tensor,     # (n, m) int32
        scale_x : Optional[torch.Tensor],   # (n,)
        scale_y : Optional[torch.Tensor],   # (m,)
        dtype : torch.dtype
    ):
    with torch.cuda.device(mat.device):
        out = torch.empty(mat.size(0), mat.size(1), dtype=dtype, device="cuda")
        stream = torch.cuda.current_stream()
        _scale(
            dtype,
            mat.size(0), mat.size(1),
            mat.data_ptr(), 
            scale_x.data_ptr() if scale_x is not None else 0,
            scale_y.data_ptr() if scale_y is not None else 0,
            out.data_ptr(),
            stream
        )
        return out

def gemm_scale_round(
        mat : torch.Tensor,     # (n, m)    dtype
        scale_y : torch.Tensor, # (m,)      dtype
    ):
    with torch.cuda.device(mat.device):
        out = torch.empty(mat.size(0), mat.size(1), dtype=torch.int8, device="cuda")
        scale_x = torch.empty(mat.size(0), dtype=mat.dtype, device="cuda")
        stream = torch.cuda.current_stream()
        _scale_round(
            mat.dtype,
            mat.size(0), mat.size(1),
            mat.data_ptr(),
            scale_y.data_ptr(),
            scale_x.data_ptr(),
            out.data_ptr(),
            stream.cuda_stream
        )
        return out, scale_x
