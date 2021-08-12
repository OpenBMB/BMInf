import cupy

if hasattr(cupy, "_core"):
    core = cupy._core
else:
    core = cupy.core

create_ufunc = core.create_ufunc
create_reduction_func = core.create_reduction_func

cublas = cupy.cuda.cublas