import cupy

if hasattr(cupy, "_core"):
    core = cupy._core
else:
    core = cupy.core

create_ufunc = core.create_ufunc
create_reduction_func = core.create_reduction_func

cublas = cupy.cuda.cublas

def _check_version():
    global cupy_ver
    import cupy
    if not cupy.__version__.startswith("9."):
        raise RuntimeError("cupy 9 is required")
        
_check_version()

del _check_version