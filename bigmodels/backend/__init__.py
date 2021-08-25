import cupy

if hasattr(cupy, "_core"):
    core = cupy._core
else:
    core = cupy.core

create_ufunc = core.create_ufunc
create_reduction_func = core.create_reduction_func

cublas = cupy.cuda.cublas
cupy_ver = 0

def _check_version():
    global cupy_ver
    import cupy
    if not (cupy.__version__.startswith("9.") or cupy.__version__.startswith("8.")):
        raise RuntimeError("cupy 8/9 is required")
    if cupy.__version__.startswith("9."):
        cupy_ver = 9
    else:
        cupy_ver = 8
    
_check_version()

del _check_version