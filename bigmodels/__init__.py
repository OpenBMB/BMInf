from . import models

def _check_version():
    import cupy
    if not cupy.__version__.startswith("9."):
        raise RuntimeError("cupy 9 is required")
    
_check_version()

del _check_version