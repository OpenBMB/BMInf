
import os
def lookup_dll(prefix):
    paths = os.environ.get("PATH", "").split(os.pathsep)
    for path in paths:
        for name in os.listdir(path):
            if name.startswith(prefix) and name.lower().endswith(".dll"):
                return os.path.join(path, name)
    return None

def unix_find_lib(name):
    import ctypes.util
    
    cuda_path = os.environ.get("CUDA_PATH", None)
    if cuda_path is not None:
        lib_name = os.path.join(cuda_path, "lib64", "lib%s.so" % name)
        if os.path.exists(lib_name):
            return lib_name

    cuda_path = "/usr/local/cuda"
    if cuda_path is not None:
        lib_name = os.path.join(cuda_path, "lib64", "lib%s.so" % name)
        if os.path.exists(lib_name):
            return lib_name

    lib_name = ctypes.util.find_library(name)
    return lib_name

class LibFunction:
    def __init__(self, lib, name, *args) -> None:
        if len(args) > 0:
            ret = args[-1]
            args = args[:-1]
        else:
            ret = None

        self._name = name
        self._args = args
        self._ret = ret

        self._func = getattr(lib, self._name)
        self._func.argtypes = self._args
        self._func.restype = self._ret

    def __call__(self, *args):
        return self._func(*args)