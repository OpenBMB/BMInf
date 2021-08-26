import os
import setuptools
import sys
import struct

CONFIG = {
    "package_name": "bigmodels",
    "author": "a710128",
    "author_email": "qbjooo@qq.com",
    "description": "A toolkit for big models",
    "version": None
}

def lookup_dll(prefix):
    paths = os.environ.get("PATH", "").split(os.pathsep)
    for path in paths:
        for name in os.listdir(path):
            if name.startswith(prefix) and name.lower().endswith(".dll"):
                return os.path.join(path, name)
    return None

def lookup_cuda_dll():
    cudart_lib_name = "cudart%d_" % (struct.calcsize("P") * 8)
    return lookup_dll(cudart_lib_name)

def lookup_cublas_dll():
    cublas_lib_name = "cublas%d_" % (struct.calcsize("P") * 8)
    return lookup_dll(cublas_lib_name)

def get_cuda_version():
    env_version = os.environ.get("CUDA_VERSION", None)
    if env_version is not None:
        return int(env_version)

    if sys.platform.startswith("win"):
        dll_path = lookup_cuda_dll()
        if dll_path is None:
            raise RuntimeError("Couldn't find CUDA runtime on windows")
        import ctypes
        lib = ctypes.WinDLL(dll_path)
    else:
        import ctypes
        import ctypes.util
        lib_name = ctypes.util.find_library("cudart")
        if lib_name is None:
            raise RuntimeError("Couldn't find CUDA runtime")
        lib = ctypes.cdll.LoadLibrary(lib_name)

    version_func = getattr(lib, "cudaRuntimeGetVersion")
    version_func.restype = ctypes.c_int
    version_func.argtypes = [ ctypes.POINTER(ctypes.c_int) ]
    version = ctypes.c_int()
    cuda_status = version_func( ctypes.byref(version) )
    if cuda_status != 0:
        raise RuntimeError("Failed to query CUDA version, cudaStatus_t = %d" % cuda_status)
    major = version.value // 1000
    minor = (version.value % 1000) // 10
    return major * 10 + minor

def check_cublas():
    if sys.platform.startswith("win"):
        if lookup_cublas_dll() is None:
            raise RuntimeError("Couldn't find cublas on windows")
    else:
        import ctypes
        import ctypes.util
        if ctypes.util.find_library("cublas") is None:
            raise RuntimeError("Couldn't find cublas")

def get_readme(path):
    ret = ""
    with open(os.path.join(path, "README.md"), encoding="utf-8") as frd:
        ret = frd.read()
    return ret

def get_requirements(path):
    ret = []
    with open(os.path.join(path, "requirements.txt"), encoding="utf-8") as freq:
        for line in freq.readlines():
            ret.append( line.strip() )
    return ret

def get_version():
    if "version" in CONFIG and CONFIG["version"] is not None:
        return CONFIG["version"]
    if "CI_COMMIT_TAG" in os.environ:
        return os.environ["CI_COMMIT_TAG"]
    if "CI_COMMIT_SHA" in os.environ:
        return os.environ["CI_COMMIT_SHA"]
    return "test"

def main():
    path = os.path.dirname(os.path.abspath(__file__))

    version = get_version()
    open( os.path.join(path, CONFIG["package_name"], "version.py"), "w", encoding="utf-8" ).write('__version__ = "%s"' % version)

    requires = get_requirements(path)

    if os.environ.get("BM_BUILD", None) is not None:
        check_cublas()
        cuda_version = get_cuda_version()
        cupy_version = "cupy-cuda%d>=8,<10" % cuda_version
        requires = requires + [cupy_version]

    setuptools.setup(
        name=CONFIG["package_name"],
        version=version,
        author=CONFIG["author"],
        author_email=CONFIG["author_email"],
        description=CONFIG["description"],
        long_description=get_readme(path),
        long_description_content_type="text/markdown",
        packages=setuptools.find_packages(exclude=("tools",)),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: C++"
        ],
        python_requires=">=3.6",
        setup_requires=["wheel"],
        install_requires= requires,
    )

if __name__ == "__main__":
    main()
