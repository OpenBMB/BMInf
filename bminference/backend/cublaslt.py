import ctypes, sys, struct
from ..utils.lib import LibFunction, lookup_dll, unix_find_lib
if sys.platform.startswith("win"):
    dll_path = lookup_dll("cublasLt%d_" % (struct.calcsize("P") * 8))
    lib = ctypes.WinDLL(dll_path)
else:
    lib_name = unix_find_lib("cublasLt")
    lib = ctypes.cdll.LoadLibrary(lib_name)

CUBLASLT_VERSION = getattr(lib, "cublasLtGetVersion")() // 1000

class cublasLt:
    VERSION = CUBLASLT_VERSION 
    cublasLtHandle_t = ctypes.c_void_p
    cublasLtMatrixTransformDesc_t = ctypes.c_void_p
    cudaStream_t = ctypes.c_void_p
    cublasLtMatmulDesc_t = ctypes.c_void_p
    cublasLtMatrixLayout_t = ctypes.c_void_p
    cudaDataType = ctypes.c_int
    cublasStatus_t = ctypes.c_int
    cublasComputeType_t = ctypes.c_int
    cublasLtMatmulDescAttributes_t = ctypes.c_int
    cublasLtMatrixLayoutAttribute_t = ctypes.c_int
    cublasLtMatrixTransformDescAttributes_t = ctypes.c_int

    CUDA_R_8I = 3
    CUDA_R_32I = 10
    CUDA_R_16F = 2
    CUDA_R_32F = 0

    CUBLAS_OP_N = 0
    CUBLAS_OP_T = 1

    CUBLASLT_ORDER_COL = 0
    CUBLASLT_ORDER_ROW = 1
    CUBLASLT_ORDER_COL32 = 2
    CUBLASLT_ORDER_COL4_4R2_8C = 3
    CUBLASLT_ORDER_COL32_2R_4R4 = 4


    CUBLASLT_MATRIX_LAYOUT_TYPE = 0
    CUBLASLT_MATRIX_LAYOUT_ORDER = 1
    CUBLASLT_MATRIX_LAYOUT_ROWS = 2
    CUBLASLT_MATRIX_LAYOUT_COLS = 3
    CUBLASLT_MATRIX_LAYOUT_LD = 4
    CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 5
    CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 6
    CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET = 7

    CUBLASLT_MATMUL_DESC_COMPUTE_TYPE = 0
    CUBLASLT_MATMUL_DESC_SCALE_TYPE = 1
    CUBLASLT_MATMUL_DESC_POINTER_MODE = 2
    CUBLASLT_MATMUL_DESC_TRANSA = 3
    CUBLASLT_MATMUL_DESC_TRANSB = 4
    CUBLASLT_MATMUL_DESC_TRANSC = 5
    CUBLASLT_MATMUL_DESC_FILL_MODE = 6
    CUBLASLT_MATMUL_DESC_EPILOGUE = 7
    CUBLASLT_MATMUL_DESC_BIAS_POINTER = 8

    CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE = 0
    CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE = 1
    CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA = 2
    CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB = 3

    CUBLAS_COMPUTE_16F               = 64
    CUBLAS_COMPUTE_16F_PEDANTIC      = 65
    CUBLAS_COMPUTE_32F               = 68
    CUBLAS_COMPUTE_32F_PEDANTIC      = 69
    CUBLAS_COMPUTE_32F_FAST_16F      = 74
    CUBLAS_COMPUTE_32F_FAST_16BF     = 75
    CUBLAS_COMPUTE_32F_FAST_TF32     = 77
    CUBLAS_COMPUTE_64F               = 70
    CUBLAS_COMPUTE_64F_PEDANTIC      = 71
    CUBLAS_COMPUTE_32I               = 72
    CUBLAS_COMPUTE_32I_PEDANTIC      = 73

    @staticmethod
    def checkCublasStatus(cublas_status):
        cublas_errors = {
            0: "CUBLAS_STATUS_SUCCESS",
            1: "CUBLAS_STATUS_NOT_INITIALIZED",
            3: "CUBLAS_STATUS_ALLOC_FAILED",
            7: "CUBLAS_STATUS_INVALID_VALUE",
            8: "CUBLAS_STATUS_ARCH_MISMATCH",
            11: "CUBLAS_STATUS_MAPPING_ERROR",
            13: "CUBLAS_STATUS_EXECUTION_FAILED",
            14: "CUBLAS_STATUS_INTERNAL_ERROR",
            15: "CUBLAS_STATUS_NOT_SUPPORTED",
            16: "CUBLAS_STATUS_LICENSE_ERROR"
        }
        if cublas_status == 0:
            return
        if cublas_status in cublas_errors:
            raise RuntimeError("cublas error: %s" % cublas_errors[cublas_status])
        else:
            raise RuntimeError("cublas error code: %d" % cublas_status)
    
    cublasLtCreate = LibFunction(lib, "cublasLtCreate", ctypes.POINTER(cublasLtHandle_t), cublasStatus_t)
    cublasLtDestroy = LibFunction(lib, "cublasLtDestroy", cublasLtHandle_t, cublasStatus_t)
    cublasLtGetVersion = LibFunction(lib, "cublasLtGetVersion", ctypes.c_size_t)
    cublasLtMatrixTransformDescCreate = LibFunction(lib, "cublasLtMatrixTransformDescCreate", ctypes.POINTER(cublasLtMatrixTransformDesc_t), cudaDataType, cublasStatus_t)
    if CUBLASLT_VERSION == 10:
        cublasLtMatmulDescCreate = LibFunction(lib, "cublasLtMatmulDescCreate", ctypes.POINTER(cublasLtMatmulDesc_t), cudaDataType, cublasStatus_t)
    else:
        cublasLtMatmulDescCreate = LibFunction(lib, "cublasLtMatmulDescCreate", ctypes.POINTER(cublasLtMatmulDesc_t), cublasComputeType_t, cudaDataType, cublasStatus_t)
    cublasLtMatmulDescSetAttribute = LibFunction(lib, "cublasLtMatmulDescSetAttribute", cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, ctypes.c_void_p, ctypes.c_size_t, cublasStatus_t)
    cublasLtMatrixLayoutCreate = LibFunction(lib, "cublasLtMatrixLayoutCreate", ctypes.POINTER(cublasLtMatrixLayout_t), cudaDataType, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int64, cublasStatus_t)
    cublasLtMatrixLayoutSetAttribute = LibFunction(lib, "cublasLtMatrixLayoutSetAttribute", cublasLtMatrixLayout_t, cublasLtMatrixLayoutAttribute_t, ctypes.c_void_p, ctypes.c_size_t, cublasStatus_t)
    cublasLtMatrixLayoutGetAttribute = LibFunction(lib, "cublasLtMatrixLayoutGetAttribute", cublasLtMatrixLayout_t, cublasLtMatrixLayoutAttribute_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t), cublasStatus_t)
    cublasLtMatrixTransform = LibFunction(lib, "cublasLtMatrixTransform", cublasLtHandle_t, cublasLtMatrixTransformDesc_t, ctypes.c_void_p, ctypes.c_void_p, cublasLtMatrixLayout_t, ctypes.c_void_p, ctypes.c_void_p, cublasLtMatrixLayout_t, ctypes.c_void_p, cublasLtMatrixLayout_t, cudaStream_t, cublasStatus_t)
    cublasLtMatrixTransformDescSetAttribute = LibFunction(lib, "cublasLtMatrixTransformDescSetAttribute", cublasLtMatrixTransformDesc_t, cublasLtMatrixTransformDescAttributes_t, ctypes.c_void_p, ctypes.c_size_t, cublasStatus_t)
    cublasLtMatrixTransformDescGetAttribute = LibFunction(lib, "cublasLtMatrixTransformDescGetAttribute", cublasLtMatrixTransformDesc_t, cublasLtMatrixTransformDescAttributes_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t), cublasStatus_t)
    cublasLtMatmul = LibFunction(lib, "cublasLtMatmul", cublasLtHandle_t, cublasLtMatmulDesc_t, ctypes.c_void_p, ctypes.c_void_p, cublasLtMatrixLayout_t, ctypes.c_void_p, cublasLtMatrixLayout_t, ctypes.c_void_p, ctypes.c_void_p, cublasLtMatrixLayout_t, ctypes.c_void_p, cublasLtMatrixLayout_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, cudaStream_t, cublasStatus_t)
    cublasLtMatrixLayoutDestroy = LibFunction(lib, "cublasLtMatrixLayoutDestroy", cublasLtMatrixLayout_t, cublasStatus_t)
    cublasLtMatmulDescDestroy = LibFunction(lib, "cublasLtMatmulDescDestroy", cublasLtMatmulDesc_t, cublasStatus_t)
    cublasLtMatrixTransformDescDestroy = LibFunction(lib, "cublasLtMatrixTransformDescDestroy", cublasLtMatrixTransformDesc_t, cublasStatus_t)
    