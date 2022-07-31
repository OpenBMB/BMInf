#include <cuda_fp16.h>


template<typename T>
__inline__ __device__ T warpReduceMax(T x) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) 
        x = fmaxf(x, __shfl_down_sync(0xFFFFFFFF, x, offset));
    return x;
}

template<typename T>
__inline__ __device__ T blockReduceMax(T x) {
    static __shared__ T shared[32]; // blockDim.x / warpSize
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    x = warpReduceMax(x);
    if (lane == 0) shared[wid] = x;
    __syncthreads();
    x = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : T(-INFINITY);
    if (wid == 0) x = warpReduceMax(x);
    return x;
}

__device__ half abs(const half x) {
    return __habs(x);
}

__device__ half max(const half a, const half b) {
#if __CUDA_ARCH__ >= 800
    return __hmax(a, b);
#else
    return fmaxf((float)a, float(b));
#endif
}

// gridDim(n, 1, 1),    blockDim(m, 1, 1)
template<typename T>
__device__ void calc_scale(
    int32_t m,
    const T* mat,       // (n, m)
    T* out              // (n)
) {
    T local_max = 0;
    int32_t base_idx = blockIdx.x * m;
    for (int32_t i = threadIdx.x; i < m; i += blockDim.x){
        local_max =  max(local_max, abs(mat[base_idx + i]));
    }
    local_max = blockReduceMax<T>(local_max);

    if (threadIdx.x == 0) {
        out[ blockIdx.x ] = local_max / T(127);
    }
}

// gridDim(n, m / 1024, 1),    blockDim(1024, 1, 1)
template<typename T>
__device__ void round_int8(
    int32_t m,
    const T* mat,   // (n, m)
    const T* scale, // (n,)
    int8_t *out     // (n, m)
) {
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    int pos = blockIdx.x * m + col;
    T s = scale[blockIdx.x];
    if (col < m) {
        out[pos] = (int8_t)nearbyintf((float)(mat[pos] / s));
    }
}

// gridDim(n, m / 1024, 1),    blockDim(1024, 1, 1)
template<typename T>
__device__ void scale(
    int32_t m,
    const int32_t* mat,   // (n, m)
    const T* scale_x,     // (n,)
    const T* scale_y,     // (m,)
    T* out          // (n, m)
) {
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    int pos = blockIdx.x * m + col;
    if (col < m) {
        T s1 = scale_x ? scale_x[blockIdx.x] : T(1);
        T s2 = scale_y ? scale_y[col] : T(1);
        out[pos] = T(float(mat[pos]) * float(s1) * float(s2));
    }
}

// gridDim(n, 1, 1),    blockDim(m, 1, 1)
template<typename T>
__device__ void scale_round(
    int32_t m,
    const T* mat,       // (n, m)
    const T* scale_y,   // (m,)
    T* scale_out,       // (n,)
    int8_t* out         // (n, m)
) {
    __shared__ T global_max;

    T local_max = 0;
    int32_t base_idx = blockIdx.x * m;
    for (int32_t i = threadIdx.x; i < m; i += blockDim.x) {
        local_max =  max(local_max, abs(mat[base_idx + i] * scale_y[i]));
    }
    local_max = blockReduceMax<T>(local_max);
    if (threadIdx.x == 0) {
        local_max = local_max / T(127.0);
        global_max = local_max;
        scale_out[blockIdx.x] = local_max;
    }
    __syncthreads();
    local_max = global_max;
    for (int32_t i = threadIdx.x; i < m; i += blockDim.x) {
        out[base_idx + i] = (int8_t)nearbyintf(float(mat[base_idx + i] * scale_y[i] / local_max));
    }
}

extern "C" __global__ void bminf_linear_calc_scale_half(
    int32_t m,
    const half *mat,        // n, m
    half *out  // n
) {
    calc_scale<half>(m, mat, out);
}

extern "C" __global__ void bminf_linear_calc_scale_float(
    int32_t m,
    const float *mat,        // n, m
    float *out  // n
) {
    calc_scale<float>(m, mat, out);
}

extern "C" __global__ void bminf_linear_round_half(
    int32_t m,
    const half* mat,   // (n, m)
    const half* scale, // (n,)
    int8_t *out     // (n, m)
) {
    round_int8<half>(m, mat, scale, out);
}

extern "C" __global__ void bminf_linear_round_float(
    int32_t m,
    const float* mat,   // (n, m)
    const float* scale, // (n,)
    int8_t *out     // (n, m)
) {
    round_int8<float>(m, mat, scale, out);
}

extern "C" __global__ void bminf_linear_scale_half(
    int32_t m,
    const int32_t* mat,   // (n, m)
    const half* scale_x,  // (n,)
    const half* scale_y,  // (m,)
    half* out       // (n, m)
) {
    scale<half>(m, mat, scale_x, scale_y, out);
}

extern "C" __global__ void bminf_linear_scale_float(
    int32_t m,
    const int32_t* mat,    // (n, m)
    const float* scale_x,  // (n,)
    const float* scale_y,  // (m,)
    float* out       // (n, m)
) {
    scale<float>(m, mat, scale_x, scale_y, out);
}

extern "C" __global__ void bminf_linear_scale_round_half(
    int32_t m,
    const half* mat,        // (n, m)
    const half* scale_y,    // (m,)
    half* scale_out,        // (n,)
    int8_t* out             // (n, m)
) {
    scale_round<half>(m, mat, scale_y, scale_out, out);
}

extern "C" __global__ void bminf_linear_scale_round_float(
    int32_t m,
    const float* mat,       // (n, m)
    const float* scale_y,   // (m,)
    float* scale_out,       // (n,)
    int8_t* out             // (n, m)
) {
    scale_round<float>(m, mat, scale_y, scale_out, out);
}
