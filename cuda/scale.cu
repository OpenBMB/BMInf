#include <cuda_fp16.h>


template<typename T>
__inline__ __device__ T warpReduceMax(T x) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) 
        x = fmaxf(x, __shfl_down_sync(0xFFFFFFFF, x, offset));
    return x;
}

__inline__ __device__ T blockReduceMax(T x) {
    static __shared__ T shared[WARP_SZ]; // blockDim.x / warpSize
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    x = warpReduceMax(x);
    if (lane == 0) shared[wid] = x;
    __syncthreads();
    x = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : T(-INFINITY);
    if (wid == 0) x = warpReduceMax(x);
    return x;
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
        out[pos] = (int8_t)nearbyint(mat[pos] / scale);
    }
}

// gridDim(n, m / 1024, 1),    blockDim(1024, 1, 1)
template<typename T>
__device__ void scale(
    int32_t m,
    int8_t* mat, // (n, m)
    T* scale_x,  // (n,)
    T* scale_y,  // (m,)
    T* out       // (n, m)
) {
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    int pos = blockIdx.x * m + col;
    if (col < m) {
        T s1 = scale_x ? scale_x[blockIdx.x] : T(1);
        T s2 = scale_y ? scale_y[col] : T(1);
        out[pos] = T(mat[pos]) * s1 * s2;
    }
}

extern "C" __global__ void bminf_linear_calc_scale_half(
    int32_t m,
    const half *mat,        // n, m
    half *out  // n
) {
    calc_scale<half>(n, m, mat, out);
}

extern "C" __global__ void bminf_linear_calc_scale_float(
    int32_t m,
    const float *mat,        // n, m
    float *out  // n
) {
    calc_scale<float>(n, m, mat, out);
}

extern "C" __global__ void bminf_linear_round_half(
    int32_t m,
    const half* mat,   // (n, m)
    const half* scale, // (n,)
    int8_t *out     // (n, m)
) {
    round_int8<half>(n, mat, scale, out);
}

extern "C" __global__ void bminf_linear_round_float(
    int32_t m,
    const float* mat,   // (n, m)
    const float* scale, // (n,)
    int8_t *out     // (n, m)
) {
    round_int8<float>(n, mat, scale, out);
}

extern "C" __global__ void bminf_linear_scale_half(
    int32_t m,
    int8_t* mat,    // (n, m)
    half* scale_x,  // (n,)
    half* scale_y,  // (m,)
    half* out       // (n, m)
) {
    scale<half>(m, mat, scale_x, scale_y, out);
}

extern "C" __global__ void bminf_linear_scale_float(
    int32_t m,
    int8_t* mat,    // (n, m)
    float* scale_x,  // (n,)
    float* scale_y,  // (m,)
    float* out       // (n, m)
) {
    scale<float>(m, mat, scale_x, scale_y, out);
}

