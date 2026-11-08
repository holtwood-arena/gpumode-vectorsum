#!POPCORN leaderboard vectorsum_v2

"""CUDA float4 版本：绕开 Triton 只生成标量 ld.global.b32 的限制。

块内：float4 向量加载 + 每线程寄存器累加（4 路）+ warp/block 树形归约（float32），
跨块：double 累加。与 Triton 版的精度结构一致。
"""

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

_CUDA_SRC = r"""
#include <cuda_runtime.h>

__global__ void partial_sum_vec4(const float* __restrict__ x,
                                 double* __restrict__ partial,
                                 long n, long n4) {
    long idx  = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);

    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (long i = idx; i < n4; i += stride) {
        float4 v = __ldg(x4 + i);
        acc.x += v.x;
        acc.y += v.y;
        acc.z += v.z;
        acc.w += v.w;
    }
    float s = (acc.x + acc.y) + (acc.z + acc.w);

    /* tail: n 不是 4 的倍数时，由 0 号线程补上剩余元素 */
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (long i = n4 * 4; i < n; ++i) s += x[i];
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_down_sync(0xffffffffu, s, off);

    __shared__ float ws[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) ws[wid] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        int nw = blockDim.x >> 5;
        for (int i = 0; i < nw; ++i) t += ws[i];
        partial[blockIdx.x] = (double)t;
    }
}

__global__ void final_sum(const double* __restrict__ partial,
                          float* __restrict__ out, int m) {
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    double s = 0.0;
    for (int i = idx; i < m; i += stride)
        s += partial[i];

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_down_sync(0xffffffffu, s, off);

    __shared__ double ws[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) ws[wid] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        double t = 0.0;
        int nw = blockDim.x >> 5;
        for (int i = 0; i < nw; ++i) t += ws[i];
        out[0] = (float)t;
    }
}

void launch(long long x_ptr, long long partial_ptr, long long out_ptr,
            long long n, long long m, long long block, long long grid) {
    /* 指针以整数传入再 reinterpret_cast：pybind 不会把 Python int 直接绑到 T* */
    partial_sum_vec4<<<(int)grid, (int)block>>>(
        (float*)x_ptr, (double*)partial_ptr, n, n / 4);
    final_sum<<<1, 256>>>((const double*)partial_ptr, (float*)out_ptr, (int)m);
}
"""

_CPP_SRC = ("void launch(long long, long long, long long, "
            "long long, long long, long long, long long);")

_MOD = load_inline(
    name="vs_reduce_f4",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["launch"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)

_GRID = 4096
_BLOCK = 256
_PARTIAL: dict = {}


def _buf(device, m):
    b = _PARTIAL.get(m)
    if b is None or b.device != device:
        b = torch.empty(m, dtype=torch.float64, device=device)
        _PARTIAL[m] = b
    return b


def _custom_kernel(data: input_t) -> output_t:
    x, out = data
    n = x.numel()
    p = _buf(x.device, _GRID)
    _MOD.launch(x.data_ptr(), p.data_ptr(), out.data_ptr(), n, _GRID, _BLOCK, _GRID)
    return out[0]


custom_kernel = _custom_kernel
