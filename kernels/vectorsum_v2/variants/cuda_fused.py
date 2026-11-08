#!POPCORN leaderboard vectorsum_v2

"""CUDA 单内核融合：partial + final 合并成一个 kernel，末尾 CTA 用 atomic ticket 收尾。

动机：单遍成绩 141.865 μs 里约有 14.6 μs 是首次读的固定开销（由 3-pass 探针反推）。
本版把两次 kernel launch 压成一次，用来判断这部分开销里有多少是 kernel 边界。

正确性保障：
  - 每个 CTA 写 partial[blockIdx.x] 后执行 __threadfence()，再用 atomicAdd 取 ticket
  - 拿到最后一个 ticket（old == m-1）的 CTA 读全部 partial 做最终归约
  - 收尾时把 counter 写回 0，供下次调用复用；首次由 torch.zeros 初始化
"""

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

_CUDA_SRC = r"""
#include <cuda_runtime.h>

__global__ void fused_reduce(const float* __restrict__ x,
                             double* __restrict__ partial,
                             float* __restrict__ out,
                             int* __restrict__ counter,
                             long n, int m) {
    long n4 = n / 4;
    long per_block = (n4 + gridDim.x - 1) / gridDim.x;
    long start = (long)blockIdx.x * per_block;
    long end   = start + per_block;
    if (end > n4) end = n4;

    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (long i = start + threadIdx.x; i < end; i += blockDim.x) {
        float4 v = __ldg(x4 + i);
        acc.x += v.x;
        acc.y += v.y;
        acc.z += v.z;
        acc.w += v.w;
    }
    float s = (acc.x + acc.y) + (acc.z + acc.w);

    /* n 不是 4 的倍数时补尾部 */
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (long i = n4 * 4; i < n; ++i) s += x[i];
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_down_sync(0xffffffffu, s, off);

    __shared__ float ws[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    int nw   = blockDim.x >> 5;
    if (lane == 0) ws[wid] = s;
    __syncthreads();

    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int i = 0; i < nw; ++i) t += ws[i];
        partial[blockIdx.x] = (double)t;
    }

    /* 保证 partial 写对其他 CTA 可见，再抢 ticket */
    __threadfence();

    __shared__ bool is_last;
    if (threadIdx.x == 0) {
        int old = atomicAdd(counter, 1);
        is_last = (old == m - 1);
    }
    __syncthreads();

    if (is_last) {
        double s2 = 0.0;
        for (int i = threadIdx.x; i < m; i += blockDim.x)
            s2 += partial[i];

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            s2 += __shfl_down_sync(0xffffffffu, s2, off);

        __shared__ double ws2[32];
        if (lane == 0) ws2[wid] = s2;
        __syncthreads();

        if (threadIdx.x == 0) {
            double t = 0.0;
            for (int i = 0; i < nw; ++i) t += ws2[i];
            out[0] = (float)t;
            *counter = 0;          /* 重置，供下次调用 */
        }
    }
}

void launch(long long x_ptr, long long partial_ptr, long long out_ptr,
            long long counter_ptr, long long n, long long m, long long block) {
    fused_reduce<<<(int)m, (int)block>>>(
        (const float*)x_ptr, (double*)partial_ptr, (float*)out_ptr,
        (int*)counter_ptr, n, (int)m);
}
"""

_CPP_SRC = ("void launch(long long, long long, long long, long long, "
            "long long, long long, long long);")

_MOD = load_inline(
    name="vs_reduce_fused",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["launch"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)

_GRID = 3200
_BLOCK = 256
_BUF: dict = {}


def _slot(device):
    s = _BUF.get(device)
    if s is None:
        s = (
            torch.empty(_GRID, dtype=torch.float64, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
        )
        _BUF[device] = s
    return s


def _custom_kernel(data: input_t) -> output_t:
    x, out = data
    n = x.numel()
    p, c = _slot(x.device)
    _MOD.launch(x.data_ptr(), p.data_ptr(), out.data_ptr(), c.data_ptr(),
                n, _GRID, _BLOCK)
    return out[0]


custom_kernel = _custom_kernel
