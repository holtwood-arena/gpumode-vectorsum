#!POPCORN leaderboard vectorsum_v2

"""vectorsum_v2 — 带宽受限的向量求和。

设计依据（详见 NOTES.md）：
  * 计分只取最后一个 benchmark case（size=52,428,800，200 MiB）。
  * 精度关键不在块内：块内 fp32 树形归约 + 跨块 fp64 累加，模拟最坏情形余量 ≥62x；
    真正会失败的是"跨块用 fp32 顺序累加/atomic"（余量 0.076x）。
  * 所有 benchmark size 都是 _PER_PROG 的整数倍，走无 mask 快路径。
  * 绝不 atomic_add 进未清零的 output —— 用两阶段 + 最终 store。
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t

_BLOCK = 8192          # 每个 tile 的元素数
_ITERS = 4             # 每个 program 的 tile 数
_PER_PROG = _BLOCK * _ITERS
_NUM_WARPS = 8
_FINAL_BLOCK = 2048
_FINAL_WARPS = 8

_PARTIAL_CACHE: dict = {}


@triton.jit
def _partial_sum_kernel(X, P, n, BLOCK: tl.constexpr, ITERS: tl.constexpr,
                        EVEN: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * (BLOCK * ITERS)
    lanes = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in tl.static_range(ITERS):
        offs = base + i * BLOCK + lanes
        if EVEN:
            acc += tl.load(X + offs, eviction_policy="evict_first")
        else:
            acc += tl.load(X + offs, mask=offs < n, other=0.0,
                           eviction_policy="evict_first")
    tl.store(P + pid, tl.sum(acc, axis=0).to(tl.float64))


@triton.jit
def _final_sum_kernel(P, OUT, m, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float64)
    for start in range(0, m, BLOCK):
        offs = start + lanes
        acc += tl.load(P + offs, mask=offs < m, other=0.0)
    tl.store(OUT, tl.sum(acc, axis=0).to(tl.float32))


def _partial_buffer(device, m):
    buf = _PARTIAL_CACHE.get(m)
    if buf is None or buf.device != device:
        buf = torch.empty(m, dtype=torch.float64, device=device)
        _PARTIAL_CACHE[m] = buf
    return buf


def _custom_kernel(data: input_t) -> output_t:
    x, out = data
    n = x.numel()
    m = triton.cdiv(n, _PER_PROG)
    p = _partial_buffer(x.device, m)
    _partial_sum_kernel[(m,)](
        x, p, n,
        BLOCK=_BLOCK, ITERS=_ITERS, EVEN=(n % _PER_PROG == 0),
        num_warps=_NUM_WARPS,
    )
    _final_sum_kernel[(1,)](p, out, m, BLOCK=_FINAL_BLOCK, num_warps=_FINAL_WARPS)
    return out[0]


custom_kernel = _custom_kernel
