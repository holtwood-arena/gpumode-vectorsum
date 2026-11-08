#!POPCORN leaderboard vectorsum_v2

"""PyTorch 基线：用来判断 141 μs 究竟是我的实现不够快，还是硬件就到这。

如果 torch.sum 也在 140 μs 附近，说明 cuBLAS/reduction 的成熟实现同样只能做到
~1.5 TB/s，那 72.9% 的"利用率"就不是实现问题，而是 Modal A100 的真实带宽。
"""

import torch
from task import input_t, output_t


def _custom_kernel(data: input_t) -> output_t:
    x, out = data
    torch.sum(x, dim=0, out=out[0])
    return out[0]


custom_kernel = _custom_kernel
