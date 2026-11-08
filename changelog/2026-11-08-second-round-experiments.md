# 2026-11-08 第二轮实验：三条假设全部证伪

目的：找出 141 μs（72.9% 标称带宽）的根因。

## 1. tail effect（本地推算 + 2 次远端实测）

CTA 数 3200→6400→12800，理论 tail 浪费 8%→8%→1.25%，但实测 141 / 143 / 143 μs。
**证伪**——更多 CTA 反而略慢。

## 2. occupancy（本地 ptxas -v，0 次远端配额）

`ptxas -v` 实测 I1/I4/I8/I16 × warps 4/8 全部 **32 registers、0 spill**。
A100 上对应 64 warps/SM，即 100% occupancy。**证伪**。

## 3. 加载指令宽度（新写 CUDA 版本，2 次远端配额）

Triton 3.8 恒生成标量 `ld.global.b32`，所有诱导手段无效。改用 `load_inline`
写 CUDA：`__ldg` 的 `float4` 16 字节加载 + 每线程 4 路寄存器累加 +
`__shfl_down_sync` 树形归约，跨块 double。

实测 **147 ± 0.2 μs**，比 Triton 的 141 还慢。**证伪**。

### 首次提交失败与修复

`cudaErrorIllegalAddress`。原因：`load_inline` 的 pybind 绑定不会把 Python int
直接绑到 `float*` 形参，指针被截断。改为形参用 `long long` 接收地址、函数体内
`reinterpret_cast`，test 5/5 通过。

## 结论与判断

三条硬件侧假设（tail / occupancy / 加载宽度）全部排除后，141 μs 很可能已经接近
Modal 上这张 A100 在该评测环境中的实际可达带宽。若 Modal 用的是 A100 40GB PCIe
（标称 1555 GB/s），1.487 TB/s 即 95.6%，基本没有余量。

榜首 101.237 μs 对应 2.072 TB/s，与该假设冲突，但成因不明；在每小时 6 次配额下
继续盲试性价比很低，因此暂停性能冲刺。

## 新增文件

- `kernels/vectorsum_v2/variants/cuda_f4.py`（CUDA float4 实验，已被证明更慢，
  保留作为证伪记录）
- `NOTES.md` 增补「第二轮假设验证」章节
