# gpumode-vectorsum

[GPU MODE](https://www.gpumode.com) **PMPP_v2** 题集里 `vectorsum_v2` 的解题记录：
一个 Triton 实现的向量求和归约，以及围绕它的调参、诊断与提交过程。

## 成绩

| 项 | 值 |
| --- | --- |
| Leaderboard | `vectorsum_v2`（[榜单 544](https://www.gpumode.com/leaderboard/544)） |
| 卡型 | NVIDIA A100 |
| 成绩 | **141.865 μs** |
| 名次 | **43 / 120** |
| 提交 id | 969448 |

题目要求把 `(N,)` 的 float32 张量求和成单个标量，计分只取最大的一档
`N = 52,428,800`（200 MiB）。对应实测带宽 1.48 TB/s，约为 A100 标称
2.039 TB/s 的 72.6%。

## 核心实现

`kernels/vectorsum_v2/submission.py`：

- **两阶段归约**：第一个 kernel 每个 program 处理 16,384 个元素
  （`BLOCK=4096 × ITERS=4`），块内 fp32 逐 lane 累加后做树形归约，
  部分和以 float64 写回；第二个 kernel 把这些部分和归约成 float32 标量。
- **精度策略**：块内用 fp32 树形归约，跨块用 float64 累加。这个组合在
  `offset∈[-100,100]` 分布的 CPU 模拟下最坏情形仍有 ≥62x 容差余量；
  而"跨块用 fp32 顺序累加"只有 0.076x，会直接失败。
- **无 mask 快路径**：所有 benchmark 尺寸都是每 program 元素数的整数倍，
  用 `EVEN` constexpr 跳过边界判断；测试尺寸走带 mask 的分支。
- 加载使用 `eviction_policy="evict_first"`（流式读，不污染 L2）。

## 几个容易踩的坑

1. **官方示例不能照抄**。模板首行写的是旧榜名 `vectorsum_py`，与现役榜
   `vectorsum_v2` 不一致会被 kernelbot 直接拒绝；模板还把 `atomic_add`
   打到 `torch.empty(1)` 的未初始化内存上。
2. **精度不是难点，归约拓扑才是**。官方 `solutions/correct` 就是 fp32
   `data.sum()`。危险的是跨块顺序 fp32 累加，不是块内 fp32 树。
3. **测量噪声很大**。同一份代码连续提交，public 分数在
   141.865–171.653 μs 之间波动（约 21%），secret 侧则稳定在 141.8–142.0。
   榜单保留 best score，所以名次不完全反映代码差异。
4. **提交限流**：`test` / `benchmark` 共享每小时 6 次配额。

## 目录

```
kernels/vectorsum_v2/
  submission.py        # 正式提交版本（B4096_I4_W8）
  variants/            # 参数扫描用的配置变体
SCORES.md              # 每次提交的时间/卡型/模式/μs/名次，含失败记录
NOTES.md               # 题面事实、已证伪方向、参数扫描与 PTX 诊断结论
changelog/             # 逐次变更记录
```

## 未完成的方向

当前 72.6% 的带宽利用率仍未打满。已用本地离线编译（不消耗远端配额）排除了
Triton 未向量化加载这一假设——SIMT 下一条 `ld.global.b32` 由整个 warp 执行、
搬 128 字节，指令发射不是瓶颈。剩余嫌疑是 occupancy 与 tail effect，
需要在配额空闲时段连续实测才能定位。
