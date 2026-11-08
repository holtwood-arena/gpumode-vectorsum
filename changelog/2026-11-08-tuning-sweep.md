# 2026-11-08 参数扫描与限流

## 首战结果（已完成）

`kernels/vectorsum_v2/submission.py` v2 首次提交即通过全部校验：
- test id 969428：5/5 用例
- benchmark id 969429：143 ± 0.4 μs
- leaderboard id 969430：**142.930 μs，A100 第 54 名 / 120 人**

## 参数扫描（第一轮，4 次 benchmark 提交）

| 配置 | per-program | μs |
| --- | --- | --- |
| B4096 I8 W8（首战） | 32768 | 143 |
| **B4096 I4 W8** | 16384 | **141** |
| B4096 I8 W16 | 32768 | 143 |
| B4096 I16 W8 | 65536 | 148 |
| B8192 I4 W8 | 32768 | 143 |

`_ITERS` 8→4 已写入 `submission.py`。提升仅 1.4%，判定参数调优饱和。

## PTX 诊断（本地离线编译，消耗 0 次远端配额）

- Triton 3.8 对本形态生成为标量 `ld.global.b32`，非 128-bit `v4.b32`。
- 尝试过：纯拷贝对照、2D 连续索引、`tl.max_contiguous` + `tl.multiple_of`、
  1D `tl.make_block_ptr` —— 全部仍为 `b32`。
- 结论：这不是主要瓶颈。SIMT 下一条 `ld.global.b32` 由整个 warp 执行、搬 128 字节，
  指令发射量不是限制。
- 真实瓶颈仍是带宽利用率：141 μs = 1.487 TB/s = 标称 2.039 TB/s 的 72.9%。
  榜首 101.237 μs 对应 2.072 TB/s（超标称 1.6%），说明标称偏保守。

## 限流

- `test` / `benchmark` 共享 **6 次/小时**；错误文本：
  `429 Too Many Requests: Rate limit exceeded: 6/6 test submissions per hour`。
- 第二批 4 个变体（I2/I1/W4/noev）全部被挡，未获得数据。

## 后续

- 已设置一次性定时任务（17:15，durable），自动用 B4096_I4_W8 提交 leaderboard。
- 若限流仍在，任务会读取 `Try again in Ns` 并报告，不盲目重试。
