# NOTES — 假设、已证伪方向与待验证项

## 题目事实（已从官方仓库一手核实）

- leaderboard 名 `vectorsum_v2`，目录 `problems/pmpp_v2/vectorsum_py`。
- 可用卡型：B200 / H100 / A100 / L4（`problems/pmpp_v2.yaml`）。
- 输入：`(N,)` float32，`scale * randn + offset`，`offset ∈ [-100, 100]`，`scale ∈ [0.1, 10]`。
  第二个 `output` 张量是 `torch.empty(1)`，**未初始化**。
- 参考：`data.to(torch.float64).sum().to(torch.float32)`。
- 容差：`rtol=1e-5, atol=1e-8`，作用于**单个标量**。
- 输出契约：`type(x) is torch.Tensor`（非子类）、shape `()`、dtype float32、device cuda。
- benchmark 尺寸：1,638,400 / 3,276,800 / 6,553,600 / 13,107,200 / 26,214,400 / 52,428,800。
- **计分只取最后一个 case（52,428,800，200 MiB）**：kernelbot `RankCriterion.LAST`，
  `run_eval.py` 只把 benchmarks 最后一行写入 case 文件。小 case 只管正确性。
- 计时用 CUDA event 包住整段 `custom_kernel(data)`，但计时前有 `clear_l2_cache()`
  （268 MB 的 `fill_`），在四张卡上耗时 34~894 μs，**远大于 5–10 μs 的 Triton dispatch，
  Python launch 开销被它掩盖**。
- `leaderboard` 模式 `recheck=True`：每轮 `generate_input` 重建输入且 `seed += 13`；
  `benchmark` 模式复用同一 `data`。

## 已证伪的方向（前一版结论错误，勿再犯）

- ~~“必须全程 float64 累加”~~ **错**。官方 `solutions/correct/pytorch.py` 就是
  `output[...] = data.sum()`（纯 fp32）。按同分布 CPU 模拟 N=5.24e7，任何分块树形归约
  余量 ≥23x；只有**单线程顺序 fp32 累加**会失败（超容差 3.6~5.9 万倍）。
- ~~“B200 的 FP64 被砍到 1.2 TFLOPS，全 fp64 会 compute-bound”~~ **错**。1.2 TFLOPS / 1/64
  对应的是 Blackwell Ultra（B300/GB300）。B200 是 CC 10.0，FP32:FP64 = 2:1，FP64 ≈37 TFLOPS。
- ~~“官方示例可以直接跑”~~ **错**，两个坑：
  1. 模板首行是 `#!POPCORN leaderboard vectorsum_py`（旧榜名），与现役榜 `vectorsum_v2`
     不一致时 kernelbot 直接抛错。
  2. `atomic_add` 打到 `torch.empty(1)` 未初始化内存上。
- ~~“必须用 atomic 累加”~~ **错**。atomic 的正确定位是计数器或少量分槽，赢面来自单次
  kernel 启动 + 末块收尾，不是 atomic 本身。默认模板“每 block atomic 到同一地址”在 B200
  只有第 83 名（292 μs）。
- ~~“需要 CUDA Graph / torch.compile 压 launch 开销”~~ **错**。launch 开销已被
  `clear_l2_cache` 掩盖；且 leaderboard 每轮换输入指针，graph 要拷贝 200 MiB，不划算。

## 精度：真正的危险在哪

- 危险的是**跨块用 fp32 顺序累加**（含 atomic 到单一 fp32 地址）：模拟最坏情形
  （offset=-100、scale=0.1）余量仅 **0.076x，会失败**。
- 安全的是**块内 fp32 树形 + 跨块 fp64**：最坏情形余量 ≥**62x**；其余情形精确到输出 ULP。
- 容差实际由 rtol 主导：`|expected| ≈ N·|offset|` 常达 1e3~5e9，对应容差 0.01~5e4。
  `atol=1e-8` 只在总和恰好接近 0 时才起作用（概率极低）。

## 现役榜（2026-11-08 读取 https://www.gpumode.com/leaderboard/544）

| 卡 | 第 1 名 | 上榜人数 | 备注 |
| --- | --- | --- | --- |
| B200 | 37.033 μs | 83 | 前 20 名全在 40.993 μs 内，最卷 |
| H100 | 67.922 μs | 69 | 达到榜首 85% 带宽也只到第 20 名 |
| A100 | 101.237 μs | 119 | 第 2 名 116.890、第 3 名 125.409，头部稀 |
| L4 | 692.015 μs | 45 | 第 1 与第 2（816.290）差 124 μs，断档 |

**选卡结论：首选 A100，次选 L4，首战不用 B200/H100。**
- A100：达到榜首带宽 85% 即 rank 3，90% 即 rank 2；前 20 名只需 75% 带宽。
- L4：45 人，达 85% 即 rank 2；但 FP64 只有 fp32 的 1/64，不能用逐元素 fp64。

## 参数扫描结果（A100，benchmark 模式）

| 配置 | per-program | μs | 备注 |
| --- | --- | --- | --- |
| B4096 I8 W8（首战） | 32768 | 143 ± 0.4 | baseline |
| **B4096 I4 W8** | 16384 | **141 ± 0.2** | 当前最优 |
| B4096 I8 W16 | 32768 | 143 ± 0.1 | |
| B4096 I16 W8 | 65536 | 148 ± 0.6 | per-program 过大 |
| B8192 I4 W8 | 32768 | 143 ± 0.6 | |

结论：per-program = 16384 略优，但整体只有 1.4% 的提升，**参数调优已饱和**。

## PTX 诊断（本地离线编译）

- Triton 3.8 对这个归约形态生成的是**标量 `ld.global.b32`**，不是 128-bit `v4.b32`。
  纯拷贝 kernel、2D 连续索引、`tl.max_contiguous` / `tl.multiple_of` 提示、
  `tl.make_block_ptr` 全部试过，均无法改变。
- 但重新核算后这**不是主要瓶颈**：GPU 是 SIMT，一条 `ld.global.b32` 由整个 warp 执行、
  每指令搬 128 字节，指令发射量并非瓶颈。
- 当前 141 μs = 1.487 TB/s = A100 标称 2.039 TB/s 的 **72.9%**；榜首 101.237 μs 对应
  2.072 TB/s（超标称 1.6%），说明标称偏保守、极限确实在 2.0 TB/s 以上。

## 提交限流（重要）

- `test` / `benchmark` 模式共享 **6 次/小时** 的配额，超出返回
  `429 Too Many Requests: 6/6 test submissions per hour`。
- 首战 3 次 + 第一轮扫描 4 次 = 7 次，已触发限流，需等约 46 分钟。
- 后续调参必须省着用：一次只验证一个最优候选，不要批量扫。

## 待验证

1. 更小的 per-program（8192 / 4096，对应 m=6400 / 12800）能否改善负载均衡与
   tail effect——受限于流未能测到。
2. 降低每线程元素数（BLOCK=1024/2048）是否因寄存器压力下降而提升 occupancy。
3. 两阶段 vs 单 kernel + atomic counter 的差距（预期 ≤5%）。
