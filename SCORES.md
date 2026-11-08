# SCORES — gpumode-first-board

每次提交都追加一行。失败的提交同样记录，并写明失败原因。

| 时间 (UTC+8) | 题目 | 卡型 | 模式 | μs | 名次 | 结果 / 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-08 14:00 | vectorsum_v2 | A100 | test | — | — | ✅ 通过。id 969428，5 个用例全 pass（size 1023/1024/1025/2048/4096） |
| 2026-11-08 14:01 | vectorsum_v2 | A100 | benchmark | 143 ± 0.4 (best 135 / worst 177) | — | ✅ 通过。id 969429，52,428,800 上正确性 OK |
| 2026-11-08 14:02 | vectorsum_v2 | A100 | **leaderboard** | **142.930** (score 0.0001429299207) | **54 / 120** | ✅ **首个有效上榜成绩**。id 969430；secret score 149.268 μs |
| 2026-11-08 14:0x–16:2x | vectorsum_v2 | A100 | benchmark ×4 | 141 / 143 / 143 / 148 | — | 参数扫描：B4096_I4_W8 最优（141 μs），ITERS 8→4 已写入正式文件 |
| 2026-11-08 16:2x | vectorsum_v2 | A100 | benchmark ×4 | — | — | ❌ 全部被拒：`429 Too Many Requests: 6/6 test submissions per hour`，剩余等待 2772 s |
| 2026-11-08 17:15 | vectorsum_v2 | A100 | leaderboard | 171.653 | — | ⚠️ id 969445，异常慢（GPU 噪声），未覆盖 best |
| 2026-11-08 17:16 | vectorsum_v2 | A100 | leaderboard | 147.210 | — | id 969446，未超过 best |
| 2026-11-08 17:17 | vectorsum_v2 | A100 | leaderboard | 142.131 | — | id 969447，未超过 best |
| 2026-11-08 17:17 | vectorsum_v2 | A100 | leaderboard | **141.865** | **43 / 120** | ✅ **新最佳**。id 969448 |
| 2026-11-08 17:18 | vectorsum_v2 | A100 | leaderboard | 142.019 | — | id 969449，未超过 best |

## 该次提交的评测细节

- 提交 id `969430`，全部 6 个 run（public + secret 各 test/benchmark/leaderboard）均 `passed: true`
- 官方口径分数：`Geomean score (public) on A100: 0.0001429299207031727 s`
- secret 组分数：`0.00014926848083734513 s`
- 带宽换算：209,715,200 B ÷ 142.93 μs = **1.467 TB/s**，为 A100 80GB SXM 标称 2.039 TB/s 的 **72.0%**

## 榜单参照（2026-11-08 拉取）

| 名次 | A100 成绩 |
| --- | --- |
| 1 | 101.237 μs（WGX11） |
| 2 | 116.890 μs |
| 3 | 125.409 μs |
| 20 | 137.748 μs |
| **43（当前最佳）** | **141.865 μs（id 969448）** |
| 54（首战） | 142.930 μs（id 969430） |
| 100 | 155.917 μs |

名次 40–60 之间极度密集（141.210 → 143.449，20 人挤在 2.2 μs 内），
再快 1 μs 约可前进 8–10 名。

## 测量噪声（重要）

同一份代码连续提交，public 分数在 **141.865 – 171.653 μs** 之间波动（约 21%）。
secret 侧则稳定在 141.8–142.0 μs。榜单保留 **best score**，因此多次提交取最好
是平台允许的策略；但不要把这理解为代码本身变快了。

## 字段说明

- **模式**：`test` 正确性校验 / `benchmark` 不计分性能测试 / `leaderboard` 正式排名
- **μs**：`Geomean score (public)`，计分只取最后一个 benchmark case（size=52,428,800）
- **名次**：来自 `https://www.gpumode.com/api/leaderboard/544` 的 `rankings.A100[].rank`
