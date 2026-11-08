# 2026-11-08 首次成功上榜

## 认证

- GitHub OAuth 通道不可用：服务端换 token 时 GitHub 返回
  `incorrect_client_credentials`（KernelBot 的 `client_secret` 与其 `client_id` 不匹配），
  客户端无解。
- 改用 Discord OAuth 成功：`popcorn reregister discord`，浏览器授权后服务端返回
  `{"status":"success","user_name":"shane_84215","user_id":"1250473852715139204"}`。
- 最终 CLI ID：`29ece843-431a-4487-b162-d1dd59204255`，写入 `~/.popcorn.yaml`。
- 经验：长授权 URL（247/277 字符）经人工复制会被截断，导致服务端 `state` 的 base64
  解码出残缺 JSON（`Unterminated string`）。改为从 WSL 用 PowerShell 7
  `Start-Process` 直接在 Windows 默认浏览器打开，绕开复制环节。

## 提交记录

| id | 模式 | 结果 |
| --- | --- | --- |
| 969428 | test | 5/5 用例通过（size 1023–4096） |
| 969429 | benchmark | 143 ± 0.4 μs（best 135 / worst 177） |
| 969430 | leaderboard | **142.930 μs，A100 第 54 名 / 120 人** |

leaderboard 模式实际跑了 6 个 run（public 与 secret 各 test / benchmark / leaderboard），
全部 `passed: true`。

## 当前性能位置

- 带宽利用率：1.467 TB/s ÷ 2.039 TB/s = **72.0%**
- 与榜首差距：101.237 μs（WGX11）→ 本次慢 41%
- 理论下界：209,715,200 B ÷ 2.039 TB/s ≈ 102.9 μs

## 变更

- 新增 `kernels/vectorsum_v2/submission.py`（v2：块内 fp32 树 + 跨块 fp64，
  BLOCK=4096 / ITERS=8 / num_warps=8，EVEN 无 mask 快路径，`evict_first`）。
- 更新 `SCORES.md`、`NOTES.md`。

## 未完成

- 带宽利用率只有 72%，未达榜首的 ~101%。下一步做参数扫描（ITERS / num_warps / BLOCK）。
