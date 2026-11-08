# 2026-11-07 项目初始化与 vectorsum_v2 第一版

## 环境

- 新建目录结构：`kernels/`（按题组织）、`SCORES.md`、`NOTES.md`、`changelog/`。
- 安装 `popcorn-cli` v1.3.32：从 GitHub Releases 下载 `popcorn-cli-linux.tar.gz`
  （sha256 `74d6fb6d3b73bff86beb89058366261446a42dfe5df6ab2f17954015ff7697b7`），
  校验包内只有单个二进制后，安装到 `~/.local/bin/popcorn-cli` 并建立 `popcorn` 符号链接。
  未使用官方 `curl | bash` 一行安装，避免直接执行远端脚本。
- 系统无 NVIDIA GPU（`nvidia-smi` 不存在），本地无法验证 Triton kernel，只能依赖远端评测。

## 认证

- 执行 `popcorn register github`，生成本地 CLI ID 并写入 `~/.popcorn.yaml`。
- GitHub OAuth 尚未完成：需要用户在浏览器打开授权链接，服务端才会把 CLI ID 绑定到账号。
  在此之前所有需要认证的请求返回 401（`Invalid or unauthorized X-Popcorn-Cli-Id`）。

## 调研结论（写入 NOTES.md）

- 从 `gpu-mode/reference-kernels` 读取了 `problems/pmpp_v2/vectorsum_py` 的
  `task.yml`、`task.py`、`reference.py`、`submission.py`，以及 `pmpp_v2/eval.py`、`pmpp_v2/utils.py`。
- 确认容差 `rtol=1e-5 / atol=1e-8` 作用于单个标量，参考实现用 float64 累加。
- 发现官方示例 `submission.py` 把 `atomic_add` 打到 `torch.empty(1)` 未初始化内存上，不能照抄。

## 代码

- 新增 `kernels/vectorsum_v2/submission.py`：两阶段归约，块内 float64 逐元素累加 + 树形归约，
  第二内核把 float64 部分和归约为 float32 标量写回调用方传入的输出张量。
