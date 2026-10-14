# 从这里开始：V6

当前主线是 V6（预测式联想记忆，对象 token，无感知）。按顺序读：
[AGENTS.md](AGENTS.md) → [docs/CODEX_V6_HANDOFF.md](docs/CODEX_V6_HANDOFF.md) →
[docs/V6_PREDICTIVE_MEMORY.md](docs/V6_PREDICTIVE_MEMORY.md) → [docs/PREREGISTRATION_V6.md](docs/PREREGISTRATION_V6.md)。
V6 尚无 GPU 结果；预注册阈值待用户确认后冻结。

---

# V5 first run: local Codex to DGX (historical)

本仓库包含 V5 首轮 visual world model + bounded memory 的源码和 CPU 测试。
尚未运行 V5 A100 实验，没有真实 GPU 结果。

当前任务通过 Mac 本地已有的 SSH 别名 `dgx-127-slurm` 连接服务器，
在 `/aifs4su/guhao/KVCache/WM` 中运行，将结果上传回本仓库，
从 GitHub 重新读取并核验归档完整后，再删除本次新建的远端目录。

1. 在可使用该 SSH 别名的 Mac 本地 Codex 中拉取本仓库的最新 `main`。
2. 阅读并执行 [DGX 完整交接任务](docs/CODEX_DGX_HANDOFF.md)。
3. 查询真实 Slurm 账户、分区、A100、Python/CUDA 环境和存储情况。
4. **先适配平台**：`scripts/launch_visual_v5.py` 与
   `pmr/visual_memory/__main__.py` 仍绑定原 NYU 账户；不能直接照搬到 DGX。
5. 在独立环境准备官方 I-JEPA，CPU 检查和实际 A100 profile 通过后，
   按依赖运行完整首轮，首轮累计上限 60 GPUh。

实验协议及原 NYU 后端说明见 [VISUAL_MEMORY_V5.md](docs/VISUAL_MEMORY_V5.md)。
原 NYU 课程的 Cloud Bursting 限制属于 NYU 后端，不作为 DGX 账户配置。
DGX 的实际资源名称、权限和认证材料不在本仓库中假设或保存。

CPU 软件检查：

```bash
python -m unittest discover -s tests/visual_memory -v
```

源码已经作为普通仓库文件提供；无需另外获取聊天中的 ZIP。
