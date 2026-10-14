# V6.1 执行交接（给连接 GPU 的 Codex）

先读：根目录 [AGENTS.md](../AGENTS.md) → 本文 → [V6_BF_DELTA.md](V6_BF_DELTA.md) → [PREREGISTRATION_V6.1.md](PREREGISTRATION_V6.1.md)。调研依据在 [research/](research/)。

## 0. 当前状态（2026-10-14，作者会话）

- **代码位置**：分支 `v6-predictive-memory`，代码在 `pmr/predictive_memory_v6/`。旧 V6.0 草案只留在 git 历史里，从未运行。
- **测试情况**：作者环境装不了 torch。numpy 部分（全部臂的代数、chunk 核、ML-II、几何、数据条件、指标、判定规则、VSC 聚合）由 `tests/predictive_memory_v6` 本地验证；torch 部分（训练、评测、探针、端到端 SMOKE）只经 GitHub Actions 的 CPU 冒烟验证，**尚未在 GPU 上运行**。
- **结果与预注册**：没有任何 V6.1 GPU 结果。预注册状态为 `approved-dev`：可直接运行 S0–S6，FINAL 需要用户确认。

## 1. 集群、目录与环境

- **集群与目录**：连接、资源查询、隔离目录、归档与清理，沿用 [CODEX_DGX_HANDOFF.md](CODEX_DGX_HANDOFF.md) 第 1、2、4、5 节。不猜账户和分区，不冒充 Slurm 变量，不动他人作业和共享环境。
- **资源**：一个节点，8×A100（40 或 80 GB）或 4×H800，记录实际型号。CPU ≥32 核用于预处理，磁盘 ≥150 GB。
- **环境**：Python 3.11，沿用镜像里的 PyTorch 2.9.1/CUDA 12.8，**不要重装 torch**。执行 `python -m pip install -r requirements-v6.txt`。
- **CPU 自检**：
  ```bash
  python -m unittest discover -s tests/predictive_memory_v6 -t . -v     # 含 torch 对拍测试
  bash scripts/run_v6.sh SMOKE /abs/runs/v6_1_smoke                     # CPU 端到端，合成数据，约 10 分钟
  ```
  SMOKE 失败就按第 7 节修，不要跳过。

## 2. 数据（CPU）

```bash
bash scripts/download_memory_maze.sh /abs/data/memory_maze 9x9    # 先列出 Drive 目录，只下 9x9
export PMR_V6_DATA=/abs/data/memory_maze                          # 下面有 train/ 与 eval/（或同义目录）
export PMR_V6_CACHE=/abs/cache/v6_1_9x9
```

- **gdown 列不出目录时**（配额或 robots 限制）：用 rclone 的 Drive remote 下载，`rclone lsf` 后 `rclone copy --include "*9x9*"`。需要集群出口放行 drive.google.com 与 *.googleusercontent.com。
- **评测集识别**：`prepare.discover` 以路径中含 `eval`/`test` 的目录为官方评测集。目录名不同就报告实际结构，**不要猜**；必要时只改 `EVAL_DIR_WORDS`，并单独提交。

## 3. S0（CPU 门槛）

```bash
bash scripts/run_v6.sh S0 /abs/runs/v6_1_<id>
```

看 `S0.json` 里的 `gates`，要求全部为 true：

| 项 | 含义 |
|---|---|
| `unit_tests` | 单测全部通过 |
| `axis_convention_fit` | 由数据确定 targets_vec 的轴约定，误差 ≤0.05 格 |
| `visibility_calibration` | FOV+视线派生的可见性与图像颜色检测一致率 ≥95% |
| `npz_len_1001`、`layout_and_targets_static`、`agent_on_corridor`、`forward_action_moves_forward` | 数据结构核对 |
| `cache_complete`、`dev_size_ok` | token cache 完整，dev ≥500 条 |

- **可见性校准不过时**：先当 bug 查（颜色检测阈值、边框裁剪、FOV、轴约定），把 `calibration.json` 里的 `visibility_calibration` 表发给用户。允许修的是**校准工具**（`maze.color_detect`），不允许改 FOV 规则来"通过"。
- **记录**：把 `np.load(f).files` 的实际键列表和每个门槛数值写进归档。

## 4. S1：测速与过拟合（≤2 GPUh）

```bash
bash scripts/run_v6.sh S1 /abs/runs/v6_1_<id>
```

- **过拟合检查**：`S1/fit_slow/L0_overfit/status.json` 与 `S1/fit_phi/overfit/status.json` 的 `overfit_ratio` 应 ≤0.5。
- **测速**：用 `seconds` 与 `S1/fit_phi/profile` 估算 S2–S4 的实际 GPUh。和预注册上限偏差超过 2 倍时，先告诉用户并给出新估计，**不要自行提高上限**。

## 5. S2 → S3 → S4（首个决策点 ≤30 GPUh）

```bash
cd /path/to/repo
export PMR_V6_PREREG_COMMIT=$(git log -1 --format=%H -- pmr/predictive_memory_v6/prereg.py)
bash scripts/run_v6.sh S2 /abs/runs/v6_1_<id>     # T0d + T0：oracle key，3 噪声水平 × 3 种子
cat /abs/runs/v6_1_<id>/report/REPORT.md          # gates.json 的 H1_T0
bash scripts/run_v6.sh S3 /abs/runs/v6_1_<id>     # T1：dk 选择、5 个 φ 种子、学习门、探针
bash scripts/run_v6.sh S4 /abs/runs/v6_1_<id>     # T2：L1、MOVE、闪烁物体
```

- **阶梯由运行器执行**：H1_T0 不过，S3 会被拒绝。这时停止 BF 这条线，按预注册写负结果报告；不要改阈值、换种子、换数据。
- **H1_T1 不过时**：只允许一轮 $d_k$ 容量诊断（看 `S3/dk_select/dk.json` 的 `rule_met` 与各 $d_k$ 增益），然后报告。
- **S4 完成后停下**，把下面这些交给用户，等确认后再进入 S5/S6：
  - `report/REPORT.md`、`report/hypotheses.json`、`report/gates.json`；
  - `budget_ledger.json`；
  - 各 `S*/eval/*/summary.json` 与 `diag.json`；
  - 失败或异常的 job.log。
- **并发**：默认每卡 4 个进程（`PMR_V6_PROCS_PER_GPU`），各臂都很小。显存吃紧时调到 2。

## 6. 用户确认后：S5（R 层）、S6（VSC 校准）、FINAL

**S5**：DINOv2 ViT-S/14 固定配置，hub 提交钉死在 `features_r.DINOV2_REPO`。需要能访问 GitHub 和 dl.fbaipublicfiles.com；不行时设 `PMR_DINOV2_CKPT` 指向本地权重。

```bash
PMR_V6_PREREG_COMMIT=... bash scripts/run_v6.sh S5 /abs/runs/v6_1_<id>
```

R 门槛（`S5/r_gate/r_gate.json`）只跑一次。不过就报告，**不做任何编码器、池化、分辨率或 PCA 维数扫描**。

**S6**：在**单独**的 Python 环境里准备 Cambrian-S，环境按 cambrian-s README 搭建，transformers 版本与 V6 环境冲突。

```bash
CAMBRIAN_S_DIR=/abs/cambrian-s HF_HOME=/abs/hf PMR_VSC_PYTHON=/abs/vsc-env/bin/python bash scripts/vsc/setup_vsc.sh
# 在 2 个视频上分别跑补丁前、补丁后（不设 PMR_VSC_DUMP），确认输出逐位一致（U12），把结果写进归档
PMR_V6_PROCS_PER_GPU=1 CAMBRIAN_S_DIR=... PMR_VSC_PYTHON=... HF_HOME=... \
  PMR_V6_PREREG_COMMIT=... bash scripts/run_v6.sh S6 /abs/runs/v6_1_<id>
```

S6 只跑一次官方 base pass 并打日志。`vsc_calib` 只读 10 个校准视频的日志，其余 40 个在 FINAL 前不看。

**FINAL**：只有在用户完成 [PREREGISTRATION_V6.1.md](PREREGISTRATION_V6.1.md) "FINAL 之前"一节的全部事项并明确同意后才能运行：

```bash
PMR_V6_FINAL_ACK=1 PMR_VSC_GT_JSON=/abs/hf/vsc_gt_10mins.json PMR_V6_PROCS_PER_GPU=1 \
  PMR_V6_PREREG_COMMIT=... bash scripts/run_v6.sh FINAL /abs/runs/v6_1_<id>
```

## 7. 出错时（AGENTS.md 第 10、11 条）

- **修代码**：先最小复现，写回归测试，单独提交，提交信息写明错在哪里。不在修 bug 的提交里改科学内容（阈值、条件、臂、指标）。
- **改协议**：任何会改变科学结论的修改都需要用户批准，并新起预注册版本。之后 `PMR_V6_PREREG_COMMIT` 指向新提交。
- **续跑**：已完成的 job（`status.json` 为 complete）会被跳过。中断后用同一命令续跑。
- **预算**：按阶段上限和 160 GPUh 硬上限执行，运行器不会自行提高上限。

## 8. 归档

- **放什么**：只放小文件，即 REPORT、hypotheses、gates、ledger、各 summary/diag、S0.json、calibration.json、job_spec。放到新分支 `results/v6_1-<date>` 的 `results/v6_1/<run-id>/` 下。
- **大文件**：rows.jsonl.gz、checkpoint 视大小走 GitHub Release，并记录 sha256。
- **核验**：只核验一次。核验记录不要多于结果提交。
