# V6 执行交接（给连接 GPU 的 Codex）

先读：根目录 `AGENTS.md`（防漂移规则）→ 本文 → [V6_PREDICTIVE_MEMORY.md](V6_PREDICTIVE_MEMORY.md) → [PREREGISTRATION_V6.md](PREREGISTRATION_V6.md)。
科学依据与全部引用：`docs/V6_RESEARCH_REPORT.md`。

## 0. 当前状态（2026-10-13，作者会话）

- 分支 `v6-predictive-memory`，代码在 `pmr/predictive_memory_v6/`。
- 作者环境无法安装 torch（PyPI 被网络策略拦截）：numpy 部分（生成器、参考、指标、加载器、计数、编排、代数性质）41 个测试本地通过；
  torch 部分（全部记忆变体、训练、评测）只经 GitHub Actions 的 CPU 冒烟验证（`tests.yml` 的 `v6` job），**尚未在 GPU 上运行**。
- 没有任何 V6 GPU 结果；预注册状态为 PROPOSED。

## 1. 集群、目录与环境

- 连接、资源查询、隔离目录、归档与清理：完全沿用 [CODEX_DGX_HANDOFF.md](CODEX_DGX_HANDOFF.md) 第 1、2、4、5 节
  （SSH 别名 `dgx-127-slurm`，工作根 `/aifs4su/guhao/KVCache/WM/<run-id>`；不猜账户/分区，不冒充 Slurm 变量，不动他人作业）。
- 资源：一个节点 8×A100 或 4×H800（任选其一，记录实际型号）；CPU ≥32 核用于 L1 渲染。
- 环境：Python 3.11 + 现有 PyTorch 2.9.1/CUDA 12.8（不要重装 torch）；`python -m pip install -r requirements-v6.txt`。
- 先在登录/CPU 节点跑：`python -m unittest discover -s tests/predictive_memory_v6 -t . -v`。torch 测试若在 GPU 镜像中失败，按第 6 节处理。

## 2. 数据准备（CPU，可与 S1/S2 并行）

**nuScenes 元数据（只要表格）**：
```bash
aws s3 ls --no-sign-request s3://motional-nuscenes/          # 找到 v1.0-trainval 的 *meta* 包，名称以实际列表为准
aws s3 cp --no-sign-request s3://motional-nuscenes/<path>/v1.0-trainval_meta.tgz $CACHE/raw/
tar -xzf $CACHE/raw/v1.0-trainval_meta.tgz -C $CACHE/raw/nuscenes
```
同时把 nuScenes 使用条款/许可原文保存到 `$CACHE/raw/nuscenes/LICENSE_AS_DOWNLOADED.txt`（论文写 CC BY-NC-SA 4.0，AWS 页面写 "Commercial" 并链接条款，两者不一致，以下载时原文为准并记录）。
若 AWS 不可达，用 nuscenes.org 免费账号下载同一元数据包。不下载传感器数据。

**MOT17/MOT20 纯标签**：从 motchallenge.net 的 "Get files (no img) only" 链接下载（MOT17 约 9.7 MB；常见 URL 为 `https://motchallenge.net/data/MOT17Labels.zip`、`MOT20Labels.zip`，以页面实际链接为准），保存许可原文。

```bash
python -m pmr.predictive_memory_v6 prepare-real --cache $CACHE \
  --nuscenes-meta $CACHE/raw/nuscenes/v1.0-trainval --mot-roots $CACHE/raw/MOT17/train $CACHE/raw/MOT20/train
```
检查 `$CACHE/real/*/manifest.json` 的 `event_tracks` 与隐藏段直方图；nuScenes 自然事件 <200 时按预注册改用 R-b 伪遮挡作主指标。

**L1 Block World（约 1.4 万段 × 500 步，CPU 渲染）**：
```bash
bash scripts/setup_v6_miniworld.sh $CACHE            # clone @ b28988d，独立 venv
export PMR_MINIWORLD_PYTHON=$CACHE/miniworld-venv/bin/python
python -m pmr.predictive_memory_v6 prepare-l1 --miniworld $CACHE/Miniworld --cache $CACHE --procs 48 --splits dev
# dev 通过 check_dynamics（fold 一致率 >0.99）与 depth_check（≥0.95）后再生成 train test peekaboo backforth
```
若 fold 检查失败（墙的位置不是 1 与 15），先用 dev 数据的最小/最大块坐标确认反射点，修改 `worlds_l1.LO/HI` 并加测试；不要改动生成命令。

## 3. 阶段执行

在 Slurm 分配内（模板 `scripts/slurm_v6.sbatch`，分区/账户从实际输出填写）：

```bash
export OUT=/aifs4su/guhao/KVCache/WM/<run-id>/runs/v6 CACHE=/aifs4su/guhao/KVCache/WM/<run-id>/cache
PMR_V6_CACHE=$CACHE PMR_V6_MINIWORLD=$CACHE/Miniworld bash scripts/run_v6.sh S0 $OUT     # 所有 gate 必须通过
bash scripts/run_v6.sh S1 $OUT                                                             # ≤2 GPUh
```

S1 之后：
1. 看 `$OUT/SMOKE|S1/**/status.json`：全部变体 `overfit_final_loss < 0.5 × first_loss`；`S1/numerics` 两个 gate 通过。
2. 用 profile 作业的 `seconds/steps_done` 估算 S2 成本；与预算偏差 >2× 时先更新 PREREGISTRATION 第 5 节并告知用户。
3. **请用户确认预注册阈值**（一条消息，附 S1 摘要和第 6 节残余新意检查结果）。确认后按 PREREGISTRATION 第 1 节冻结，记录 hash。

然后依次运行（每个阶段独立分配，可续跑：已完成作业自动跳过）：
```bash
export PMR_V6_PREREG_COMMIT=<hash> PMR_V6_CACHE=$CACHE
bash scripts/run_v6.sh S2 $OUT     # 30 GPUh
bash scripts/run_v6.sh S3 $OUT     # 8 GPUh  —— S0–S3 是首个决策点（≤40 GPUh）
```
首个决策点：`python -m pmr.predictive_memory_v6 report --out $OUT`，把 `REPORT.md` 的假设表与关键数字发给用户。
若触发 H1 放弃条件（gdn_neg 或 lact 在 9T 上 ER≥0.8，或 ours 未通过 H1），**停下等用户决定**；否则继续：
```bash
bash scripts/run_v6.sh S4 $OUT; bash scripts/run_v6.sh S5 $OUT; bash scripts/run_v6.sh S6 $OUT; bash scripts/run_v6.sh S7 $OUT
bash scripts/run_v6.sh FINAL $OUT  # 只此一次打开 test split
```

预算：运行器按"阶段墙钟 × 占用 GPU 数"记账到 `$OUT/budget_ledger.json`，阶段上限来自 `prereg.BUDGET_GPUH`，累计硬上限 160 GPUh。
超限时停止发新作业、写 partial 并非零退出；不得自行提高上限。`PMR_V6_PROCS_PER_GPU`（默认 4）控制每卡并发；
H800 显存更大可试 6–8，以 S1 profile 为准。flowm_token 显存较大，batch 自动降到 32。

## 4. 结果归档

按 CODEX_DGX_HANDOFF 第 4–5 节：新建 `results/v6-<run-id>` 分支，`results/v6/<run-id>/` 下放 `REPORT.md`、`metrics.csv`、`hypotheses.json`、
`budget_ledger.json`、各阶段 `*_status.json`、所有 `job_spec.json`/`status.json`/`manifest.json`/`training.jsonl`、评测汇总 JSON 与逐 episode 文件、
`figures/`、数据 manifest 与许可原文、环境与 GPU 信息；checkpoint 与大文件走 Release 并记录 SHA256。从 GitHub 回读核验后再清理本次远端目录。

## 5. 汇报给用户的内容

实际 GPU 型号与作业号、每阶段完成/partial、GPUh 记账、H1–H7 判定表（pass/fail/missing 与关键数字、逐种子）、
最能说明问题的 3 个失败案例（从 `*_episodes.jsonl.gz` 中挑，给出 episode id 与轨迹图）、结论边界、GitHub 链接。

## 6. 出错时

- **代码 bug**（形状错误、NaN、对拍失败）：最小修复 + 新增回归测试，单独提交并在提交信息写明原因；不改科学协议、阈值或数据。
- **gate 失败但不是 bug**：作为发现记录，停下汇报，不调阈值、不换种子重试。
- **OOM**：降低 `PMR_V6_PROCS_PER_GPU`；仍 OOM 再降该变体 batch，并在报告注明。
- **集群/调度问题**：保留现场与日志，汇报具体阻塞点；PENDING 不等于在跑，提交不等于完成。
