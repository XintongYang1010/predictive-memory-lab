# 运行与交接

## 环境

目标：Linux、Python 3.11、torch 2.9.1+cu128、可用 NVIDIA 驱动。
`setup.sh` 在项目 `.venv` 中复用镜像 PyTorch；版本不匹配才安装官方 cu128 wheel。
Python/CUDA 运行时或 GPU 不匹配时会明确失败，不偷偷改成 CPU 训练。
PyTorch 官方安装参考：https://pytorch.org/get-started/previous-versions/#v291

核心环境不依赖 Lightning、timm 或 NATTEN 编译。AV2 仅用于从原始 parquet/地图生成官方格式缓存。
`requirements-av2.txt` 固定 pandas/numpy/scipy/opencv/numba 等关键兼容版本。
数据只读挂载时，用 `PMR_PROCESSED_ROOT=/writable/path` 指定缓存位置。

## V2 离线重跑

先在可以访问 GitHub 的机器下载最新 `main`（或仓库页面 Code → Download ZIP），
将完整源码解压到 ParaCloud 的新目录。以下命令在**与 V1 相同的镜像内，已有 Slurm GPU allocation**时执行。
不在登录节点直接运行训练。`OLD` 来自 PR #1 的运行记录；若同学移动过目录，应改为实际位置。

一条命令完成上游准备、GPU 检查、smoke、协议测试和完整实验：

```bash
cd /path/to/predictive-memory-lab-v2
bash scripts/retest_v2.sh /share/home/s2026316084/yxt_pmr_20261002
```

输出在新的 `runs/week1_v2_<UTC时间戳>/`。可用 `PMR_V2_OUT` 固定输出位置；
用 `PMR_DATA_ROOT`、`CHECKPOINT`、`PMR_PYTHON`、`PMR_UPSTREAM` 覆盖迁移后的路径。
要断点恢复完整实验，直接执行下面的 `run_week1.sh` 命令并附加 `--resume`，不要重新运行包装脚本。
如需分步运行，命令如下：

```bash
cd /path/to/predictive-memory-lab-v2
OLD=/share/home/s2026316084/yxt_pmr_20261002
export PATH="$OLD/bin:$PATH"
export PMR_PYTHON="$OLD/.venv/bin/python"
export MPLCONFIGDIR="$PWD/runs/matplotlib-cache"
"$PMR_PYTHON" -m pmr setup --source "$OLD/external/RealMotion"
"$PMR_PYTHON" -m pmr doctor --device cuda --strict
"$PMR_PYTHON" -m pytest -q
"$PMR_PYTHON" -m pmr smoke --device cuda --out runs/gpu-smoke-v2
CHECKPOINT="$OLD/checkpoints/realmotion.pth" \
  bash scripts/run_week1.sh "$OLD/data/av2_week1_processed" --out runs/week1_v2
```

这条路径复用已安装的依赖和官方权重，V2 没有新增依赖。
从新源码目录运行使 `python -m pmr` 使用当前 V2 源码，即便旧虚拟环境的 editable install 指向旧目录。
`pmr setup --source` 只读检查旧的固定版本上游，在新目录生成适配副本。
不要在新目录覆盖安装、复制旧 controller 权重或复用旧输出目录。

在线从头安装仍用 `bash scripts/setup.sh --av2`。镜像缺 `libGL.so.1` 时，参考 V1 环境使用
`opencv-python-headless==4.11.0.86`；已经跑通的离线环境不需要重装。

输出默认为 `runs/week1_v2`。同一版本意外中断，可在相同命令末尾加 `--resume`；
要重新独立运行，用新的 `--out`。重点传回 `REPORT.md`、`summary.json`、
`round0.controller.selection.json` 和完整输出目录。`keep_fallback` 表示没有候选通过开发集门槛，不能解释成方法取得了提升。

## V2.1 训练诊断

完整实验自动运行小样本拟合，并生成 `roundN.controller.diagnostics.json`、
`roundN.controller.training.json` 和 `training_curves.png`。
`training.json` 在训练中每 50 步及首尾更新，可在作业未完成时查看。
小样本拟合失败会报告 `needs_investigation`；重复输入有冲突目标时报告 `conflicting_targets`，
不会直接宣称实现错误或自动更改训练预算。

若已有 V2 输出，可在新源码目录、原镜像内仅做 CPU 诊断：

```bash
/share/home/s2026316084/yxt_pmr_20261002/.venv/bin/python \
  -m pmr diagnose --run /path/to/existing_v2_output
```

该命令只读取 `round0.train.jsonl` 和 `round0.dev.jsonl`，不读取 Eval、不加载预测骨干，
默认保存 `round0.controller.diagnostics.json`。指定其他轮次用 `--round 1`。
它不会重新选策略，也不会把旧实验变成新校准目标下的结果；比较 V2.1 效果需要新目录完整重跑。
V1 输出不兼容，不能用公共 50 帧指标冒充最终 60 帧标签。

## 输入

原始目录：`train/<scenario>/scenario_<id>.parquet` 及同目录地图 JSON；val 同理。
处理后目录：`train/scenario_<id>.pt` 和 `val/scenario_<id>.pt`。
未标注 test split 不用于训练和评分。

官方权重来源：RealMotion README 中 remotion.pth 的公开链接。
Google Drive 配额或网络故障时可手动下载，设置 `CHECKPOINT=/path/remotion.pth`。
严格检查全部 model keys，不在加载失败后偷偷换成随机权重。
`--trust-inputs` 只应用于作者权重与自己生成的缓存，启用其 Python pickle 反序列化。

## 推荐执行顺序

1. 安装，运行 CPU 单元测试和 GPU smoke。
2. 小规模运行 profile/pilot，记录 forward 时间、显存和实际场景数。
3. 确认 Keep 与已知官方基线接近。本文默认样本子集不等于完整官方验证集；复核用 `--eval-scenes 0`（全部）。
4. 跑默认首周设置，传回整个输出目录（数据和 checkpoint 不需传回）。
5. 有可信信号再增加种子/轮数和数据。

batch=1 用于严格独立的分支状态，不使用 DDP，也不占满两张卡。
纯 PyTorch 局部 attention 可能比 NATTEN 慢；实际预算以 profile 为准。
数据文件内容哈希在读入前验证；长任务故障保存 `failure.json`，不会跳过坏场景后给出不完整总分。
中断恢复只复用完成阶段。不要对同一输出目录并发运行两个进程。

## GitHub

计划仓库：`XintongYang1010/predictive-memory-lab`，默认 private。
提交作者邮箱：`xintongyang1010@outlook.com`。
身份由 GitHub 登录决定，git author email 本身不能登录账号。

如从 ZIP 接手，已登录该账号的 GitHub CLI 可创建并推送：

```bash
bash scripts/create_github_repo.sh
```

脚本先确认登录账号，拒绝覆盖已有 origin。数据、权重、上游源码副本、日志不会上传。
私有仓库的合作者需要仓库权限，或由仓库所有者提供代码 ZIP。
