# 运行与交接

## 环境

目标：Linux、Python 3.11、torch 2.9.1+cu128、可用 NVIDIA 驱动。
`setup.sh` 在项目 `.venv` 中复用镜像 PyTorch；版本不匹配才安装官方 cu128 wheel。
Python/CUDA 运行时或 GPU 不匹配时会明确失败，不偷偷改成 CPU 训练。
PyTorch 官方安装参考：https://pytorch.org/get-started/previous-versions/#v291

核心环境不依赖 Lightning、timm 或 NATTEN 编译。AV2 仅用于从原始 parquet/地图生成官方格式缓存。
`requirements-av2.txt` 固定 pandas/numpy/scipy/opencv/numba 等关键兼容版本。
数据只读挂载时，用 `PMR_PROCESSED_ROOT=/writable/path` 指定缓存位置。

## V3 离线复测

在与已完成 V2.1 作业相同的 Python/CUDA 镜像内，已有 Slurm GPU allocation 时执行。
仓库源码需放在新目录，旧目录只用于读取环境、数据、权重和固定版本上游。

```bash
cd /path/to/predictive-memory-lab-v3
bash scripts/retest_v3.sh /share/home/s2026316084/yxt_pmr_20261002
```

自动按顺序完成：准备上游副本 → 严格检查 Python 3.11/torch 2.9.1/cu128 → 测试 →
GPU 三组 smoke → 真实数据三组训练/评估 → 候选逐场景复核 → 自动打包。
运行中持续显示进度；控制器日志每 50 步及首尾更新。新增特征不引入依赖。

PR #2 核实的 Python 为 `source_852f57d/.venv/bin/python`，上游为
`source_852f57d/external/RealMotion`。脚本优先识别这两个位置，兼容旧的根目录布局。
路径变化时可显式设置：

```bash
PMR_PYTHON=/path/to/working/python \
PMR_UPSTREAM=/path/to/pinned/RealMotion \
PMR_DATA_ROOT=/path/to/processed \
CHECKPOINT=/path/to/realmotion.pth \
  bash scripts/retest_v3.sh /path/to/old-root
```

默认使用 seed 42、500 步、每组 3 个 Dev 候选及 1000/200/200/1000 场景。
缩小数据规模的功能试跑：

```bash
bash scripts/retest_v3.sh /share/home/s2026316084/yxt_pmr_20261002 \
  --train-scenes 100 --dev-scenes 30 --calib-scenes 30 --eval-scenes 100 --steps 100
```

三组均在相同 Keep 状态上收集标签，所有模型在 Eval 之前由 Dev 选择并固定。
为保持对照可比，包装脚本固定 `--ablation --rounds 1`，不接受覆盖 `--out`/`--feature-set`。
用 `PMR_V3_OUT` 指定全新的输出位置；包装脚本不覆盖旧结果。

成功时打印 `RETEST_V3_COMPLETE` 和压缩包路径。输出包括主实验目录、同名 `.log`、
`.tar.gz`、`.tar.gz.sha256`，smoke 在单独的 `.smoke` 目录。
结果包包含主实验所有日志、三组候选的权重、逐场景数据和文件校验清单，
不包含 AV2、官方骨干权重和合成数据。将结果包传回即可。

## 分步运行与恢复

```bash
PMR_OLD=/share/home/s2026316084/yxt_pmr_20261002
PMR_PYTHON="$PMR_OLD/source_852f57d/.venv/bin/python"
export PATH="$PMR_OLD/bin:$PATH"
"$PMR_PYTHON" -m pmr setup --source "$PMR_OLD/source_852f57d/external/RealMotion"
"$PMR_PYTHON" -m pmr run --device cuda --trust-inputs \
  --data-root "$PMR_OLD/data/av2_week1_processed" \
  --checkpoint "$PMR_OLD/checkpoints/realmotion.pth" \
  --ablation --rounds 1 --out runs/week1_v3
```

中断后对完全相同的 `pmr run` 命令加 `--resume`。只重用完成的轮次，未完成轮次重新运行；
不是 optimizer step 级恢复。`settings.json` 记录精确参数和代码指纹，不能跨版本绕过检查。
不要对同一输出目录同时运行多个进程。

```bash
"$PMR_PYTHON" -m pmr audit-selection --run runs/week1_v3
"$PMR_PYTHON" scripts/package_results.py --run runs/week1_v3
```

`audit-selection` 不运行骨干，从每个候选的逐场景 Dev 记录重算配对均值、95% CI、普通组损伤、
动作数量和最终选择，并验证候选权重哈希。旧 V1/V2 包缺少这些记录，不兼容此复核命令。
如果门槛未通过，报告显示 `keep_fallback`，不能解释成学到了改进。

已有 V3 输出可单独诊断 Train/Dev：

```bash
"$PMR_PYTHON" -m pmr diagnose --run runs/week1_v3 --feature-set temporal_influence
```

仅做一组实验可省去 `--ablation`，用 `--feature-set base|temporal|temporal_influence` 指定输入。
多轮 `--rounds` 只用于单组探索，不能与共享 Keep 标签的三组对照同时使用。

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
