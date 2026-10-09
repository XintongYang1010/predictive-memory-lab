# Predictive Memory Lab

研究持续记忆如何保留隐藏状态、接受新证据，并避免破坏未变化对象。

**目标镜像：Python 3.11 / PyTorch 2.9.1 / CUDA 12.8，单张 A100 即可启动。**
最新实验为 **V4 可控状态记忆对照**；V1–V3 的 RealMotion 冻结骨干实验仍可独立运行。

## V4：隐藏状态、局部纠正与保留

V3 真实 AV2 复测没有可靠提升。下一轮直接检查：近期可见输入相同而早期历史不同，模型能否保持不同的隐藏状态？新证据纠正一个对象时，另一个仍不可见的对象是否继续被正确追踪？

五组对照：短窗口、8 个观测事件的 FIFO 检索、全局 GRU、逐对象状态、解析跟踪。
训练长度 24，评价长度 24/48/96，三个训练种子；训练记忆更新与读取，Dev 选择 checkpoint。
主机制比较是 T=48 下 slot 相对 GRU 的隐藏位置误差，同时检查纠正、保留和恢复失败率。
这是一轮有已知对象 ID 和速度读数的合成状态实验，**不是视觉 JEPA、AV2 提升或新方法的证明**。

在之前已分配到 A100 的工作镜像内：

```bash
git clone https://github.com/XintongYang1010/predictive-memory-lab.git predictive-memory-lab-v4
cd predictive-memory-lab-v4
bash scripts/retest_v4.sh /share/home/s2026316084/yxt_pmr_20261002
```

一条脚本完成环境检查、测试、GPU smoke、训练评估和打包。不需要 AV2、RealMotion 源码或预训练权重，不替换现有 PyTorch。
缺少依赖时用工作环境的 Python 执行 `python -m pip install -r requirements-state-memory.txt`。
默认单 GPU、三种子顺序运行、3600 秒检查式预算；超时保留 partial 并报错，不自动增加预算。
回传 `runs/state_memory_v4_*.tar.gz` 和对应 `.sha256`，其中包含全部逐场景数据、曲线和失败案例。
协议、指标含义、论文阅读及失败后的下一步见 [V4 说明](docs/STATE_MEMORY_V4.md)。

## V3：冲突变化与记忆影响度

V2.1 已在 A100 完整跑通，但三个 Dev 候选都未通过门槛，部署结果全部 Keep，
Eval ES 1.573144 与 Keep 完全相同。这不是已实现的提升。
V3 检验：加入冲突的时间变化及旧记忆对当前预测的影响，是否帮助识别值得干预的场景？
现已完成 A100 复测（[PR #3](https://github.com/XintongYang1010/predictive-memory-lab/pull/3)，Slurm 450）：
Keep ES 1.573144；temporal ES 1.581897，配对改善 -0.008753，95% CI [-0.017783, 0.000219]；base/full 回退 Keep。
这没有提供可靠部署收益，详细复核见 [验证记录](docs/VALIDATION.md)。

|控制器|可见输入|用途|
|---|---|---|
|`controller_base`|原有 14 个特征|相同训练方式的输入对照；不是 V2 权重原样复现|
|`controller_temporal`|14 + 12 个时间/方向特征|区分增长、减弱与持续的冲突|
|`controller`|全部 31 个特征|增加 Keep/Reset 输出差异和冲突×影响度|

`--ablation` 共享同一批 Keep 产生的 Train/Dev 分支，三组使用相同架构、初始化种子、
500 步训练和每组最多 3 个 Dev 候选。输入在归一化后按组屏蔽，未使用特征不能影响输出。
三组均执行每次决策 2 次额外探测；报告同时包含原始 `keep` 和相同探测开销的 `keep_probe`。
Rebuild 的额外前向单独计入总成本。**影响度是输出敏感性，不是真实收益或正确性标签。**

### 已跑通 ParaCloud 环境：下载后直接复测

在之前的 GPU 镜像内、已分配到 A100 的作业中执行：

```bash
git clone https://github.com/XintongYang1010/predictive-memory-lab.git predictive-memory-lab-v3
cd predictive-memory-lab-v3
bash scripts/retest_v3.sh /share/home/s2026316084/yxt_pmr_20261002
```

脚本自动识别 `source_852f57d/.venv/bin/python` 和该目录下的上游源码，
复用原数据及官方权重，执行环境检查、测试、GPU smoke、三组训练/评估、候选日志复核和打包。
无需重新安装依赖或下载数据。若 GPU 镜像不能访问 GitHub，在联网机器下载源码 ZIP 后解压进去，再执行最后一行。
私有仓库 clone 需要已有仓库权限。

完整结果自动保存为 `runs/week1_v3_<时间戳>_<作业号>.tar.gz`，同时产生 SHA256 文件。
传回这个结果包即可；它包含各组曲线、每个候选的权重及逐场景 Dev 记录，不含 AV2 数据或骨干权重。
`PMR_V3_OUT`、`PMR_PYTHON`、`PMR_UPSTREAM`、`PMR_DATA_ROOT`、`CHECKPOINT` 可覆盖迁移后的路径。
更多命令及中断恢复见 [运行说明](docs/OPERATIONS.md)。

V3 的 schema 已更新，必须重新收集特征，不能混用 V1/V2 日志、controller 或 `--resume` 目录。
默认仍为 Train 1000 / Dev 200 / Calibration 200 / Eval 1000、seed 42、单轮。
同一 Eval 子集已用于 V1/V2 研究判断，本轮属于探索；正式确认需要未查看的场景和额外种子。

## V1–V3 RealMotion 启动

下载或 clone 本仓库，进入目录。已有目标镜像时：

```bash
bash scripts/setup.sh --av2
source .venv/bin/activate
python -m pytest -q
python -m pmr smoke --device cuda --ablation --out runs/gpu-smoke-v3
```

最后一条不需要数据或预训练权重：它使用**合成轨迹和随机初始化的实际 RealMotion 架构**，跑完整标签生成、controller 训练、评估链路。只用于检查软件，输出不能作为论文结果。

有 AV2 原始数据（包含 `train/`、`val/`，每个场景有 parquet 和地图 JSON）后：

```bash
bash scripts/run_week1.sh /path/to/argoverse2 --ablation
```

这个命令会检查 GPU、运行测试、按需预处理数据、下载官方公开权重，然后运行首周诊断和训练。
数据目录也可以直接指向官方格式的 `realmotion_processed/`，这样会跳过预处理。
**AV2 数据需要先从 [官方数据页](https://www.argoverse.org/av2.html)取得；本仓库不包含数据，也不会自动下载整个数据集。**

已有权重时：

```bash
CHECKPOINT=/path/to/remotion.pth bash scripts/run_week1.sh /path/to/realmotion_processed
```

首跑想缩小规模：

```bash
bash scripts/run_week1.sh /path/to/realmotion_processed \
  --train-scenes 100 --dev-scenes 30 --calib-scenes 30 --eval-scenes 100 \
  --steps 100 --out runs/pilot
```

默认抽取上限：Train-repair 1,000 场景、Dev 200、Calibration 200、Evaluation 1,000。
一个场景有两个原生更新事件，因此最多得到约 2,000 个训练事件。
文件少于上限时会使用实际可用数量，准确数量记录在 `manifest.json`。
原始数据默认最多预处理 6,000 个 train、1,000 个 val 场景；增加范围用 `scripts/preprocess.py`。

## 输出看哪里

`runs/week1_v3/`：

|文件|用途|
|---|---|
|`REPORT.md`、`metrics.csv`|预测指标和 controller 与 Keep 的配对改善|
|`summary.json`|场景级 bootstrap 95% CI；普通/高冲突分组；分更新时间的干预结果|
|`update_curves.png`|在相同绝对未来区间上比较两次更新|
|`diagnostic.jsonl`|从同一状态进行单次干预的每个分支、评分、计算耗时|
|`evaluation.jsonl`|实际部署策略逐场景结果，不含未来选最优分支|
|`round0.train.jsonl`、`round0.dev.jsonl`|31 维因果特征、各动作最终 ES、诊断标签和来源|
|`round0.controller.pt`|小 controller 权重及训练/开发场景 ID|
|`round0.controller.selection.json`|候选策略在 Dev 上完整运行的 ES、动作分布、是否通过门槛及最终选择|
|`round0.controller.training.json`|收益回归 MSE、训练时间和候选预算|
|`round0.controller.candidates/`|Keep 及每个候选的逐场景 Dev 结果、候选权重、权重哈希|
|`ablations/base/`、`ablations/temporal/`|另外两组的完整训练、选择和曲线|
|`selection_audit.json`|从候选逐场景记录重算均值、CI、分组和最终选择的结果|
|`baseline_adapter.json`|Keep 包装层与无 hook 的固定版本模型的输出对齐检查；不等于官方 NATTEN/论文分数复现|
|`manifest.json`、`checkpoint.json`|数据路径、场景划分、内容哈希、骨干权重哈希|
|`correctness.json`、`environment.json`、`run_stats.json`|一致性检查、运行环境、前向次数、耗时和显存|

先看 Keep 是否合理，再看 `drop_worst` 是否优于数量匹配的 `random_drop`，最后看 controller 是否超过校准规则。不要仅凭训练集 oracle repair 的收益决定继续项目。

## 干预的准确含义

|名称|实现|
|---|---|
|Keep|保留官方 memory 路径|
|Reweight|新到达观察与旧模式计算残差；把 `log prior - MSE/(2 sigma²)` 作为 trajectory attention 的加性偏置|
|Rebuild|重新编码**同样保留的上一帧观察窗口**，去掉其历史 context；仅替换上一帧预测 bank，当前 scene context 保持原值|
|Reset|当前更新跳过 trajectory interaction/correction，写回当前新预测；保留 scene context|
|Drop-worst|屏蔽残差最大的 2 个模式，仍保留 4 个|
|Random-drop|同一 bank 随机屏蔽 2 个模式；数量、年龄与 Drop-worst 匹配|
|Decay|按 bank 年龄衰减 trajectory correction，时间常数仅在 Calibration 选择|
|Reset-all|清空整个前一帧 memory，作为额外诊断|
|Threshold|超过校准残差阈值时 Reset，否则 Keep|
|Controller|按指定输入组预测动作收益；收益不足、无新增证据或开发集筛选失败时保留历史|

官方代码只有上一更新的 bank，不是任意长度的多年龄 memory。原生观察更新时间为 3/4/5 秒。
在 4 秒时，Rebuild 必然与 Keep 相同，因为它重建的 3 秒状态尚无更早 context；5 秒才是非平凡重建。
结果按时间单独输出。它也不等于从当前观察构造任意新表示。

Reweight 作用于旧 bank 的读取，并不直接改变最终输出模式概率。匹配误差使用重对齐前的**绝对世界坐标**和新增观测区间。

## 数据与实验边界

- 官方 train 按场景 ID 的固定 SHA256 分成 Train-repair / Dev / Calibration；官方 val 用作评估，已查看过的子集只报告探索结果。
- 动作和 controller 特征看不到 `target` 或 `target_mask`。未来只用于训练分支评分和评价。
- 所有候选从完整克隆状态开始，观察日程相同。删除使用局部固定随机种子。
- V3 延续 V2 训练目标，为每个动作相对 Keep 的最终 60 帧 ES 改善，使用 MSE，同时监督正、负收益。
- `label/weight/tie` 保留为公共窗口的 V1 诊断字段，不再作为控制器训练目标。
- Dev 中先按分支收益粗筛最多 3 个候选，再运行真实连续决策；仅当配对收益 CI 下界大于 0 且普通组平均收益非负时允许干预，否则回退 Keep。
- 开发集筛选的 CI 是调参门槛，不是筛选后显著性结论。V1 已看过的 val 子集重跑属于探索分析；正式确认应使用未看过的场景和额外种子。
- 4/5 秒的曲线都评价绝对 `[50,100)` 帧。最终标准预测指标使用 5 秒时完整 60 帧未来。
- `b_minFDE6` 与官方仓库中另一种计算顺序的 `b_minFDE6_upstream` 同时保留，详见 [实验协议](docs/PROTOCOL.md)。
- V1–V3 实现首周问题诊断和冻结骨干 controller 训练。三版均已在 A100 跑通；**V2.1 与 V3 未取得可靠部署收益。不声称已复现论文分数、完成 SHARP/世界模拟实验或证明 RSI 有独立收益。**

## 进一步命令

测实际设备耗时和显存：

```bash
python -m pmr profile --data-root /path/to/realmotion_processed \
  --checkpoint checkpoints/realmotion.pth --trust-inputs --device cuda
```

多轮重新采样、验证和训练（探索功能，默认只做一轮）：

```bash
bash scripts/run_week1.sh /path/to/realmotion_processed --rounds 3 --out runs/refresh3
```

中断后重用已完成阶段：在**完全相同命令**后加 `--resume`。未完成轮次重新开始；不是 optimizer-step 级断点恢复。
默认采用一个 GPU。第二张 GPU 可以用不同输出目录、不同 seed 独立复跑：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run_week1.sh /path/to/realmotion_processed \
  --seed 43 --out runs/week1_seed43
```

CPU 开发环境：`bash scripts/setup.sh --cpu`，然后 `python -m pmr smoke`。

## 实现与来源

安装脚本拉取 [RealMotion](https://github.com/fudan-zvg/RealMotion) 的固定 commit
`e292609f03ccd415c4f2ac6dd7fd907c098508e8`，在忽略目录 `external/portable/` 建立本地适配副本。
适配清单记录原文件和修改后的哈希；不会改写上游 checkout。

1. 将 NATTEN / timm 的两个层导入替换为 `pmr.compat` 的纯 PyTorch 实现，无需自定义 CUDA kernel。
2. 修正没有其他车辆时上游 `view(..., -1, 2)` 的歧义；其余形状下计算不变。

局部 attention 的前向和梯度已与独立 dense reference 比对。正式研究前仍需在目标机器上核对官方 checkpoint 的基线分数；纯 PyTorch backend 的速度不能当作 NATTEN 性能。

上游代码与权重在安装时从作者来源取得，不随本仓库重新分发。参考 [依赖与运行说明](docs/OPERATIONS.md) 和 [验证记录](docs/VALIDATION.md)。
