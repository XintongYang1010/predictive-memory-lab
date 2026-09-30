# Predictive Memory Lab

首周实验仓库：在 **RealMotion 官方骨干**上检查旧预测 memory 的作用，再训练一个小型修正 controller。

**目标镜像：Python 3.11 / PyTorch 2.9.1 / CUDA 12.8，单张 A100 即可启动。**
骨干冻结；训练对象是四分类 controller（Keep / Reweight / Rebuild / Reset）。

## 最快启动

下载或 clone 本仓库，进入目录。已有目标镜像时：

```bash
bash scripts/setup.sh --av2
source .venv/bin/activate
python -m pytest -q
python -m pmr smoke --device cuda --out runs/gpu-smoke
```

最后一条不需要数据或预训练权重：它使用**合成轨迹和随机初始化的实际 RealMotion 架构**，跑完整标签生成、controller 训练、评估链路。只用于检查软件，输出不能作为论文结果。

有 AV2 原始数据（包含 `train/`、`val/`，每个场景有 parquet 和地图 JSON）后：

```bash
bash scripts/run_week1.sh /path/to/argoverse2
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

`runs/week1/`：

|文件|用途|
|---|---|
|`REPORT.md`、`metrics.csv`|预测指标和 controller 与 Keep 的配对改善|
|`summary.json`|场景级 bootstrap 95% CI；普通/高冲突分组；分更新时间的干预结果|
|`update_curves.png`|在相同绝对未来区间上比较两次更新|
|`diagnostic.jsonl`|从同一状态进行单次干预的每个分支、评分、计算耗时|
|`evaluation.jsonl`|实际部署策略逐场景结果，不含未来选最优分支|
|`round0.train.jsonl`、`round0.dev.jsonl`|修复训练样本、动作标签、tie、收益和来源|
|`round0.controller.pt`|小 controller 权重及训练/开发场景 ID|
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
|Controller|仅利用当前可见的 12 个特征选择四种修正操作|

官方代码只有上一更新的 bank，不是任意长度的多年龄 memory。原生观察更新时间为 3/4/5 秒。
在 4 秒时，Rebuild 必然与 Keep 相同，因为它重建的 3 秒状态尚无更早 context；5 秒才是非平凡重建。
结果按时间单独输出。它也不等于从当前观察构造任意新表示。

Reweight 作用于旧 bank 的读取，并不直接改变最终输出模式概率。匹配误差使用重对齐前的**绝对世界坐标**和新增观测区间。

## 数据与实验边界

- 官方 train 按场景 ID 的固定 SHA256 分成 Train-repair / Dev / Calibration；官方 val 用作一次确认评价。
- 动作和 controller 特征看不到 `target` 或 `target_mask`。未来只用于训练分支评分和评价。
- 所有候选从完整克隆状态开始，观察日程相同。删除使用局部固定随机种子。
- 训练标签使用 Energy Score；改善不超过 0.01 米时标为 Keep/tie。
- 4/5 秒的曲线都评价绝对 `[50,100)` 帧。最终标准预测指标使用 5 秒时完整 60 帧未来。
- `b_minFDE6` 与官方仓库中另一种计算顺序的 `b_minFDE6_upstream` 同时保留，详见 [实验协议](docs/PROTOCOL.md)。
- 本版本实现首周问题诊断和冻结骨干 controller 训练。**不声称已复现论文分数、验证 A100 性能、完成 SHARP/世界模拟实验，或证明 RSI 有独立收益。**

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
