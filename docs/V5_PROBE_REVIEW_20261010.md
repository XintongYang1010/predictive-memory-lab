# V5 最终设计审查与时间证据确认协议

日期：2026-10-10 UTC / 2026-10-09 America/New_York。本文结合三次咨询及最新 `ca365e1` 的 matched temporal controls；不是旧 `df9e780` 阶段的执行状态说明。

## 结论

**当前真正卡点是运动前置判据的效度，而不是连接、Slurm、GPU 型号或一个已定位的标签错误。** 固定 ridge 正则化确实影响原读出器的表现；但同预算单帧对照说明 pooled 的正位移分数尚不能证明使用了时间证据。原 V5 gate 保持失败，不能用新的 alpha 补算通过，也没有证据开始 memory 架构比较。

作为原方案设计方，我应收紧原门槛的解释：`position > .1 && displacement > 0` 只是特定读出器的启发式筛查，不是“表征含/不含运动信息”的充分判据。**失败不证明信息不存在；成功也不排除单帧位置先验。** 本次不修补冻结科学代码，也不把直接帧差作为已证实的修复。

## 1. 证据边界与可证事实

已通过授权连接读取私有 GitHub 的冻结源码及下列固定提交。没有操作 SSH、远端目录、Slurm 或 GPU。

- 科学源码：[d0b4fef](https://github.com/XintongYang1010/predictive-memory-lab/tree/d0b4fef25e98eefb58bfba737997cb0145f27daa)。
- 原 H800 实跑及失败：[d4b6fc6](https://github.com/XintongYang1010/predictive-memory-lab/tree/d4b6fc622d0fe08e28605aaf6e48100201ab2008/results/wm/visual-v5-dgx-20261009)。无 P0、memory checkpoint 或 Test。
- 远端留存及 GitHub 回读记录：[d7035c8](https://github.com/XintongYang1010/predictive-memory-lab/blob/d7035c89b88d58342fb3cea68de3c59d3256806e/results/wm/visual-v5-dgx-20261009-verification.json)。
- 独立 CPU 核心诊断：[8b2ee24](https://github.com/XintongYang1010/predictive-memory-lab/tree/8b2ee24d1beabf69efa4f9b2329fdf43963f3883/results/wm/visual-v5-probe-diagnostic-20261009)。
- **最新 matched controls 与复核：[ca365e1](https://github.com/XintongYang1010/predictive-memory-lab/tree/ca365e1aa1026c51be0d23008b63650354c5bf31/results/wm/visual-v5-probe-diagnostic-20261009)。** 本文最终判断以此为准。

原特征张量未放入 GitHub 归档；我没有在本环境重跑这些缓存 fits。补充诊断到达前，我已独立核对原 CSV、256 个 Train/Dev episode 的目标/时间/反射及 RGB oracle；其可复核结果在 [archive_audit.json](../results/wm/visual-v5-probe-review-20261010/archive_audit.json)。收到新诊断后没有重复其计算。这里区分“审阅已提交的复核记录”与“自己重跑了特征拟合”。

| 证据 | 支持的结论 | 不支持的结论 |
|---|---|---|
| 原 pooled 位移 −0.0209268；Train-only CV 选 alpha=1 后 0.2438953 | 原固定正则化不适合该小样本读出目标；选参流程可改进 | 原 gate 已通过；运动信息已确认 |
| pooled 单帧 t7=0.2580565；双帧减它 −0.014161，95% 区间 [−0.034358, 0.004930] | 此 matched 控制没有建立双帧优势 | 两帧绝无信息；两模型严格等效 |
| raw 双帧=0.2334773，比 Train 选择的 t6 高 0.034022；但报告的 t7=0.2357446 | 对预先选定 t6 的比较成立，必须同时保留 t7 结果 | 两帧必要；事后改选 t7 作为新的正式检验 |
| 交换 query 顺序后 raw/pooled≈−1.022/−1.081，符号反转率约 7.42%/6.45% | 读出器不满足理想位移映射的顺序一致性；值得调查 | 单独定位 pooling 或证明某个因果机制；该操作同时改变了输入分布 |
| pooled 帧差固定 .01=−0.282749，CV 选 10≈0.010472；共享位置解码器≈−1.861991 | “改成帧差/位置相减就会好”被本轮证据反驳 | 任何非线性/更多样本读出都不可能成功 |
| raw 重新池化仍约 −0.020736/−0.020055 | 所测 batch/存储量化差异不能解释本次主要失败 | 任意精度或硬件因素均已全面排除 |
| 单帧真位置≈0.314991、t7 RGB 质心≈0.314306；两帧质心≈0.977881 | 生成器存在很强的位置先验；像素位移可观测 | 特权真位置或颜色 oracle 可以进入 actor；I-JEPA 已保留同等信息 |

[独立 Torch 验证](https://github.com/XintongYang1010/predictive-memory-lab/blob/ca365e1aa1026c51be0d23008b63650354c5bf31/results/wm/visual-v5-probe-diagnostic-20261009/validation/torch_reference.json)具体覆盖 pooled concat 的 6 个 alpha × 4 个 fold，最大差 1.17e−12；不是声称所有 22 fits 都被第二套 Torch 实现逐项重算。[temporal_review.json](https://github.com/XintongYang1010/predictive-memory-lab/blob/ca365e1aa1026c51be0d23008b63650354c5bf31/results/wm/visual-v5-probe-diagnostic-20261009/validation/temporal_review.json)记录逐维 CSV 复核最大差 2.66e−15。这些记录进一步降低了数值实现错误的可能性。

## 2. 是否遗漏了实现错误

**在已审查路径内，没有发现足以解释失败的新实现错误。** 这不是对全部项目代码的无条件担保。

| 检查 | 结论 |
|---|---|
| 时间帧 | 默认协议的早期时间 0…7 位于所有额外观测之前；`features[6:8]` 与 `raw_cue` 对应可见 t6/t7。256 个原 Train/Dev episode 已核对。 |
| 标签与反射 | 标签是 `p7` 及反射后的 `p7−p6`。独立折返计算与冻结生成器、CSV 一致；不是隐藏 velocity。水平轴 Dev 中 A/B 各仅 1/128 在该时间间隔反射，主要错误发生在非反射样本，不能归咎于反射。 |
| 池化 | 已核对固定版本官方 PatchEmbed 的行优先空间展平及冻结源码的 transpose/reshape/pool；未发现 H/W 或通道错置。原始 16×16 patch tokens 变为 4×4，不是平均两个时间步。 |
| 配对/划分 | split 进入种子；两个 variant 留在同一 world，worker 按 pair 分片。原 64+64 world 未发现种子重合。独立样本单位是 world，不是把 128 episode 当作 128 world。 |
| 拟合 | 中心化、归一化和监督拟合只用 Train；补充 CV 在每个训练 fold 内重新计算统计量，未见 Dev 选 alpha 的路径。 |
| 命名/分数 | `velocity_r2` 是遗留名称，实际是观测位移；分母是 Dev 相对 **Train 均值**的 MSE，是预测 skill，而非通常的 Dev-mean R²。应在新输出中准确命名，保留旧字段以避免篡改归档。 |
| encoder 数值配对 | 原 raw 是另一次 batch=2 推理，pooled 来自较大 batch 且量化顺序不同，因此原比较并非纯池化干预。重新池化结果已把它从主要解释中降级。 |

Dev-mean 分母不大于 Train-mean 分母，因此换成通常的 R² 只会使对应分数不升反降，不能修复负 gate。

另一个应修正的是文字解释：[现有 static_review.md](https://github.com/XintongYang1010/predictive-memory-lab/blob/ca365e1aa1026c51be0d23008b63650354c5bf31/results/wm/visual-v5-probe-diagnostic-20261009/static_review.md)中固定 alpha 下“两帧改善”的段落，不能继续被读成已建立时间依赖。最新 matched-CV 结果优先于这一较早解释。这里指出差别，不覆盖该诊断分支。

## 3. 探针局限：已知数学性质与仍待验证的假设

确定的性质：

1. 多输出 ridge 的 8 个输出列独立求解。位置 target 量纲大不会抢占位移的拟合能力。单纯缩放标签再还原，也不会改变固定惩罚下的预测 skill。
2. 全局 feature-energy 归一化消除了统一缩放 X 的效果，但没有白化方向、解耦两个时间步或控制 nuisance。可能的问题在输入几何与惩罚，而不是目标列竞争。
3. 令 `s=(z6+z7)/sqrt(2)`、`d=(z7−z6)/sqrt(2)`。从 concat 改成 `[s,d]` 是正交变换，核、归一化和同一 ridge 预测不变。仅改命名没有修复作用。difference-only 或重加权改变的是归纳偏置，现有 difference-only 结果并不有利。
4. 128 条拟合记录只有 64 个 world；中心化后的核秩最多 127。高维输入不等于 128 条独立轨迹，更不意味着固定 lambda 已有统计保障。
5. 原求解是 `K+lambda I`，不是 `K/n+lambda I`。学习曲线中固定该 lambda 会改变平均损失意义上的正则强度。新协议明确采用 `K+n*alpha I`，不能把两种 alpha 数字混写。
6. pooled 位置剩余 RMSE 约 7.35/3.44/5.95/3.00 像素，而位移基线 RMS 约 2.6–3.0 像素。位置 R²≈0.975 并不意味着足以稳定恢复帧间小差值。

仍待检验：4×4 表征是否保留足够精细、可迁移的两帧关系；读出非线性是否必要；有限 world 数、纹理/位置 nuisance 与池化分别贡献多少。现有证据不足以给这些因素分配因果比例。尤其不能从“更大 alpha 得到正数”推导已经读取运动，也不能把 pooled 的失败定性为 encoder 丢光了运动信息。

现有缓存的原分数复现、正则化、样本数、重池化、帧差、共享解码器、同预算单帧、顺序干预已足够形成这次判断。**不建议再在同一 Dev 上扩大搜索来找通过者。**

## 4. 新确认协议：同时固定终点与起点

目的不是禁止位置表征。正确解码两帧位置再作差，本来就是合法的时间证据。要排除的是仅看一个时刻、凭生成器的位置先验猜位移。

仅做时间反转增强不保证 `E[delta | 单帧图像]=0`，因为两个时刻的空间分布和标签条件关系仍可能不同。强制输出反对称也不够：任何 `g(I7)−g(I6)` 都会反号，哪怕 g 只响应无关纹理。因此同时要求**受控条件下的准确预测**，不能只检查符号翻转。

对每个独立 block 抽一个二维双物体中心 C 和非零四维位移 d，生成三个 RGB：`I−=render(C−d)`、`I0=render(C)`、`I+=render(C+d)`。四个有序样本如下：

| view | t6 | t7 | 真位移 | 配套控制 |
|---|---|---|---|---|
| 0：end+ | I− | I0 | +d | 与 view 1 完全相同的终帧 |
| 1：end− | I+ | I0 | −d | 与 view 0 完全相同的终帧 |
| 2：start+ | I0 | I+ | +d | 与 view 3 完全相同的起帧 |
| 3：start− | I0 | I− | −d | 与 view 2 完全相同的起帧 |

所有 view 的时间戳都为 [6,7]，两个 shutter 都开。颜色、glyph、背景与黄色 nuisance 在 block 内相同；nuisance 不随这两个时间步移动。主确认的 C±d 位于合法边界内，无 cue 反射；这有意隔离基础时间读出，不声称已覆盖反射动力学。

每张 unique image **只编码一次**，再复用相同 feature tensor 构造四个 view 和 `[I0,I0]`。不能把“相同终帧”分别送入不同 batch 后默认数值完全相同。actor/probe 只收图像特征及固定公共字段，不收 role、block/view ID、坐标、variant 或标签。

对同一个冻结读出器定义：

`c_end = (pred(view0) − pred(view1))/2`

`c_start = (pred(view2) − pred(view3))/2`

两者的正确目标都是 d。逐坐标 skill 为 `1−mean((c−d)^2)/mean(d^2)`。

**排除单帧捷径的理由是输入相等，不是有限几个 baseline 的表现：** 任意只看 t7 的确定函数在 view 0/1 上输出相同，故 `c_end=0`、该项 skill 精确为 0。任意只看 t6 的函数在 view 2/3 上同理，`c_start=0`。两项必须由**同一个**读出器同时通过，不能各挑一个模型。这也适用于非线性位置先验。

两项 contrast 还不够：一个大 common bias 可以在相减中消失，所以同时考察所有四个 view 的绝对位移误差。四个标签精确平衡，零位移就是预先定义的基线。

## 5. 数据、选择、指标与停止规则在执行前固定

完整机器可读定义：[v5_temporal_confirmation_plan.json](../scripts/v5_temporal_confirmation_plan.json)。以下是必须一起执行的约束。

### Fresh 数据

- 新 FIT：namespace `v5-temporal-confirm-1/fit`，block 0…255，共 256 block / 1024 有序样本。
- 新确认：`v5-temporal-confirm-1/confirm`，block 0…255，同样 256 block。它们与 FIT 及旧 Train/Dev 使用不同完整 hash namespace。
- **本次只用 software_test namespace 做代码测试；FIT/confirm 图像、特征和结果均未生成。** 先完成 FIT 内选择与模型冻结，之后一次性释放确认数据，不能先看图挑样本、拒绝采样或重抽种子。
- 原分布桥接：冻结 V5 生成器的 `dev` pair 128…383，两个 variant 全留，共 256 world / 512 episode，继续包含反射和原 nuisance。执行方须在读数据前证明这些 ID 未被使用；若不满足，停止并先登记修改，不能读到结果后换 ID。它检查新读出器能否回到原任务，避免仅在静态 nuisance 的新小任务上成功。
- 原保留的 **Test namespace 完全不生成、不提取、不查看**。

### 冻结选择

主表征固定原 4×4 pooled I-JEPA tokens；raw 只能作次要诊断，不能代替 pooled 开门。FIT 上比较两个预先固定的 CPU kernel 家族：原 concat linear，以及 `(1+K_linear)^2` 的 quadratic kernel，后者再次仅用 FIT 做 RKHS 中心化/trace 归一化。这样有限地检验非线性读出，不在 Dev 上试任意架构。

每个家族六个平均损失惩罚 `alpha=[1e−4,1e−3,.01,.1,1,10]/128`，求解 `K+n_fit*alpha*I`。4-fold 按完整四-view block 分组，seed=8317；所有中心化/尺度均在各 fitting fold 内计算。唯一选择目标为四个位移坐标的 normalized MSE；只用新 FIT。精确同分时先 linear，再较强惩罚。t6-only、t7-only 各有相同的 12 个候选与同一 CV 预算，全部报告，确认集不用于选单帧赢家。

在打开确认数据前提交：所有 fold 分数/分组、唯一选中家族和惩罚、拟合系数及 Train 统计量 hash、FIT cache/encoder/source/protocol/environment hash、输入字段审计。**当前已看过的 Dev 不再参与选择。** 旧 alpha=1 模型可以作为冻结的次要迁移对照；其结果不能替代主候选。

### 可明确失败的标准

1. 所有身份、数值、anchor equality 与标签不变量成立；确认含恰好 256 个完整 block。
2. 同一主读出器的 **8 个逐坐标 anchor-contrast skill**（两类 anchor × 四坐标），以及 **绝对位移平均 skill**，共 9 项的下置信界均大于 0。不能靠 A-dy 提升平均数掩盖 horizontal 失败。
3. 对完整 block 重采样 10,000 次，seed=7709；每项取单侧 `.05/9` 分位作为 Bonferroni 调整下界。它是固定模型、固定数据计划下的 percentile bootstrap，有限样本覆盖率为近似值；不会把软件测试或旧 Dev 区间当作确认区间。
4. 位置平均 skill 对冻结 FIT 均值仍需 >0.1。作为 reversal-balanced 拟合的数值一致性检查，反转配对预测之和、重复中心帧的位移输出，最大绝对值都应 ≤1e−6。它们可能由对称设计保证，**自身不证明运动信息**；证据来自有真标签的准确率/contrast。
5. 原分布桥接使用完全同一主读出器：位置 skill >0.1；平均位移 skill 的单侧 95% world-bootstrap 下界 >0；四个位移坐标点估计均 >0。报告普通 Dev-mean R²和全部反射分层，但不据分层修改门槛。桥接本身有位置先验，时间利用的证据由 quartet 提供。
6. 任一必要条件失败即停。不给失败样本补种子、不追加到显著为止、不换成 raw 或别的候选、不降阈值、不打开 Test。若要修改假设/读出族，另立版本和未用确认数据；不能把当前失败重新包装为通过。

这些条件是**新协议**，不是对原 `feature_gate.json` 的重新解释。256 block 是固定的机制确认样本量，未宣称做过功效校准；区间宽或不通过都是合法结果。

## 6. 何时允许 P0 与 memory 比较

目前不允许补算原 V5 gate 为通过。只有新确认与原分布桥接均通过、数据/模型冻结记录经独立复核后，才有理由在新 source commit 和新 run 目录启动 V5.1 的有限 P0 阶段。不要给旧 CLI 塞入假的 `passed:true`，也不要把新 gate 文件放回原运行目录。

建议目录已登记：

- 审查：`results/wm/visual-v5-probe-review-20261010/`
- FIT：`results/wm/visual-v5-temporal-fit-v1-20261010/`
- 确认：`results/wm/visual-v5-temporal-confirm-v1-20261010/`
- 桥接：`results/wm/visual-v5-temporal-bridge-v1-20261010/`
- 之后的 P0：`results/wm/visual-v5-p0-v5_1-20261010/`

P0 仍必须通过原设计的小样本拟合检查和 history gate：正确历史相对 recent-only 的 paired 改善下界为正、优于 swapped history、每个 family 的均值方向一致。**确认两帧线索可用，不等于预测未来成功，也不等于长程记忆有效。** 只有 P0 的这些条件成立，才冻结公共 P0 checkpoint，进入同预算 memory writer/reader 对照与 replay。

## 7. 本分支交付及未执行事项

[辅助代码](../scripts/v5_temporal_confirmation.py)提供独立 quartet 生成、12 候选的 Train-only kernel 选择、contrast/确认指标函数；[11 项软件检查](../tests/visual_memory/test_temporal_confirmation.py)覆盖相同起终帧、精确反转、无状态/ID输入、单帧函数必失败的反例、正确两帧 oracle、kernel 数学与分组隔离。测试只用 8 个 `software_test` block，不是科学拟合结果。

独立 worktree 中可运行：

```bash
python -m unittest discover -s tests/visual_memory -p test_temporal_confirmation.py -v
```

这是供本地 Codex 独立审查的 sidecar，不含 GPU 提取 runner，不自动改生产 gate 或继续 P0。后续执行方需要先把 encoder/cache/receipt 接入独立 runner，核对本协议，冻结该 runner commit；本次没有生成 fresh scientific 数据、申请 GPU 或操作 `/aifs4su/guhao/KVCache/WM`。此前草拟的重复缓存搜索脚本未纳入交付，避免重做已经完成的独立诊断。
