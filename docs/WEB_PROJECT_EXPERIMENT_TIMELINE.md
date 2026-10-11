# 项目实验时间线：从预测库干预到视觉运动诊断

更新至 2026-10-10，依据各轮已保存的报告、逐例统计和核验收据整理。本文解释每一步为何发生、实际做了什么，以及下一步为什么改变。它不替代冻结协议或原始结果；V5 完整模型和大缓存的 GitHub Release 入口在各轮报告中，V1–V4 的独立分支及 V4 外部大包例外在下文明确列出。

**当前最重要的事实：原 V5 的 P0、history gate 和 memory 方法训练比较没有完成。** 首轮在特征门槛失败后停止；V5.1–V5.8 是为定位这一前置失败而进行的读出器、数据干预、对应和空间压缩诊断。后续保存的核读出器模型不是原 P0 或 memory writer 的训练检查点。V5.6 的更新 replay 也是有限状态接口的压力测试，不是原 memory 主实验。

## 阅读数值前先区分这些口径

- V1–V3 的 ES 是 RealMotion/AV2 预测评价；V4 是直接读取位置、速度和对象 ID 的合成状态实验；V5 是 RGB 经冻结 I-JEPA 后的视觉运动问题。三者不能直接比较分数。
- V5 中传统字段 `velocity_r2` 实际评分的是 **t6→t7 的观测位移**，含反射后的真实端点变化，不是生成器隐藏的瞬时速度。位置标签是 t7 位置。
- 后续 motion skill 逐坐标计算 `1 − MSE / mean((target − frozen FIT mean)^2)`，再平均四个运动坐标。水平 MSE 只平均 A_dx/B_dx。像素 RMSE 为归一化坐标 MSE 开方再乘 224。四轴 skill、两轴 dx RMSE 和位置 RMSE 不是同一指标。
- 一个 original-generator world 有两个配对 variants；512 episodes 通常只有 256 个独立 world。确认用 quartet 或多视图时，同 world 的视图仍不是独立样本。配对 bootstrap 保留整个 world；区间通常条件于已拟合模型，不包含重新抽 FIT 或重新训练的不确定性。
- 冻结 I-JEPA 使用 BF16 前向。转换为 FP32、FP32 池化、BF16 保存再转 FP32 缓存，不等于全 FP32 提取。特征张量的存储 dtype 不能单独说明模型计算精度。

## V1–V4：项目起点与转向

| 阶段 | 科学问题与实际执行 | 实际结论 | 尚未得到的证据 |
|---|---|---|---|
| V1 | 冻结 RealMotion，在旧预测库更新时训练干预控制器；完成单 A100、1,000 个 AV2 评估场景、seed 42 的真实运行。 | Keep ES 1.573144，controller 1.592829；以 Keep 减 controller 定义的改善为 −0.019684，95% CI [−0.034507, −0.004533]，控制器反而更差。 | 不是完整 AV2 论文分数复现，也不是连续视觉世界状态记忆。 |
| V2 / V2.1 | 将训练目标和筛选对齐最终 60 帧收益，增加诊断与回退；V2 本地验证后，V2.1 完成 A100 实验（Slurm 425）。 | 三个 Dev 候选均未通过门槛；最终 2,000 次决策全 Keep，ES 与 Keep 同为 1.573144。 | 回退保证没有采用不可靠候选，不代表学到了有效干预；V2.1 旧包缺逐场景候选 rollout，候选 CI 当时不能独立重算，V3 补齐。 |
| V3 | 增加冲突的时间变化和旧预测对当前输出的影响度，对比 base、temporal、full；A100 Slurm 450，归档包含候选逐场景记录。 | temporal ES 1.581897，改善 −0.008753，95% CI [−0.017783, 0.000219]；base/full 回退 Keep，没有可靠部署收益。 | 输出敏感性不等于干预效用；同 Eval 子集此前已用于 V1/V2，仍属探索，不能当全新确认。 |
| V4 | 转向“近期输入相同、早期历史不同”及“局部纠正与保留”的受控状态记忆；已知对象 ID，直接输入位置、速度、可见性。CPU 试跑后另在独立结果分支完成 A100 Slurm 615：五模型×三种子，12 个学习模型各1500步，Train/Dev/Eval=4096/512/512。 | clean/T48 的 GRU−slot hidden-position RMSE 收益为 0.521162 / 0.475473 / 0.421307，三个种子内的配对 CI 均为正；但 T96 仍明显漂移，解析恒速对照更稳健，seed44 的 preserved-object 指标 slot 还差于 GRU。 | 已知 ID/速度绕过视觉识别和对应，槽局部性部分由结构保证，不支持视觉 JEPA 结论。**外层与容器 GPU UUID 不一致，使用调度器分配物理卡的核验未通过**；计算完成不代表资源绑定正确。 |

证据：[逐版验证记录及原 PR 入口](VALIDATION.md)、[V4 机制与协议](STATE_MEMORY_V4.md)、[RealMotion 入口与研究边界](../README.md)。V1–V4 结果分支未全部合并到当前树，故还需以下永久入口：[V1 包](https://github.com/XintongYang1010/predictive-memory-lab/blob/8b6adfa6db484174a351eccc4d075391780253af/results/2026-10-02/yxt_pmr_week1_20261002.tar.gz)、[V2.1 包](https://github.com/XintongYang1010/predictive-memory-lab/blob/40cdcae3dcb43c1cb08ed747c8767bb967c01bf8/results/2026-10-03/yxt_pmr_week1_v2_20261003.tar.gz)、[V3 包](https://github.com/XintongYang1010/predictive-memory-lab/blob/2fb3e285a4af19ecd71d9b6e9958c7b73f6f6cdb/results/2026-10-04/week1_v3_20261004T015400Z_450.tar.gz)、[V4 实测报告](https://github.com/XintongYang1010/predictive-memory-lab/blob/542bb55a05762658d3c1265b7bf183c1e2048a14/results/2026-10-09/v4-job615/README.md)。

`VALIDATION.md` 的 V4“尚未正式 A100”是较早本地记录，已被上述 Slurm615 实测取代。V4 资源绑定修正模板只完成软件检查，未新跑 A100。其 GitHub 结果分支保存报告、汇总、曲线、日志与核验；原约343MiB全包当时留在 Slack 和服务器，不能说其中全部 checkpoint 已镜像到 GitHub。

## V5 首轮：先验证视觉特征是否支撑后续任务

**问题。** 从直接可读状态走向 RGB：官方冻结 I-JEPA ViT-H/14 的空间特征，经 4×4 平均池化后，能否读出位置和短时位移？原计划只有特征门槛通过后才进入完整缓存、P0、历史必要性检验和 memory 对比。

**实际执行。** 将 NYU 启动器适配到实际 DGX/Slurm。用户选择共用既有分配中的八张 H800；这不是 A100 实测。官方 encoder profile 为 `415453.14`，八 worker probe 为 `415453.15`。Train/Dev 各 128 episodes 完整生成，所有 worker 完成提取；评分按预定门槛失败，后续阶段未启动。

**结果。** raw 的位置/位移分数为 0.957126 / 0.080553，pooled 为 0.974691 / **−0.020927**。池化位移未超过零；raw 的平均正分也掩盖了负的单坐标，不能称运动全部可读。没有 P0、九组 memory/reference 训练、reader adaptation、Test 或新的训练检查点。

**边界。** 这是某一输入、探针和正则参数下的失败，不是 I-JEPA 完全不含运动信息的证明，也没有检验记忆是否有用。`415453.15` 的非零退出来自科学 gate，不是八 worker 未完成。失败现场保留。

证据：[首轮报告](../results/wm/visual-v5-dgx-20261009/README.md)、[原门槛独立复算](../results/wm/visual-v5-dgx-20261009/run/feature_diagnostics/verification.json)、[原协议](VISUAL_MEMORY_V5.md)、[DGX 执行记录](DGX_V5_EXECUTION.md)。

## V5 后续 CPU probe 审查：正分是否只是位置先验？

**问题与执行。** 在原 128/128 已看缓存上检查固定 ridge 正则、单帧/重复帧、时间差分、raw 重新池化、原生成器反射与 RGB 可观察性；仅 Train 内分组交叉验证选择正则，Dev 用于诊断，不新增 GPU 或 Test。

**结果。** pooled concat 的原 α=0.01 得分 −0.020927，Train-CV 选 α=1 后为 0.243895；但同一流程的单帧 t7 已达 **0.258056**，两帧没有优于这一对照。原颜色先验 RGB 质心两帧 oracle 约 0.978，单帧位置也能预测一部分运动，说明生成器存在位置—位移先验。换帧顺序及端点控制没有支持原 probe 已经可靠利用时间信息。

**结论与边界。** “正则会影响分数”和“正位移分数不等于时间理解”同时成立；不能重调参数后回写原 gate 为通过。代码审查与同源重新池化未找到能解释原失败的阻断实现错误，但这不是穷尽所有错误的证明。RGB oracle 用了场景颜色先验，不是 encoder 或模型自身能力。

证据：[CPU 诊断总报告](../results/wm/visual-v5-probe-diagnostic-20261009/README.md)、[输入与单帧控制](../results/wm/visual-v5-probe-diagnostic-20261009/input_audit/README.md)、[时间控制](../results/wm/visual-v5-probe-diagnostic-20261009/temporal_controls/README.md)、[原始设计复核](../results/wm/visual-v5-probe-review-20261010/README.md)。

## V5.1：新时间对照通过，回原分布仍失败

**问题与执行。** 构造共享起点/终点的四视图 anchor，使单独依赖首帧或末帧不能通过全部时间差分对照。先拟合并冻结核读出器、发布完整模型，再生成新 256-world confirmation 和原生成器 Dev128–383 的 256 对 bridge。GPU steps 为 `.18/.20/.22`，CPU 拟合/评分为 `.19/.21/.23`，均属于父分配 `415453`。

**结果。** 新 confirmation 的 concat motion skill **0.674117**，位置 0.988371，预定时间对照通过，单帧未通过。可是 original-generator bridge 的 motion skill 为 **−1.538523**；两个水平坐标严重失败。反射与非反射组都差，反射不能单独解释失败。

**结论与边界。** 新构造分布里，固定读出器确实能利用时间变化；它不能泛化回目标 bridge。多个 anchor 对比代数相关，不能将九项正的下界说成九次独立重复。confirmation 通过不覆盖 bridge，因而仍不启动 P0，旧 V5 gate 不变。

证据：[V5.1 报告及 Release 入口](../results/wm/visual-v5_1-20261010/README.md)、[confirmation 原报告](../results/wm/visual-v5_1-20261010/run/confirm_report/report.json)、[bridge 原报告](../results/wm/visual-v5_1-20261010/run/bridge_report/report.json)。

## V5.2：固定黄色干扰物，直接检验 bad case 的可干预来源

**问题与执行。** 保留 V5.1 已看 bridge 的目标、背景、顺序和冻结模型，只改变黄色干扰物：原运动、固定在 t6、固定在 t7、交换运动，四条件全部报告，不选最好者。原始条件预测逐位复现。GPU `.26`，CPU `.27`，256 对旧 world，不是新确认。

**结果。** 原始四轴 skill −1.538523；固定 t6 / t7 后分别 **0.389818 / 0.381036**，交换黄色运动仍为 −1.492813。原始 dx MSE 减两个静止条件平均为 **0.000624824**，95% CI [0.000573715, 0.000676973]，相对减少 **85.63%**。245/256 对改善；同时保留时间槽边际匹配的 balanced 对比。

**结论与边界。** 黄色渲染运动对这一冻结 encoder→pool→reader 整体的误差有很强因果贡献。85.63% 不是 encoder 的责任占比。原 FIT 的黄色位移恒为零，更准确的缺口是缺少区分目标与干扰运动的监督约束，不能说已证明训练学到了“黄色速度与标签”的相关性。静止后还有方向错和幅度不足，原移动干扰任务并未修复。

证据：[V5.2 报告](../results/wm/visual-v5_2-20261010/README.md)、[四条件原评分](../results/wm/visual-v5_2-20261010/run/report/report.json)、[坏例与干预图](../results/wm/visual-v5_2-20261010/analysis/README.md)。

## V5.3：用匹配的静止/平衡监督修复干扰混合

**问题与执行。** S 用静止干扰监督，B 用静止及双向移动干扰的平衡监督；同一 FIT 图像、同一共享核参考、固定 α、每 world 相同总权重、自由截距。拟合并发布后，评估新 anchor256 与未访问的原 Dev384–639。GPU `.29/.31/.33`；CPU `.30/.32/.34`。

**结果。** Fresh bridge 四轴 skill 从 S 的 **−1.664064** 升到 B 的 **0.338657**；四个坐标均为正。主端点 S−B dx MSE **0.000638206**，95% CI [0.000584358, 0.000692627]。Anchor 中 B 保留非零目标响应，并把移动干扰路径从 S 的约 −0.510 改到约 0.698；静止目标负控的 overall RMS/FIT 从 1.110 降到 0.207。

**结论与边界。** 匹配干扰监督能够改善新原分布上的读出，且不只是把输出缩成零。S 也接触共享的无标签归一化分布；单帧 S/B 分布等价与预测一致是实现检查。B 仍远非准确追踪，后验坏例提示位置覆盖、幅度和串扰问题。新结果不替换旧 gate，更不是原 memory 训练成功。

证据：[V5.3 报告](../results/wm/visual-v5_3-20261010/README.md)、[bridge 原评分](../results/wm/visual-v5_3-20261010/bridge_report/report.json)、[anchor 原评分](../results/wm/visual-v5_3-20261010/anchor_report/report.json)、[后验坏例与覆盖分组](../results/wm/visual-v5_3-20261010/bad_cases/README.md)。

## V5.4：扩大位置覆盖，未确认总体水平收益

**问题与执行。** N 复用旧窄域 FIT；W 用相同 latent seeds 和量化前分位数扩大目标中心范围，位移、颜色和黄色抽样保持一致。两组各 256 world、4096 监督视图，用联合 N+W 的 8192 参考视图重新定义共同核并分别拟合。冻结发布后评估新宽域 anchor256 和原 Dev640–895。GPU `.35/.37`，CPU `.36/.38/.39`。

**结果。** 新 bridge N / W 的 dx MSE 为 **0.000112544 / 0.000118455**；N−W 为 −0.000005911，95% CI [−0.000016793, 0.000005880]，**没有确认 W 的主收益**。W 的四轴 skill 为 0.433999，高于 N 的 0.353835，主要指标口径不同，不应挑四轴均值掩盖水平主比较。两组 FIT motion skill 约 0.9995，但新 bridge 明显下降。

**结论与边界。** 固定样本量扩域带来密度和相对干扰几何变化；分组显示域内外取舍，不能由本次阴性排除全部覆盖方案。新 N 也重算了联合核，不能当旧 V5.3 B 的数值复现。训练高分与新样本低分确认泛化差距，不能单独定位 encoder、池化或正则化故障。

证据：[V5.4 报告](../results/wm/visual-v5_4-20261010/README.md)、[bridge 原评分](../results/wm/visual-v5_4-20261010/bridge_report/report.json)、[FIT 与新样本差距](../results/wm/visual-v5_4-20261010/training_generalization/README.md)、[固定坏例](../results/wm/visual-v5_4-20261010/bad_cases/README.md)。

## V5.5：对象因子干预与 lane 读出器

**问题与执行。** 在已看 Dev640–895 固定首帧，将末帧 A、B、黄色物体的位置各取原 t6/t7，组成 8 个端点。冻结全局 W；另仅用旧 FIT 拟合固定 lane 读出器：A 访问 pooled 上两行，B 访问下两行。保存同次 encoder 的原 tokens、池化前值和 canonical 池化值。GPU `.40`、CPU `.41`；随后 GPU `.42`、CPU `.43` 完成全量 batch4 补验。

**结果。** 原始 `000→111` 查询的 dx MSE 降 **29.78%**；至少一个目标水平错向的 episodes 从 **196/512 降为 139/512**，四轴 skill 0.4340→0.5799。对象因子分解显示 lane 增强自身目标响应、降低跨目标响应，黄色响应没有一致降低。844/1 两个方向恢复但幅度很小；761/1 仍双错；891/1 的 B 仍被其他变化压过。

**结论与边界。** 同一已有表示的读出方式可改进，但 lane 同时改变特征访问、核几何和有效正则；不能直接归因为“删除了无关 token”。空间 tokens 仍有全局上下文。Shapley 响应不是错误责任百分比；池化后保留约 9.98% 的 token 差分投影能量也不等于只保留约 10% 的运动信息。

**已解决的数值疑问。** 首轮原图 batch4、编辑图 batch12 的潜在混淆，后来用全部 4096 图像槽统一 batch4 排除：83,886,080 个 canonical 特征值逐位相等，三模型全部预测差为零。它不应继续列为这批结果未排除的原因。原 W 的帧交换反号、重复帧约零和非零跨帧项也否定了“只是两个独立单帧位置函数相减”的简化描述，但不保证预测正确。

证据：[V5.5 报告](../results/wm/visual-v5_5-20261010/README.md)、[对象分解与池化审计](../results/wm/visual-v5_5-20261010/report/report.json)、[全量 batch4 补验](../results/wm/visual-v5_5-20261010/batch_match_run/comparison/report.json)、[RGB 可观察性后验对照](../results/wm/visual-v5_5-followup-design-20261010/rgb_observability/report.json)。RGB 质心约 0.403px 的 dx RMSE 依赖已知颜色，不能与无此先验的模型当同等输入比较。

## V5.6：拆读出混杂、试三个方向，并做一次新组合确认

**问题与执行。** 这一轮包含 CPU 上的 12 臂 reader 消融、六种同源对应/等字节 bank、64-transition 更新 replay；另用八卡生成对象组合 FIT/轴干预，再冻结 J/X 模型并发布后生成 fresh64。GPU `.44/.49`；CPU `.45–.48/.50`。原 Dev640–895 仍只是已知诊断，fresh64 是新 namespace。

**读出器消融。** 固定 W4096 监督，比较共同尺度下的 full、去跨区域乘积的 additive、目标区域核，以及正确 lane、反向 lane、checkerboard；每类同时报告原 α 和只用 FIT 谱匹配有效自由度的版本。仅 df 匹配未改善全局；共同尺度目标访问仍降低约 **26–28%** dx MSE；反向 lane 和 checkerboard 失败。访问位置的针对性得到支持，但只排除了这些指定对照，不能穷尽所有核几何解释。

**对应与存储。** 固定 RGB 特权源 ROI 的二维匹配器，raw16 的 dx RMSE **1.534px**，同 raw 平均成 pool4 后 **3.878px**。同字节的 ROI bank **1.738px**，uniform **3.753px**，random **4.772px**，变化幅度优先 **6.663px**。变化优先细节几乎都覆盖黄色干扰，说明大变化不等于任务相关变化。所有 matcher 都用源 RGB 的位置/颜色先验；bank 字节不含完整 raw 选择工作区，变化优先还读取两帧，不能称在线因果 memory 策略或公平端到端存储成本。

**更新 replay 阴性。** 同一冻结 lane 位置 reader 与状态预算，latest/FIFO 位置 RMSE 2.691/2.685px；局部变化阈值 3.001px，漏写 842/2048 次真实对象变化，仍更差。这是二元端点与无观测间隔压力测试，不是训练型 writer 或完整物理轨迹。

**J/X 新组合确认。** J 重复旧 16 视图；X 旧 16 加独立 A/B 方向组合的 16 视图；共同核参考和 mean-loss α。Fresh64 全 32 视图的 dx MSE 从 **0.000106922 降至 0.000083031，减少 22.34%**；J−X 的 95% CI [0.000015458, 0.000032002]。旧组合16与混合16均报告，不能当两次独立确认。四轴 skill 0.505→0.623。

**仍未解决。** 中心静止负控仍有约 0.77–0.81px 四轴假运动。已知轴干预中，纯 y 造成的错误 x 响应反而从 J 的 A/B 0.821/0.810px 增到 X 的 1.016/0.940px；对象组合覆盖收益没有解决对象内坐标混合。改进既可能救回，也可能破坏样本，不应只列单向错误。外侧行替换不是“真实目标 token 替换”；同输出上下等大小对照未支持对两个目标普遍成立的目标外行优势。

证据：[V5.6 全报告](../results/wm/visual-v5_6-20261010/README.md)、[12 臂消融](../results/wm/visual-v5_6-20261010/reader_ablation/README.md)、[对应坏例](../results/wm/visual-v5_6-20261010/correspondence_interpretation/README.md)、[replay 独立重算](../results/wm/visual-v5_6-20261010/cpu/replay_verification.json)、[组合/轴分析](../results/wm/visual-v5_6-20261010/combination_analysis/README.md)、[fresh64 原评分](../results/wm/visual-v5_6-20261010/gpu/combination_confirm/report.json)、[空间路径控制](../results/wm/visual-v5_6-20261010/replay_spatial_control/README.md)。

## V5.7：正确 y 与简单度量改动能否救回 x？

**问题与执行。** 仅 CPU、旧 Dev 的 1024 个对象查询。第一组在六个已保存成本面上比较原二维搜索、正确 GT-y、零 y、循环 +37 donor 的错配 y 及 GT-x。第二组比较 raw16/pool4 × 不减均值/共享源均值/逐帧均值 × cosine/未归一化平方距离；无训练、无新数据。

**结果。** 提供正确 y 后仍由成本自主选 x：raw dx RMSE **1.534→1.562px**，变化区间跨零；pool **3.878→4.080px**，显著更差。原正确 x 有时被错误 y 补偿。GT 最近点在正确行的中位 rank 为 raw 5/25、pool 12/25。六个配对原点/度量设置都保留明显 raw/pool 差距；pool 逐帧均值＋L2 有小幅改善，但仍约 **3.744px**。

**结论与边界。** 不能只把水平错误归于选错 y；简单去均值或改平方距离不足以修复成本排序。GT-x 直接给了 x，它带来的 dx 接近真值不能算自主收益，只能考察没给的 dy。共享均值不改变 L2 成本的负控通过；均值本身不能被称为纯背景。此处是固定对应器，不是旧核回归或原 memory。

证据：[V5.7 报告](../results/wm/visual-v5_7-20261010/README.md)、[oracle/错配与排序](../results/wm/visual-v5_7-20261010/cost/report.json)、[12 组原点/度量](../results/wm/visual-v5_7-20261010/descriptor/report.json)、[相关原论文及未解决边界](V5_7_RELATED_WORK.md)。

## V5.8：把空间平均与粗采样重建拆开

**问题与执行。** 仅 CPU，复用同 raw、源 ROI、cosine、625 个候选。五组为：R 原 raw16；H 同相位密集 2×2 平均；BR 同中心密集 4×4 平均；DR 从 H 粗采样并双线性重建；DBR 从 BR 粗采样并重建，逐位等于旧 pool4。H 在评分前加入以控制 BR 的半格相位变化。各组直接在自己的物理网格插值。

**结果。** R/H/BR/DR/DBR 的 dx RMSE 依次 **1.534 / 1.520 / 3.392 / 4.808 / 3.878px**。同相位密集 BR−H 的 dx MSE 增加 **9.199px²**，95% CI [8.216, 10.233]；同一 2×2 场的 DR−H 增加 **20.811px²**。全部候选均不触网格边界的共同内点仍有 863/1024 查询、覆盖全部 256 world，方向一致。844/1 A 与 891/1 B 不需先粗采样，H→BR 就能从正确方向变错。

**必须同时保留的反例。** 粗网格上 DBR 优于 DR，dx MSE 差 **−8.081px²**，所以不能说平均普遍有害；平滑和采样的作用不相加。H 的水平差相对 R 未显著，且四轴 MSE 更差，H 也不是已证实的全面修复。

**结论与未排除项。** 已定位到这些空间压缩算子与当前固定 cosine 成本排序的耦合：较大平均范围、粗采样加重建均足以在这一设置下损害定位；边界复制不能单独解释。尚未分离采样与重建、证明具体频谱混叠机制、测定信息不可恢复程度或识别 encoder 某层故障。没有测试训练型读出能否恢复，更没有检验原 memory。未来的已知描述子场平移、像素平移等变性、学习型对应和查询选择仍是候选，未在本轮执行。

证据：[V5.8 报告](../results/wm/visual-v5_8-20261010/README.md)、[全部九个对比与计数](../results/wm/visual-v5_8-20261010/spatial/report.json)、[固定坏例](../results/wm/visual-v5_8-20261010/spatial/fixed_cases.json)、[结果独立复算](../results/wm/visual-v5_8-20261010/spatial_result_review.json)、[归因审阅](../results/wm/visual-v5_8-20261010/interpretation_review.json)。

## 哪些旧解释需要修正

| 旧说法或容易产生的误读 | 现在应如何表述 | 修正证据 |
|---|---|---|
| V4 只有 CPU smoke，尚未正式运行 | 独立结果分支已有 A100 Slurm615 三种子正式实验；但物理GPU绑定核验失败，完整大包不在该Git分支。 | V4 实测报告；旧 VALIDATION 状态已过时 |
| 正位移 R² 已证明时间理解 | 原 probe 的单帧位置先验即可得正分；V5.1 才加入共享端点/起点控制，且 bridge 仍失败。 | CPU probe、V5.1 |
| 所有双帧 reader 都只是两个单帧位置函数相减 | V5.5 冻结 W 有非零跨帧项；交换反号不等于加性单帧分解。该结构结果也不保证准确。 | V5.5 结构审计与报告 |
| BF16/FP32 或 batch4/12 导致对象因子结果 | FP32 缓存不代表 FP32 前向；同次池化/量化检查通过，完整 batch4 补验对全部特征与预测为零差。 | V5.5 及 batch 补验 |
| 池化丢了 90% 运动信息 | 约 90% 是 token 差分投影能量，非任务信息；后续只证明某固定 matcher 对具体算子敏感。 | V5.5、V5.6–V5.8 |
| 黄色干扰解释了全部失败，lane 已去掉黄色 | 黄色干预显著改善但留残差；lane 仍对黄色敏感，空间 tokens 本身全局上下文化。 | V5.2、V5.5、V5.6 |
| 反射、位置覆盖或一个正则系数就是唯一原因 | 非反射也失败；单次固定256-world扩域主区间跨零；df匹配本身未修复全局。 | V5.1、V5.4、V5.6 |
| 更大的变化代表更值得写入的内容 | ΔL2 细节多选黄条，匹配最差；局部变化 gate 漏写并落后 latest/FIFO。 | V5.6 |
| 扩域无用，或组合覆盖已全面修复 | V5.4 仅未确认指定扩域方案；V5.6 新组合确认有收益但纯 y→假 x 响应仍在且增大。 | V5.4、V5.6 |
| 给 GT-x 后 dx 更好证明对应器修复 | 直接提供被评分坐标是 oracle 输入；只可解释未直接提供的另一个坐标。 | V5.7 |
| pool 错只是 y 选错、均值背景或插值触边 | 正确 y 未救回；简单原点/度量改动仍留差距；共同无触边集仍退化。 | V5.7、V5.8 |
| 平均都坏，或可按误差差分分摊模块责任 | 粗网格上更强平均有条件地更好，交互显著；当前只能谈具体算子与固定 matcher。 | V5.8 |
| V5.1–V5.8 已完成原 memory 训练 | 完成的是前置诊断及一些原型；P0/history gate/完整 memory 对比未执行，Test 未打开。 | 各轮 report 的 `p0_authorized=false` / Test 标记 |

## 数据使用与执行范围索引

| 数据/实验 | 首次用途 | 后续地位 |
|---|---|---|
| 原 Train/Dev 各128 episodes | V5 首轮 probe | CPU probe 复用，事后探索。 |
| V5.1 新 quartet confirmation256；原 Dev128–383 | 时间确认与 bridge | Dev128–383 在 V5.2 已看反事实中复用。 |
| V5.3 新 anchor256；原 Dev384–639 | S/B 冻结后新评估 | 后验坏例/覆盖分析仍属探索。 |
| V5.4 新宽域 anchor256；原 Dev640–895 | N/W 冻结后新评估 | V5.5–V5.8 主 bad-case/因子/对应/空间分析反复复用；不能每轮重称 fresh。 |
| V5.6 新 namespace 的64 worlds×32 views | J/X 完整模型发布回读后生成 | 只确认预定组合增强比较；不确认长期记忆、布局交换、坐标独立或所有后续想法。 |

V5 至 V5.6 的 GPU 实测使用父分配 `415453` 中的 H800，八 worker 是并行提取，不是八个训练种子。V5.7/V5.8 仅本地 CPU，没有新 Slurm step 或 GPU 用量；截至 V5.8 的可归属累计 charge 为 **5.56779064169697 GPUh**。这不是整项长期共享 allocation 的占用或计费总量，见 [V5.6 对账](../results/wm/visual-v5_6-20261010/final_accounting.json)。

归档完成与科学成功是不同状态。V5 初始及较早失败调查现场保留；V5.5/V5.6 的成功新诊断目录只在完整 GitHub 回读通过后按范围清理，既有目录与作业保留。V5.7/V5.8 没有新远端目录可删。各轮 README/ARCHIVE 给出 Release 依赖与恢复次序；以各轮验证/清理收据为准，不能用“已 push README”代替完整模型和缓存回读。最新可查 [V5.5 回读/清理](../results/wm/visual-v5_5-20261010/cleanup_receipt.json)、[V5.6 回读/清理](../results/wm/visual-v5_6-20261010/cleanup_receipt.json)、[V5.7 回读](../results/wm/visual-v5_7-20261010/github_verification.json)、[V5.8 回读](../results/wm/visual-v5_8-20261010/github_verification.json)。

网页端讲解应沿着“原任务需要什么 → 前置失败 → 可干预因素 → 新确认/旧样本诊断 → 仍缺的记忆证据”阅读，而不是把所有后续正分连成原方法已被验证的故事。
