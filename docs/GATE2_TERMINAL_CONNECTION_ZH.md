# Gate2 终局删除 pilot 与 PM 的联系

2026-10-25。本页是跨仓库分析补充，不是一次新的 PM 实验。CV 的 Gate2 pilot 在 A100 作业11310上完成，调度状态为`COMPLETED 0:0`、耗时307秒。日志报告geometry gate和24项reader检查通过；**完整逐行结果仍待结果包回收后的独立核验**，不能把完成或日志判定当成已审计的方法收益。本轮没有运行 PM 模型、训练或新场景，没有打开 V6 FINAL，也没有修改 V6.1 的实现、阈值、预算或阶段顺序。

当前执行证据和传输限制见 [CV Gate2运行报告](https://github.com/JasonLIN0226/CV_project/blob/phase2/independent-interfaces-20261020/docs/GATE2_TERMINAL_PILOT_REPORT_ZH.md)。目前分块传输已校验21/83块；完整本地ZIP、独立数据审计与完整GitHub Release均未完成。这份PM说明只同步已取回的日志事实，不宣称完整流程或数值审计已完成。

CV 执行前源码记录为 [`3bd9f33`](https://github.com/JasonLIN0226/CV_project/commit/3bd9f33)；完整设计以 [Gate2 运行前协议](https://github.com/JasonLIN0226/CV_project/blob/3bd9f33/docs/GATE2_TERMINAL_PILOT_PROTOCOL.md)为准。该协议 SHA256 为 `44b78d5803726b8ee98b234da47b6dba63d62e5647db4488328be59072d47660`。

## 为什么做这个小实验

旧 CV 结果已经说明，保住 cue 与正确使用 cue 不是同一件事。固定时间和动作的内容替换能复现旧 reader 的失败；时间、内容和动作也存在交互，不能把其分配数值说成物理原因的百分比。扩大 reader 训练后，use-time 学生保住96/96条 cue，但没有胜过 attention/diversity。旧 local-cost 与 use-time 对比还同时改变了预测起点与目标时距，不能据此单独证明“标签时距太短”就是原因。详见[既有 CV/PM 连接报告](CV_CONFIRMATION_CONNECTION_ZH.md)。

Gate2 改问一个更窄的问题：**如果当前 reader 确实需要两份互补信息，而容量只够两项，视觉距离或 attention 会不会选择两份语义重复的信息？** 它先验证这个问题能否成立，不直接训练新的保留方法。

这一设计也不预设 attention 是平均贡献、diversity 必然失败，或删除监督必然更好。终局只有一次删除，直接评价同一未来目标，避免把任意指定的后续保留策略混入这个局部问题；它没有解决一般多步记忆控制。

## CV 本轮实际锁定了什么

每个场景包含独立的两个 bit，四种组合00/01/10/11构成完整 quartet，分别决定未来 x/y 方向。三帧证据中两帧表达同一个 bit，另一帧表达另一 bit。代码的 A/A′ 指重复轴，B 指另一轴；A 不固定等于 x。两种样式为 `benign_duplicate` 和 `distant_style_duplicate`，后者只是外观干预名称，是否真的在 I-JEPA 空间更远必须实测。

三帧位于 q−6、q−5、q−4，R=4、K=2。共同 FIFO 前缀使 q 时形成真实的三候选集合，第三帧恰好离开 recent；删除一次后冻结记忆。所有方法使用同一 reader、动作及目标 `L=(MSE_h3+MSE_h6)/2`，不会在未来重新取回丢弃证据。

TRAIN/VAL/GEOMETRY/DEV分别为240/48/24/48 quartets，每个基础 family 配对两种样式；DEV实际只有24个独立 family。四种 bits、两个重复轴和六种顺序全部报告。TRAIN独立拟合 normalization 与 input/output basis，VAL选reader checkpoint，DEV不调参。

执行顺序及止损条件如下：

1. **先测真实几何。** 用现有 diversity 实现和 TRAIN归一化后的 I-JEPA 特征，在专用 GEOMETRY 集记录三种删除的距离、tie-break和语义类别。若24个quartets里一个重复保留对都没有，停止，不训练reader、不搜索新seed。FIFO与终局均匀删除对照也在reader训练前锁定。
2. **Reader先过内容检查。** 把两槽时间固定为age `[6,4]`，只改变内容，比较AB、A′B、AA′、单证据、blank及单bit错误内容。整体和两个style分别要求：两个互补arm的四类原型识别率各≥80%，两种错误内容被识别为对应反事实未来的比例各≥80%，每个互补arm的MSE分别低于每个单证据arm超过1e−6，共24项检查。这里是四种真实未来feature的原型识别，不是物理轨迹解码准确率。
3. **再比较真实三候选删除。** 保存三个损失、精确最小集合、1e−6容差最小集合及其跨siblings一致性；另存语义正确集合，不假定learned reader与任务语义最优一致。Attention在reader冻结后计算；主对照为reader-attention、visual-diversity、FIFO与`terminal_uniform`。后者不是完整在线reservoir。
4. **完整保留负结果。** Reader失败、强基线已解决、或出现同parent差距分别解释。本轮不训练student。即便存在差距，也只得到开发线索，后续需要另一个独立新seed集合验证，而不是宣称方法成立。

各siblings的过去 cue 本来不同，因此它们的 oracle 动作不同不自动意味着不可学习；反过来，任务语义要求保留两个 bit，也不保证 reader特有的全部 oracle 差距都能从过去预测。原 reader没有完整利用除current之外的recent内容，本轮早期静止目标检查也不能被解释为一般early-reader质量证明。

## 能与 PM 结合到哪一步

PM 的研究对象仍是有界世界模型记忆的写入、修正和保护，按 [AGENTS.md](../AGENTS.md) 与 [V6.1预注册](PREREGISTRATION_V6.1.md)继续。可共用的是下面的判别问题，不能共用实验数值或直接迁移模型胜负。

| Gate2检查 | 对PM已有问题的对应 | 证据边界 |
|---|---|---|
| 两份看起来不同的信息是否语义重复 | 多条旧记录是否只是重复同一状态，另一项状态却没有得到保护 | PM旧bank的key匹配不等于Gate2两因素互补任务；本轮没有新增PM检验 |
| 高attention是否具有正的删除代价 | PM旧key被反复查询，仍可能给出有害旧值 | 被读到不等于有用；删除后回退零残差也不等于恢复新状态 |
| 共同reader能否利用完整证据 | PM先采用oracle L0输入排除上游问题，再在同输入/同预算比较memory | 不把CV编码器或reader扩样过程搬成PM感知诊断 |
| 删除损失是否与任务语义一致 | PM将总误差拆为signal/noise/cross，检查是否只追逐一次噪声 | 注入前slow-model residual仍不是物理真值 |
| 是否真的胜过强baseline | PM在同状态标量/字节、同输入、同损失下比较记忆规则 | 历史项bank与A2矩阵即使字节相同也改变表示和reader，不能全归因于淘汰 |

旧 PM 归档还提供一个必须保留的量级检查：在 [BR-0.2 的864个parent](../experiments/bank_reference/results/br-0.2-connect-20261022/parents.json)中，只有 **53/864** 的同parent局部删除oracle优于零记忆输出；原策略为47/864。该计数按 `min(candidate_parts.total) < zero_memory_parts.total − 1e−6`复核，容差改成0或1e−12计数相同。它只覆盖已观察的K+1候选中的一次删除，**不是全历史最优oracle，不是所有记忆方法的上限，更不是53个独立轨迹**；这些parent只有8个原A source，跨时刻/策略共享数据。

同时保留[既有案例图册](PM_CASE_GALLERY_ZH.md)的成功对照与噪声反例：局部oracle降低总误差，有时却恶化signal；旧值被删后覆盖下降，可能只是少犯错。旧A2的0.0125%账面修补空间也不因此升级为主线。

## CV执行记录与待核验结果

| 项目 | 当前状态 |
|---|---|
| A100作业、实际硬件与完成状态 | 作业11310：`COMPLETED 0:0`，307秒；硬件明细随完整包归档 |
| Geometry gate及各style完整计数 | 日志：benign重复保留0/48 episodes；distant为24/48，涉及6/12 quartets；geometry gate通过，逐行重算待结果包 |
| Reader选中checkpoint及24项gate | 日志报告24/24通过；选中checkpoint身份与指标待完整包核验 |
| 同parent强baseline与oracle差距 | 待回收 |
| 语义最优与reader最优是否一致 | 待回收 |
| 本轮新PM实验 | 无 |
| V6.1或FINAL边界改变 | 无 |

执行依据为CV的[调度记录](https://github.com/JasonLIN0226/CV_project/blob/phase2/independent-interfaces-20261020/results/literature_diagnosis/gate2-terminal-pilot-20261025/sacct-11310.txt)与[原始日志](https://github.com/JasonLIN0226/CV_project/blob/phase2/independent-interfaces-20261020/results/literature_diagnosis/gate2-terminal-pilot-20261025/slurm-11310.log)。远端打包receipt记载ZIP为345,671,988字节，SHA256 `523654adcb889726af0a8ce8a0dd3e59dba71b8c99e9634a2cd84ad09ed9be5c`；本页尚未把传输完成或独立审计标为通过。

结果填入后仍分别标记CV开发证据与PM已有证据。若Gate2失败，保留失败条件；若通过，也先说明差距来自什么、量级多大，再决定是否值得在独立数据上验证。两仓库目前可以共用“可见→保存→留存→正确读取→任务改善”的检查链，尚不能合并成已验证的新memory方法。
