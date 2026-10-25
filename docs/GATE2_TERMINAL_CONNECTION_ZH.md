# Gate2 终局删除 pilot 与 PM 的联系

## 2026-10-25补充：三任务协议和执行代码已冻结

CV已在commit `4e279fc4b57bbb333cf82f2bac6cde865aa47ad9`提交[正式预注册](https://github.com/JasonLIN0226/CV_project/blob/4e279fc4b57bbb333cf82f2bac6cde865aa47ad9/docs/TERMINAL_STRUCTURES_PREREGISTRATION_20261025.md)、R/D/S生成器、0–3历史共同reader、48/64预测对照、独立/集合学生及确认集自动门槛。29项针对性测试通过；原图容量profile不属于视觉效果验证。新reader使用纯MSE，避免旧排序损失改变缺信息时的均值预测。48/24/24/48个TRAIN/VAL/DEV/CONF families与所有seed在编码前锁定；CONF未通过DEV reader和模型绑定门槛不得生成。

本条只证明代码、协议和来源已公开，**不是运行完成或方法胜出**。运行结果随后单独更新；PM未启动新实验，V6.1 S0–S2和FINAL保持原协议。

## 2026-10-25补充：三结构设计与符号预检

CV下一轮草案已扩展为冗余、加性诱饵和关系XOR协同三族，均保持K=2、三个终局候选；新增相对完整K+1预测的敏感度对照。此次只完成精确符号枚举和纯预测打分函数，没有新视觉reader、A100、确认集或PM实验。详见[CV协同设计审计](https://github.com/JasonLIN0226/CV_project/blob/phase2/independent-interfaces-20261020/docs/GATE2_SYNERGY_DESIGN_AUDIT_ZH.md)。

加性诱饵不能仅凭“只有一种删除正确”就断言medoid失败：缺两轴的特征误差正交时，它会正确删诱饵。理想XOR中两个缺信息预测同为条件均值，medoid反而删必要项。XOR本身仍不要求集合条件淘汰——角色可见时逐项scorer可直接删D。因此新协同族用可见tag的(u,u,v)关系确定有效pair，并对其payload求XOR。六种tag排列、八种逐槽payload完整交叉；外观和共同context不随真实角色变化。

在这些输入条件下，精确枚举六个位置/tag分数的720种排序得到独立scalar scorer最多正确3/6，即50%；集合tag计数可全部选择正确。这是限定输入结构的数学结果，不是已训练视觉网络的结果。符号预检也保留了错误完整参照导致敏感度失败的反例。未来须共同训练并分别验证K=2/K=3 reader，不能把自洽直接当效用。

这与PM仍只共享“预测模型能否指导记忆决策”的研究问题。关系XOR不是PM CHANGE或BF-Delta的新结果，既有S0–S2、预算与FINAL边界不改。若无训练参照已解决质量问题，蒸馏学生的目标可改为摊薄计算，但本轮没有在任一repo训练这种学生或进行测试时适应。

## 2026-10-25补充：无训练预测一致性已通过CPU输入隔离检查

CV新增的TRAIN动作平均medoid只丢唯一证据2/192，平均regret=0.006753；同一组DEV上reader-attention为101/192、0.037452，diversity为48/192、0.038896。该规则用固定reader分别预测三种删除，每种删除共享16个TRAIN参考动作计划；先算同计划下的两两预测MSE，再平均并选medoid。选择器不接收实际未执行动作、真实未来或语义角色，全部192个决策先锁定，再用归档实际计划损失评价。独立数值审计重算192次选择通过。仍是24个已查看的DEV family，**没有新seed确认，也没有新的PM运行**。

这使下一步更明确：先把它作为学习型scorer的强对照，不能默认必须学习集合条件打分。它的计算成本是每决策48个reader预测样本，额外动作库768字节；768字节不包含reader、basis副本或临时工作区。CPU主推理加距离计算合计2.031秒，不能当作GPU训练时间或等算力比较。详细输入隔离、区间、成本和原始数据见[CV预测一致性报告](https://github.com/JasonLIN0226/CV_project/blob/phase2/independent-interfaces-20261020/docs/GATE2_PREDICTION_CONSENSUS_ZH.md)。

本次还修正推进条件：29/48的精确副本argmin一致性和早期reader失败不挡住终局研究。必要证据丢失率和regret为主，副本精确选择为辅。Attention的101次失败全部对应唯一证据的严格最低读取权重；diversity的48次失败体现预先测到的距离几何，而不仅是一个窄反例。

下一轮[合并草案](https://github.com/JasonLIN0226/CV_project/blob/phase2/independent-interfaces-20261020/docs/GATE2_COMBINED_STUDY_DRAFT.md)把冗余[A,A′,B]和诱饵[A,B,D]作为两个各自K=2、三个候选的任务族，平衡显眼外观的位置。前者有两种删除能保住完整信息，后者只有删D能保住完整信息，可以检验“一致”是否等于“有用”；不能预言共识规则必定胜出或失败。草案尚未冻结执行，也不是新训练结果。

与PM共用的问题是“能否让冻结预测模型直接辅助记忆决策”。V6.1 BF-Delta基于已观察到的当前残差，对保持/重置假设比较预测证据并形成遗忘门；CV规则则比较假设动作下三种删除的预测一致性。两者不是同一算法，本次不修改V6.1源码、阶段、阈值、预算或FINAL状态，不把CV数值并入PM结果。下文保留原pilot及旧bank的独立证据范围。

2026-10-25。本页是跨仓库分析补充，不是一次新的 PM 实验。**CV Gate2已完成A100运行、完整归档取回与独立数值审计，出现了真实的互补证据保留失败；尚未证明学生能够学会修复，也未证明必须使用集合条件打分。** A100作业11310为`COMPLETED 0:0`、耗时307秒。本轮没有运行 PM 模型、训练或新场景，没有打开 V6 FINAL，也没有修改 V6.1 的实现、阈值、预算或阶段顺序。

执行、数值结果与归档入口见 [CV Gate2运行报告](https://github.com/JasonLIN0226/CV_project/blob/phase2/independent-interfaces-20261020/docs/GATE2_TERMINAL_PILOT_REPORT_ZH.md)。先前网络中断已经恢复：83/83块、完整ZIP SHA256与CRC、136个源/结果文件的manifest均通过。独立NumPy审计通过68,328项检查，零失败、零警告；它复算保存的预测和指标，没有独立重新推理模型。

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

## 本轮CV结果：可以继续检验，但不是已经学到

24项reader门槛全部通过。固定时间/动作的AB和A′B识别正确分别为192/192、191/192；只翻A或B内容时，预测转到对应反事实未来，均为192/192。更关键的是，**真实时间戳下的全部384个互补保留对都识别正确**，冗余保留对仅96/192。因而这轮所见的丢唯一证据，不再只能解释为共同reader不会利用互补信息。

| DEV外观 | 策略 | 丢唯一证据 | 四未来识别 | MSE | 相对同parent oracle regret |
|---|---|---:|---:|---:|---:|
| 普通副本 | diversity | 0/96 | 96/96 | 0.104013 | 0.005414 |
| 普通副本 | reader-attention | 61/96 | 64/96 | 0.139173 | 0.040574 |
| 普通副本 | oracle | 0/96 | 96/96 | 0.098600 | 0 |
| 改变副本外观 | diversity | 48/96 | 72/96 | 0.174863 | 0.072378 |
| 改变副本外观 | reader-attention | 40/96 | 75/96 | 0.136816 | 0.034331 |
| 改变副本外观 | oracle | 0/96 | 96/96 | 0.102485 | 0 |

总体diversity regret为0.038896，24个family重采样2,000次的描述性95%区间为[0.028249,0.049900]；reader-attention为0.037452，[0.028449,0.047010]。这些是开发场景中的特征MSE差距，不是自然任务上的物理成功率，也不是独立确认。该attention臂是冻结reader的因果读取分数，并非新训练的PaFu-KV学生；本轮没有新增WR或PaFu学生实验。

保留以下三个限制，才能决定下一步是否值得做：

- **差距有特定场景结构。** Diversity的48次语义失败全在“改变副本外观＋重复竖直轴”；改变外观但重复水平轴和普通副本均有成功控制。普通副本diversity已经保住全部必要信息，其0.005414剩余regret是副本/时间读取偏好问题，不能称为信息丢失。不能从这个人为设计的两style pilot推广到PushT或自然视频。
- **语义正确与精确MSE最优仍不等价。** 全部192个oracle最小集合均语义正确，且没有精确或1e−6容差平局；但四siblings共享同一个最优动作的只有29/48 quartets，其余19组在两个冗余副本之间改变偏好。所有过去都包含足以确定任务语义的两位信息，不等于全部reader特有的最优副本选择已经可学习。
- **Reader和保留器的可见信息不同。** Reader可接收q之后的计划动作；按本轮协议，保留scorer不能接收尚未执行的动作。保留两个轴的语义要求不依赖这些正向动作的具体幅度，但精确feature-MSE的副本排序可能依赖计划动作和时间/读取交互。因此后续应检验过去信息下的期望收益，不能要求学生复现所有hindsight argmin，或把没有相同输入冲突记录当成可学性证明。

早期reader检查也仍失败：q−6时预测两个运动前静止目标，因果复制前一空白帧的MSE为0，reader却为0.342578（普通0.170035，改变外观0.515122）。这不否定query终局对照，但阻止我们直接把该reader用于可靠的早期local/use-time监督，更不能说旧时距混淆已全部解决。

审稿判断是：**可以预先锁定一个小规模学生比较来检验可学性，不能现在宣布可学缺口或集合条件方法已成立。** 独立逐条目scorer、集合条件scorer、future-attention、打乱标签和已有强基线应在同一预算/输入边界下比较，并加入能利用特殊副本外观的简单对照。先区分“避免丢唯一证据”与“选择哪个副本的精确MSE更低”，再用另一个未用于选择的新seed集合评价；独立学生已解决就如实报告集合条件不必要。本轮学生仍为零，这不是已完成的新方法实验。

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

## CV执行与审计记录

| 项目 | 当前状态 |
|---|---|
| A100作业、实际硬件与完成状态 | 作业11310：`COMPLETED 0:0`，307秒；CUDA与nvidia-smi匹配同一A100-SXM4-40GB UUID |
| GEOMETRY专用集 | benign重复保留0/48 episodes；distant为24/48，涉及6/12 quartets；逐行审计通过，不能与DEV分母混用 |
| Reader checkpoint及24项gate | VAL选择step1024；24/24通过，未用DEV选checkpoint |
| 同parent强baseline与oracle差距 | 见上表及CV完整逐行记录；尚无学生、新seed确认或新方法胜出 |
| 语义最优与reader最优 | 192/192最小集合语义正确；跨siblings共享精确/容差最优动作29/48 quartets |
| 完整归档与独立算术审计 | 本地ZIP SHA256/CRC通过；68,328项检查，零失败、零警告；不含独立模型推理复现 |
| 本轮新PM实验 | 无 |
| V6.1或FINAL边界改变 | 无 |

执行依据为CV的[调度记录](https://github.com/JasonLIN0226/CV_project/blob/phase2/independent-interfaces-20261020/results/literature_diagnosis/gate2-terminal-pilot-20261025/sacct-11310.txt)、[原始日志](https://github.com/JasonLIN0226/CV_project/blob/phase2/independent-interfaces-20261020/results/literature_diagnosis/gate2-terminal-pilot-20261025/slurm-11310.log)、[完整指标](https://github.com/JasonLIN0226/CV_project/blob/phase2/independent-interfaces-20261020/results/literature_diagnosis/gate2-terminal-pilot-20261025/summary.json)、[逐次删除](https://github.com/JasonLIN0226/CV_project/blob/phase2/independent-interfaces-20261020/results/literature_diagnosis/gate2-terminal-pilot-20261025/terminal_decisions.json)与[独立审计](https://github.com/JasonLIN0226/CV_project/blob/phase2/independent-interfaces-20261020/results/literature_diagnosis/gate2-terminal-pilot-20261025/independent_audit.json)。完整ZIP为345,671,988字节，SHA256 `523654adcb889726af0a8ce8a0dd3e59dba71b8c99e9634a2cd84ad09ed9be5c`。

以上数值始终属于CV开发证据；PM旧bank与V6.1各自保留原有证据范围。两仓库可以共用“可见→保存→留存→正确读取→任务改善”的检查链，尚不能合并成已验证的新memory方法。
