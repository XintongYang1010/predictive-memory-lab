# 已有PM案例与CV新确认结果怎样结合

这次PM部分只把已有BR-0.2及旧A2分析做成[四张案例卡](PM_CASE_GALLERY_ZH.md)，没有启动新PM训练、场景、感知轮次或V6 FINAL。另附[三篇论文阅读指南](PAPER_READING_GUIDE_ZH.md)。

**2026-10-24 A100 reader 规模 pilot 已完成：** CV 在 A100 job 11026 上，保持同一 `DynamicsPredictor` 架构、旧 I-JEPA 特征、归一化和 basis，比较嵌套的16/64/512个训练场景对。三臂使用相同初始化seed17和512次最大更新预算，以共同的新32对validation选checkpoint，再评估新48对场景（ID、long gap、distractor shift各16对）。这是对reader能力的测试，尚不是保留策略的胜出。协议、硬件与限制见[reader规模报告](https://github.com/JasonLIN0226/CV_project/blob/phase2/independent-interfaces-20261020/docs/GATE_READER_SCALE_PROBE.md)，逐条件统计和图见[结果目录](https://github.com/JasonLIN0226/CV_project/tree/phase2/independent-interfaces-20261020/results/literature_diagnosis/reader-scale-pilot-20261024/analysis)。

| 同一cue＋filler输入 | Cue MSE | 正确cue比matched错误cue更差 | 至少一个horizon的alpha为负 |
|---|---:|---:|---:|
| 原先的16对reader | 0.108929 | 24/96 | 34/96 |
| 新重训16对 | 0.104284 | 19/96 | 33/96 |
| 新重训64对 | 0.073858 | 2/96 | 10/96 |
| 新重训512对 | 0.057564 | 2/96 | 4/96 |

这里的96条episode只有48个独立场景对；alpha是沿真实未来与sibling未来之差的**特征方向投影**，不是真实轨道解码准确率。512对相对新重训16对的cue MSE降低约44.8%，三条件的平均MSE都降低。剩余“正确cue更差”的两条来自同一场景对。该结果说明在这批新场景上，旧reader的证据利用失败能随训练数据扩展而大幅减轻；没有逐个重放修复旧图册的所有具体案例，也没有证明失败仅由某个动作区间覆盖不足造成。三臂最终选中步数是128/448/448，且只有一个训练seed；因此只能评价这个固定训练与选择协议，不能把全部差异视为单独控制了优化年龄的样本量因果效应。

**扩大后的student可行性测试【待完成】：** 接下来固定512对reader，使用512对TRAIN比较local-cost、use-time、future-attention与shuffle学生。当前48对evaluation的reader结果已经看过；后续在它们上的学生比较只能标为开发可行性分析，不能重新称为独立确认。尚无这一步的完成结果或泛化结论。下面的旧CPU学生负结果保留，不能用“reader变强了”将它改写为成功。

**2026-10-24 旧16对TRAIN／旧reader的CPU use-time 负结果：** CV 已完成 [TRAIN 标签审计](https://github.com/JasonLIN0226/CV_project/blob/phase2/independent-interfaces-20261020/docs/GATE_USE_TIME_LABEL_AUDIT.md)和[固定 reader 的 use-time 反事实实验](https://github.com/JasonLIN0226/CV_project/blob/phase2/independent-interfaces-20261020/docs/GATE_USE_TIME_PROBE.md)。在130个结果尚未显现、一次删除即可毁掉最后cue的训练parent中，原局部标签有111个偏好毁cue；固定attention继续到使用时刻后，130个的最优集合全部保护cue。新标签学生在旧TRAIN 32/32保cue，但192条旧开发场景仅74/192，MSE 0.125385，差于同容量local-cost学生的0.112316，也差于attention/diversity。标签修复没有转化为泛化收益。

**文献定位更新：** [VitaBench / VitaMem（2026-10-10）](https://arxiv.org/html/2610.14731v1#S5)已用长时距反事实动作标签训练只看过去的策略；[MGPO（2026-09-29）](https://arxiv.org/html/2609.37930v1#S2)也优化记忆改写的延迟增量效用。因此use-time只能先作为proposal内的适配与机制检验，不能单独宣称新颖。完整判断见[CV可行性报告](https://github.com/JasonLIN0226/CV_project/blob/phase2/independent-interfaces-20261020/docs/GATE_DELAYED_CREDIT_FEASIBILITY_ZH.md)。这不是新增PM结果。

这里还有一个重要混淆：CV reader原本只在query时刻训练，局部标签却在每次overflow调用它；use-time同时改变了价值时距和reader的评价起点。不能把全部收益归于延迟信用，更不能把CV的数值搬到PM。PM目前只借用诊断问题：在相同L0输入、相同预算和同一读取器下，一个条目的删除/改写代价，是在决策附近还是在再次使用时测量？这段更新没有新增PM实验、修改V6阈值、打开FINAL，或启动感知诊断；既有PM结论保持其原有证据范围。

**2026-10-24 展示更新：** [新版CV分组图册](https://github.com/JasonLIN0226/CV_project/tree/phase2/independent-interfaces-20261020/results/literature_diagnosis/visual-casebook-20261024)把内容失败与成功放在一组，时间戳只作辅助；另用已有WR启动trace展示“未接纳事件→随后完整丢失”，与“保存后reader利用失败”分开。所选WR例中t=8只丢掉首个cue帧，t=10才丢尽等价cue副本；62个救回案例仍有41个改善、21个变差。配合[逐案例论文阅读指南](https://github.com/JasonLIN0226/CV_project/blob/phase2/independent-interfaces-20261020/docs/CASE_STUDY_READING_GUIDE_ZH.md)阅读，不新增PM结果或改变下列结论。新版静态图已检查，自动文件浏览被安全机制拦截；源码检查不等于完成浏览器播放／交互验收。

此前CV旧reader的受控诊断与冻结确认见[汇总](https://github.com/JasonLIN0226/CV_project/blob/phase2/independent-interfaces-20261020/docs/GATE_CONFIRMATION_AND_CASEBOOK.md)：

1. 在旧开发场景的72个可比较parent上，固定时间和动作后仍有正确cue使预测更差的案例。最明显一例只替换正确内容，MSE从0.127822升至0.324210。模型给线索近乎全部attention却预测错方向；另有明确成功对照，不能说统一学反。
2. 冻结协议后另生成96对新场景；当时冻结的masked deletion-cost probe与shuffled-label控制差异区间跨零，且比attention、diversity分别高0.008909、0.007823 MSE，两区间均高于零。预设主要判断未通过，没有改调参数或挑另一个arm替代。

两仓库可以结合为一个诊断链：观察到证据 → 存下 → 留住 → 正确读取 → 任务改善。CV内容反例与PM有害旧值都说明“attention很高”不足以保证正贡献；PM噪声案例又说明hindsight删除标签可能追逐单次噪声。它们的具体原因不同，不是同一失败的重复实验，也不能直接合并MSE。

新增A100结果给这条链增加了一个必要检查：先确认共同reader确实能利用提供给它的证据，再把保留/改写决策的差距解释为memory问题。对PM，这只改变分析顺序：按现有规则使用oracle L0输入，在同输入、同字节预算、同一读取器的memory比较中检查；不照搬CV的encoder或reader规模实验，不据此开启新的感知诊断，也不改变V6阈值或FINAL边界。CV规模pilot、旧CPU学生负结果和PM既有BR-0.2结果是三个不同层级的证据。

研究上仍需证明：在共享reader能正确使用证据的条件下，基于允许的过去信息，某种保留或更新策略是否稳定优于强对照。当前结果既没有建立删除价值监督优势，也没有证明所有因果策略都不可能成功。先把这个缺口验证清楚，再扩展方法。

旧A2的0.0125%账面修补空间仍不作为主线。若后续推进PM，保留原proposal的同输入、同字节预算、共享读取器与改写/保护配对边界；不能把丢掉有害值、返回零残差等同于恢复新状态。原始完整Z、动作、RGB与pose仍缺失，图册不会伪造这些观测。
