# 已有PM案例与CV新确认结果怎样结合

这次PM部分只把已有BR-0.2及旧A2分析做成[四张案例卡](PM_CASE_GALLERY_ZH.md)，没有启动新PM训练、场景、感知轮次或V6 FINAL。另附[三篇论文阅读指南](PAPER_READING_GUIDE_ZH.md)。

**2026-10-24 use-time 诊断更新：** CV 已完成 [TRAIN 标签审计](https://github.com/JasonLIN0226/CV_project/blob/phase2/independent-interfaces-20261020/docs/GATE_USE_TIME_LABEL_AUDIT.md)和[固定 reader 的 use-time 反事实实验](https://github.com/JasonLIN0226/CV_project/blob/phase2/independent-interfaces-20261020/docs/GATE_USE_TIME_PROBE.md)。在130个结果尚未显现、一次删除即可毁掉最后cue的训练parent中，原局部标签有111个偏好毁cue；固定attention继续到使用时刻后，130个的最优集合全部保护cue。新标签学生在旧TRAIN 32/32保cue，但192条旧开发场景仅74/192，MSE 0.125385，差于同容量local-cost学生的0.112316，也差于attention/diversity。标签修复没有转化为泛化收益。

这里还有一个重要混淆：CV reader原本只在query时刻训练，局部标签却在每次overflow调用它；use-time同时改变了价值时距和reader的评价起点。不能把全部收益归于延迟信用，更不能把CV的数值搬到PM。PM目前只借用诊断问题：在相同L0输入、相同预算和同一读取器下，一个条目的删除/改写代价，是在决策附近还是在再次使用时测量？这段更新没有新增PM实验、修改V6阈值、打开FINAL，或启动感知诊断；既有PM结论保持其原有证据范围。

**2026-10-24 展示更新：** [新版CV分组图册](https://github.com/JasonLIN0226/CV_project/tree/phase2/independent-interfaces-20261020/results/literature_diagnosis/visual-casebook-20261024)把内容失败与成功放在一组，时间戳只作辅助；另用已有WR启动trace展示“未接纳事件→随后完整丢失”，与“保存后reader利用失败”分开。所选WR例中t=8只丢掉首个cue帧，t=10才丢尽等价cue副本；62个救回案例仍有41个改善、21个变差。配合[逐案例论文阅读指南](https://github.com/JasonLIN0226/CV_project/blob/phase2/independent-interfaces-20261020/docs/CASE_STUDY_READING_GUIDE_ZH.md)阅读，不新增PM结果或改变下列结论。新版静态图已检查，自动文件浏览被安全机制拦截；源码检查不等于完成浏览器播放／交互验收。

CV侧的新结果见[受控诊断与独立确认汇总](https://github.com/JasonLIN0226/CV_project/blob/phase2/independent-interfaces-20261020/docs/GATE_CONFIRMATION_AND_CASEBOOK.md)：

1. 在旧开发场景的72个可比较parent上，固定时间和动作后仍有正确cue使预测更差的案例。最明显一例只替换正确内容，MSE从0.127822升至0.324210。模型给线索近乎全部attention却预测错方向；另有明确成功对照，不能说统一学反。
2. 冻结协议后另生成96对新场景；当前masked deletion-cost probe与shuffled-label控制差异区间跨零，且比attention、diversity分别高0.008909、0.007823 MSE，两区间均高于零。预设主要判断未通过，没有改调参数或挑另一个arm替代。

两仓库可以结合为一个诊断链：观察到证据 → 存下 → 留住 → 正确读取 → 任务改善。CV内容反例与PM有害旧值都说明“attention很高”不足以保证正贡献；PM噪声案例又说明hindsight删除标签可能追逐单次噪声。它们的具体原因不同，不是同一失败的重复实验，也不能直接合并MSE。

研究上仍需证明：在共享reader能正确使用证据的条件下，基于允许的过去信息，某种保留或更新策略是否稳定优于强对照。当前结果既没有建立删除价值监督优势，也没有证明所有因果策略都不可能成功。先把这个缺口验证清楚，再扩展方法。

旧A2的0.0125%账面修补空间仍不作为主线。若后续推进PM，保留原proposal的同输入、同字节预算、共享读取器与改写/保护配对边界；不能把丢掉有害值、返回零残差等同于恢复新状态。原始完整Z、动作、RGB与pose仍缺失，图册不会伪造这些观测。
