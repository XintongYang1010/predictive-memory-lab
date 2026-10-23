# 已有PM案例与CV新确认结果怎样结合

这次PM部分只把已有BR-0.2及旧A2分析做成[四张案例卡](PM_CASE_GALLERY_ZH.md)，没有启动新PM训练、场景、感知轮次或V6 FINAL。另附[三篇论文阅读指南](PAPER_READING_GUIDE_ZH.md)。

CV侧的新结果见[受控诊断与独立确认汇总](https://github.com/JasonLIN0226/CV_project/blob/phase2/independent-interfaces-20261020/docs/GATE_CONFIRMATION_AND_CASEBOOK.md)：

1. 在旧开发场景的72个可比较parent上，固定时间和动作后仍有正确cue使预测更差的案例。最明显一例只替换正确内容，MSE从0.127822升至0.324210。模型给线索近乎全部attention却预测错方向；另有明确成功对照，不能说统一学反。
2. 冻结协议后另生成96对新场景；当前masked deletion-cost probe与shuffled-label控制差异区间跨零，且比attention、diversity分别高0.008909、0.007823 MSE，两区间均高于零。预设主要判断未通过，没有改调参数或挑另一个arm替代。

两仓库可以结合为一个诊断链：观察到证据 → 存下 → 留住 → 正确读取 → 任务改善。CV内容反例与PM有害旧值都说明“attention很高”不足以保证正贡献；PM噪声案例又说明hindsight删除标签可能追逐单次噪声。它们的具体原因不同，不是同一失败的重复实验，也不能直接合并MSE。

研究上仍需证明：在共享reader能正确使用证据的条件下，基于允许的过去信息，某种保留或更新策略是否稳定优于强对照。当前结果既没有建立删除价值监督优势，也没有证明所有因果策略都不可能成功。先把这个缺口验证清楚，再扩展方法。

旧A2的0.0125%账面修补空间仍不作为主线。若后续推进PM，保留原proposal的同输入、同字节预算、共享读取器与改写/保护配对边界；不能把丢掉有害值、返回零残差等同于恢复新状态。原始完整Z、动作、RGB与pose仍缺失，图册不会伪造这些观测。
