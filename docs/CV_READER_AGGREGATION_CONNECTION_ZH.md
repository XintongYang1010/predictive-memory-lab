# CV reader 诊断与 PM 的联系（2026-10-25）

这是 CV 已有结果的连接说明，不是新的 PM 实验轮次。本次没有修改 PM 模型、输入、预算、阈值或 FINAL 状态，也没有运行 S0–S2、CHANGE 或 BF-Delta。

CV 的冻结源码为 [`82e31a0`](https://github.com/JasonLIN0226/CV_project/commit/82e31a0ecc555d9a9a7ffa6b13282dad00b9bef5)，[预注册](https://github.com/JasonLIN0226/CV_project/blob/82e31a0ecc555d9a9a7ffa6b13282dad00b9bef5/docs/READER_AGGREGATION_PREREGISTRATION_20261025.md)和[发布包](https://github.com/JasonLIN0226/CV_project/releases/tag/reader-aggregation-20261025)保留完整协议与数据来源。结果解释以 [CV 诊断报告](https://github.com/JasonLIN0226/CV_project/blob/phase2/independent-interfaces-20261020/docs/READER_AGGREGATION_AUDIT_ZH.md)为准。

本地 CPU 已完成两个检查，均仅使用旧 DEV 的 24 个独立场景族：

- 六种局部 payload/tag 的真实标签线性探针，在原始 I-JEPA 特征与冻结 128 维投影上均为 100%；空白输入均为 50%。这支持局部信息仍可访问，不证明配对关系或 XOR 已被 reader 正确利用。单次打乱标签对照部分高于随机，完整保留，不将其当成随机化显著性检验。
- 对原 checkpoint 强制均匀历史权重，R/D 完整 K2 分类提高，但完整预测 MSE 反而升高；S K3 从 72.48% 降至 65.10%。因此高 attention 偏向值得诊断，但不能断言去掉 softmax 竞争就能修好读取。

原来 S 的合格 K2 reader 下，medoid 的保留失败仍是有效结果；不能把全部保留失败重新归因于 reader。这里揭示的是两个必须分开的条件：世界模型能否使用某组证据，以及记忆规则是否据此做出正确决定。

五种 reader 结构的 A100 消融已按冻结协议提交为作业 11544，作业内已确认 NVIDIA A100-SXM4-40GB 且训练开始；截至这份连接说明写入时，尚无五臂训练结论。该实验只属于 CV，原确认集继续关闭。

对 PM 的启示是验证世界模型输出是否适合作为记忆决策信号。PM 继续遵循已有 L0 输入、同预算 memory-vs-memory 比较及 V6.1 协议；不据此另开编码器或 pooling 诊断轮次。
