# CV reader 诊断与 PM 的联系（2026-10-25）

这是 CV 已有结果的连接说明，不是新的 PM 实验轮次。本次没有修改 PM 模型、输入、预算、阈值或 FINAL 状态，也没有运行 S0–S2、CHANGE 或 BF-Delta。

**CV 五臂 A100 消融现已完成。** 作业11544为`COMPLETED 0:0`，单张A100分配时长2010秒，即0.55833 GPUh。该时长包含训练、编码、评价及运行开销，不能当成纯训练耗时。原确认集仍关闭，没有训练保留学生，也没有运行任何新PM实验。

CV 的冻结源码为 [`82e31a0`](https://github.com/JasonLIN0226/CV_project/commit/82e31a0ecc555d9a9a7ffa6b13282dad00b9bef5)，[预注册](https://github.com/JasonLIN0226/CV_project/blob/82e31a0ecc555d9a9a7ffa6b13282dad00b9bef5/docs/READER_AGGREGATION_PREREGISTRATION_20261025.md)和[发布包](https://github.com/JasonLIN0226/CV_project/releases/tag/reader-aggregation-20261025)保留协议与数据来源。完成结果见[CV reader aggregation 结果报告](https://github.com/JasonLIN0226/CV_project/blob/phase2/independent-interfaces-20261020/docs/READER_AGGREGATION_RESULTS_ZH.md)；前置CPU检查另见[CV诊断报告](https://github.com/JasonLIN0226/CV_project/blob/phase2/independent-interfaces-20261020/docs/READER_AGGREGATION_AUDIT_ZH.md)。

本地 CPU 已完成两个检查，均仅使用旧 DEV 的 24 个独立场景族：

- 六种局部 payload/tag 的真实标签线性探针，在原始 I-JEPA 特征与冻结 128 维投影上均为 100%；空白输入均为 50%。这支持局部信息仍可访问，不证明配对关系或 XOR 已被 reader 正确利用。单次打乱标签对照部分高于随机，完整保留，不将其当成随机化显著性检验。
- 对原 checkpoint 强制均匀历史权重，R/D 完整 K2 分类提高，但完整预测 MSE 反而升高；S K3 从 72.48% 降至 65.10%。因此高 attention 偏向值得诊断，但不能断言去掉 softmax 竞争就能修好读取。

原来 S 的合格 K2 reader 下，medoid 的保留失败仍是有效结果；不能把全部保留失败重新归因于 reader。这里揭示的是两个必须分开的条件：世界模型能否使用某组证据，以及记忆规则是否据此做出正确决定。

五种reader使用同一TRAIN/VAL、冻结编码器和投影、纯MSE损失及1024次更新，训练seed均为17。全部模型和探针锁定后，才一次性评价新诊断seed20261120的24个独立场景族。下表是该新诊断集的结果，不是旧DEV重报，也不是原CONF或多训练seed确认。

| Reader | R/D/S全部K2门槛 | R/D/S全部K3门槛 | S完整K3准确率 |
|---|---|---|---:|
| original | 未全部通过 | 未全部通过 | 73.44% |
| attention_mean | 通过 | 未全部通过 | 79.69% |
| multihead | 通过 | 通过 | 96.96% |
| item_attention | 通过 | 通过 | 93.32% |
| wide_control | 未全部通过 | 未全部通过 | 72.57% |

门槛同时检查任务及预设外观条件，包括固定内容干预、原生时间与完整证据MSE要求；总体准确率超过80%并不自动通过。K3在S是两种XOR未来的特征原型识别，R/D是四种未来，不是物理轨迹成功率。multihead与item_attention在旧DEV和新诊断集均通过全部K2/K3门槛；参数量与multihead近似匹配的wide_control仍失败。这支持当前实现中的聚合方式是实际瓶颈之一，但不能推出单头softmax在理论上无法表示关系，也没有排除优化和归纳偏置的影响。

这次完成的是**CV reader的单训练seed诊断**。具备合格的K3 reader，使后续完整集合敏感度对照具有可检验的前提；本轮尚未用新reader完成保留baseline、保留学生或多seed确认，不能宣布新记忆方法胜出。此前在原S合格K2 reader下，medoid丢必要证据1103/1152的结果仍然有效；新reader资格通过既不抹去该失败，也不证明新reader下的medoid或敏感度已经解决保留问题。

对 PM 的启示是验证世界模型输出是否适合作为记忆决策信号。PM 继续遵循已有 L0 输入、同预算 memory-vs-memory 比较及 V6.1 协议；不据此另开编码器或 pooling 诊断轮次。
