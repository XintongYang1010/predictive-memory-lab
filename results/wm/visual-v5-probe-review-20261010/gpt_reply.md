# Received GPT review

Source: ChatGPT conversation `6ac89457-ab14-83e9-85fb-745d12e0003d`, **分支 · Saining Lec6&实验**.
Message: `888ba774-01fa-57b7-96d8-68805b03d858`. This is the received review text, not independently verified evidence by itself. Local verification is in `local_validation.json`.

---

已读取最新 **`ca365e1`** 并完成审查，提交到独立分支：

- [review/v5-designer-probe-audit-20261010](https://github.com/XintongYang1010/predictive-memory-lab/tree/review/v5-designer-probe-audit-20261010)
- Commit：[`b13c9877339779b8c770d175a483ea3489d2420c`](https://github.com/XintongYang1010/predictive-memory-lab/commit/b13c9877339779b8c770d175a483ea3489d2420c)
- [完整审查](https://github.com/XintongYang1010/predictive-memory-lab/blob/b13c9877339779b8c770d175a483ea3489d2420c/docs/V5_PROBE_REVIEW_20261010.md) · [固定协议 JSON](https://github.com/XintongYang1010/predictive-memory-lab/blob/b13c9877339779b8c770d175a483ea3489d2420c/scripts/v5_temporal_confirmation_plan.json) · [CPU 辅助代码](https://github.com/XintongYang1010/predictive-memory-lab/blob/b13c9877339779b8c770d175a483ea3489d2420c/scripts/v5_temporal_confirmation.py)

**当前真正卡点是 gate 对“时间信息”的判别能力。**

可证事实是：正则化显著影响读出结果，但 pooled 双帧 **0.243895** 没有优于同预算单帧 **0.258056**；差值区间跨零。因此正位移 R² 不能让原门槛补算通过。Raw 也尚未建立“两帧必要”；query reversal 结果支持继续调查顺序一致性，但不能单独定位原因。

在核对的路径中，**没有发现新的致因实现错误**：时间帧、反射后的位移标签、池化轴、配对划分和 ridge 求解均未见能解释失败的错误。遗留 `velocity_r2` 名称及 Train-mean 分母需要准确说明，但不是本次失败的修复点。多输出 ridge 也不存在位置标签抢占位移拟合能力的问题。

仍未证实的是：池化损失、非线性读出需求和样本不足各占多少。不能据此断言 encoder 没有运动信息，也不能断言改善全部来自位置先验。

**新确认协议同时固定终点和起点。** 每个独立 block 生成三个图像位置 `C−d、C、C+d`，组成：

| t6 → t7 | 位移 | 控制 |
|---|---:|---|
| C−d → C | +d | 与下一行终帧完全相同 |
| C+d → C | −d | |
| C → C+d | +d | 与下一行起帧完全相同 |
| C → C−d | −d | |

同一 block 的外观、背景和干扰物固定，每张图像只编码一次再复用。分别计算前两行、后两行预测之差的一半，正确目标都是 `d`。

这个设计的关键是：

- **任何只看 t7 的函数**，在前两行必然输出相同，其终点对比分数精确为零。
- **任何只看 t6 的函数**，在后两行同理。
- 同一个冻结读出器必须同时通过两类对比，并准确预测绝对位移，防止相减掩盖共同偏差。

因此不依赖“几个单帧 baseline 恰好表现不好”，也不只依赖反转增强或符号反转。准确解码两帧位置再作差，属于合法的时间证据。

执行前固定的规则已写入协议：

1. 新 FIT、fresh confirmation 各 **256 个四样本 block**；只在 FIT 内分组 CV 选择预登记的读出家族和正则，随后冻结模型、代码及数据 hash。
2. **八个逐坐标对比分数，加绝对位移平均分数，共九项置信下界均须大于零**；位置分数仍须大于 0.1。区间按完整 block 重采样并做预登记调整。
3. 同一模型还须通过原分布的 fresh bridge：预定 `dev` pair **128–383**，保留反射；执行前核实这些 ID 未被查看。
4. 任一必要条件失败即停，不换赢家、不补种子、不降阈值，Test 保留。

**只有新确认与桥接均通过，才有理由进入新版本 P0。** P0 之后仍须通过正确历史相对 recent-only、swapped history 的检查，才能开始 memory 比较。

本次新增代码的 **11 项 CPU 软件检查通过**；fresh 科学数据尚未生成。没有启动 GPU、操作远端或写入 main、原 results、diagnostics 分支；冻结源码的 81 个原文件保持不变。
