# V5.3 web review and executor conclusion

Reviewed with the user-authorized original conversation `分支 · Saining Lec6&实验` (6ac89457-ab14-83e9-85fb-745d12e0003d). Review below is interpretation of the published artifacts, not a separate experiment.

已回读 [V5.3 固定报告](https://github.com/XintongYang1010/predictive-memory-lab/blob/d713ec9ad9444cb4350738cf4f8d6d2f2e65fd88/results/wm/visual-v5_3-20261010/README.md)、详细指标和冻结协议。**这轮证实了 balanced 监督对读出器选择性的修复；我的下一步建议是完成当前只读分解后，推进一个新登记、预算封顶的 P0/history gate，暂不直接展开 memory 架构比较。**

你补充的 contrast 解释是对的，而且应成为主要结论：

- **S 已有目标运动信号。** 同 nuisance 路径内做共享 end/start 差分，会消去共同的干扰响应，因此 S 可以同时表现为“contrast 很好、absolute moving motion 很差”。
- **B 的主要收益是抑制额外干扰响应，同时大致保留目标响应。** moving absolute skill 从 −.510 到 .698，nuisance-only false motion 大幅下降，而 target contrast 没有相应提高。bridge 中预测与真值的协方差也大致保留，过大的横向预测方差明显降低，与这个解释一致。
- 这支持“改善选择性”，**不证明误差严格等于一个可加的黄色运动项**；位置相关的交互仍可能存在。B_dx contrast 的小幅下降也不能未经等效性检验就称为“完全无损”。

本轮可确认：在相同冻结表征、共同 FIT normalization、固定正则和世界权重下，加入完整 nuisance-path 监督改善了**新 bridge 世界**的运动读出。主效应区间明确为正，四个运动坐标 skill 均为正，证据比 V5.2 的已观察样本干预更进一步。但这是一次 FIT 实现下的结果，world bootstrap 不包含重新抽取 FIT、重新训练的不确定性；也不能分离“增加跨帧监督约束”和“对称平衡”各自的贡献。encoder 未改变，因此无需假设 encoder 更新才能获得此次收益；这不等于 encoder、池化已没有瓶颈。

剩余约 20% 错向和幅度低估，目前至少有这些可区分解释：

| 解释 | 当前证据能说明什么 | 当前只读分解最有用的观察 |
|---|---|---|
| 残留 nuisance 响应或交互 | B 的负控 RMS 仍非零；中心 pose 的负控不覆盖全部空间布局 | 错误是否仍有 pair-common 成分，是否随黄色位移、相对位置变化 |
| FIT 空间支持不足、边缘或反射差异 | anchor 好、bridge 较弱与此相容，但不能据此定因 | 按**具体坐标的 t6→7 反射**、支持距离分解，同时检查位移幅度是否混杂 |
| 小位移的表征/栅格分辨率限制 | 较高 position skill 不保证能准确分辨两帧小差值 | 错向是否集中于小 \|displacement\|；该组贡献多少实际 MSE |
| 正则收缩、有限样本或条件均值预测 | 幅度比 .61–.68 与这些机制都相容 | 低估是否跨空间和幅度区间稳定，还是主要出现在不确定区域 |

**这些汇总不能区分 encoder、4×4 pooling、kernel 泛化和正则各自的责任。** 幅度比也不是一个现成的校准系数：正比例缩放无法修复错向。你们尚在制作的三例复核图，当前归档还不能支持逐例归因。

下一步我会这样取舍：

1. **先完成现有 posthoc，不因观察到相关性就立即加密空间特征。** 如果发现明确的覆盖缺口，优先固定表征、只扩展 FIT 空间支持；若需要判断池化瓶颈，再单独做匹配数据的粗/细特征对照。两者同时改会失去归因。
2. **若没有阻断性问题，优先测试实际预测任务是否能利用历史。** B 是有位置/位移监督的 quadratic reader，其成功不会自动转移给 P0 的未来 latent 预测目标。新 gate 应检查正确历史相对 recent-only、错误历史的收益，并同时看目标 ROI 和 paired future-response，避免只靠整图 latent MSE 判断。
3. **模型选择与最终 gate 验收分开。** P0 冻结后再用未访问的确认集验收；通过 history gate 后，才值得进行 bounded-memory 对比。

现有记录未显出足以推翻 S/B 比较的新实现矛盾。**V5.3 已取得真实的读出修复，但原失败记录仍成立，长期预测、bounded memory、P0 和 Test 都尚未得到本轮验证。**

## Executor follow-up

Posthoc analysis then found 244/256 worlds with at least one cue coordinate outside the narrow FIT coordinate envelope, and higher per-coordinate errors outside that envelope. These are descriptive support proxies, not causal evidence. Three selected bad cases have no horizontal cue-interval reflection; six re-rendered cues match actual encoded input SHA256 exactly. Yellow rectangles do not occlude target bounding boxes. The V5.3 result supports reduced nuisance response with preserved target contrast, not complete motion or memory repair. Old gates remain failed, Test is unopened, and this scene is retained.

A separate preregistered V5.4 comparison will test spatial coverage while holding encoder, pooling, kernel family, regularization, latent worlds and nuisance schedules fixed. It will use fresh evaluation and remain within the cumulative diagnostic budget; no V5.3 artifacts will be rewritten.
