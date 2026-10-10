# V5.4 pre-generation web review

Posthoc 已完成并GitHub同步：commit4620d1b，结果目录 bad_cases/README.md。新重要线索：244/256pairs至少一cue坐标超出V53 FIT外包络，B逐坐标外/内MSE分别A_dx3.052e-4/9.247e-5、A_dy2.064e-4/4.008e-5、B_dx2.027e-4/8.894e-5、B_dy1.497e-4/6.783e-5；pair全in仅12，不能因果断言。水平cue反射仅10pair；3典型坏例425/0,454/1,579/0都无水平cue反射，6RGB均原输入hash吻合。零黄物遮挡；S pair-common residual与yellowdx r=.931/.893→B .236/.044。用户已有“insight就继续实验”授权，准备V54只改变FIT空间覆盖的一次新确认，尚未生成新数据：N复用V53FIT窄图像，W相同256 latent seed/quantiles remapcenter到覆盖原cue位置的wide安全范围（x约.118–.882,Ay.168–.382,By.618–.832，保留原d/colors/yellow）；N/W都16 balancedviews/每world同loss。为了公平samekernel，在N+W共8192FITviews上联合两级normalize，重新训练两个concat quadratic reader、同alpha7.8125e-7和freeintercept，不直接用旧B模型。新wideanchor256及原Dev640..895作fresh eval，先publish/download frozenmodels再Eval。PrimaryN−WdxMSE，pairedworldbootstrap10k；supportpreservation/negativecontrols。不改encoder/pooling、不Dev校准、不自动P0。只需FIT_W1536GPUimages一次、Eval_anchor+bridge2560images合在另一次8GPUstep，同型号同6minlimit；2×1.2GPUh envelopes，已用3.57393，最坏合计5.97393<诊断6，总60保留54。请快速审核此覆盖确认是否比直接P0更有针对性，检查共同unionkernel是否引入不可接受混杂或预算/新数据问题。此处只审设计，请勿生成任何新样本/运行实验/修改仓库。

---

已回读 [4620d1b 的 posthoc](https://github.com/XintongYang1010/predictive-memory-lab/blob/4620d1b/results/wm/visual-v5_3-20261010/bad_cases/README.md)。**这条新证据使 V5.4 比直接 P0 更有针对性；我赞成先做这一轮封顶的覆盖确认。共同 union kernel 没有不可接受的混杂，但结论必须限定为“在共同核几何下，扩大 FIT 空间分布的效果”。**

冻结前重点锁住以下几点：

1. **共同 normalization 的设计成立。** 两级 moments 只能来自 N+W 的 8192 个 FIT views；query 两级中心化都须用完整 union reference，再取各 arm 的训练列。两个 arm 各有 **4096 个监督 views，因此 ridge 对角项均为 `4096×alpha`**，不是 `8192×alpha`；保留 free intercept，不再按 arm 重新归一化。新 N 不是旧 B 的数值复现，不能把跨版本分数变化归给覆盖。

2. **“宽覆盖”要准确命名。** 给出的安全中心范围扩大了位置外包络，但没有覆盖原分布全部“位置×位移×反射”的联合支持，尤其靠边的小位移和反射轨迹。固定 256 worlds 扩大范围也会降低采样密度。因此，W 不改善不能直接否定覆盖问题；W 改善则支持这项训练分布干预有效，不能证明 encoder/pooling 已无瓶颈。目标中心移动还会改变与黄色物体的相对几何，这部分机制仍不能单独区分。

3. **映射必须完全确定。** 用原始 latent quantiles 映射中心，固定 d、颜色、yellow 和 view 权重；把“约 .118”等写成精确常数，明确舍入顺序，并验证 `C±d` 合法。不要通过裁剪、拒绝采样或补抽改变位移分布。N 缓存的图像、特征和提取配置身份也要绑定到收据。

4. **统计和 preservation 要预先说清。** 主指标继续固定 N−W dxMSE，按完整 world 配对 bootstrap，区间只反映冻结模型的 Eval 抽样不确定性。特别注意：244/256 是“任一坐标越界”，水平越界实际仅 A 的 61/512、B 的 70/512；必须同时报告四坐标及按旧包络固定的分组。wide anchor 本身不足以证明窄域能力保留：可预登记其中的窄域子组；旧 anchor 可用缓存作回归检查，但不能称为 fresh confirmation。若 preservation 要作为通过条件，容忍差须现在固定。

5. **新数据隔离合理。** W 配对复用旧 FIT latent 是有意控制，不是假装一次独立 FIT 重复。fresh wide anchor 必须独立 namespace；Dev640–895 仍需访问记录和实际 seed 碰撞审计。完整模型、union reference、moments、映射及协议回读冻结后，再一次性释放两套 Eval；不根据 anchor 表现跳过 bridge。

6. **预算算术成立，余量非常薄。** `3.57393+2×1.2=5.97393 GPUh`，诊断上限只剩 **0.02607 GPUh，约94 GPU秒**。必须确认两个 envelope 已覆盖全部启动和管理开销，并用已有吞吐记录判断合并的 2560-image Eval 能否装进六分钟。超时或失败要停；不能默认重试或借用保留的54 GPUh。

本轮最合适的可证伪问题是：**固定其余设计后，这一次空间分布扩展能否降低 fresh 原分布水平误差，同时保留目标响应、避免恢复 nuisance 响应？** 若没有明确收益，就不应继续无上限扩大 probe 诊断。无论结果如何，都不自动构成 P0 或 bounded-memory 通过。
