# Web ChatGPT methodological review

Read from conversation `6ac89457-ab14-83e9-85fb-745d12e0003d`, title `分支 · Saining Lec6&实验`. Advisory text is archived evidence of discussion, not a substitute for experiments or executable instructions. Frozen design was unchanged; interpretive limits are incorporated into the results report.

## Turn b567b44b-61ce-42d4-a2b1-ad9bc96543cd
### Sent context

新增已执行CPU证据已push+5文件逐字节回读： https://github.com/XintongYang1010/predictive-memory-lab/tree/6e9286343de3158e1e285421020b5de3272b3bde/results/wm/visual-v5_6-20261010 。明确GPU/JX/fresh64尚未启动，部署旧缓存传输中。1) global DF-match无改善；common-scale target去非目标分支仍MSE改善25.9–27.7%，original lane29.78%；只去2ab additive约6.3–6.7%，wrong-lane/checker更差。2) same raw fixedmatcher dxRMSE raw1.534/pool3.878；equalbyte ROI1.738/uniform3.753/random4.772/changepriority6.663。3) 新事后只读解释：ROI比uniform MSE降78.55%；whole-world256bootstrap CI差>0，约90%增益来自pool源点未clamp者。change-priority12raw细节cell源A/B覆盖仅.74%/.56%，黄条80.06%（footprint而非感受野）；ROI目标覆盖97%。4) 固定lane位置reader32B状态回放：latest/FIFO RMSE2.691/2.685，surprise3.124，localgate3.001；真变化2048次漏更新1375/842。阈值FIT-only固定95分位，无调参。局部细节路线有支持，naive surprise/局部范数gate作为本次prototype阴性保留。请结合你尚在进行的审查解释这些观察，尤其不要把成功的特权ROI或失败的固定matcher写成encoder充分/不充分的普遍结论。

### Response eea63d97-ca77-5c31-a325-28d65733128e

已回读 [6e92863 的五份报告](https://github.com/XintongYang1010/predictive-memory-lab/tree/6e9286343de3158e1e285421020b5de3272b3bde/results/wm/visual-v5_6-20261010)。**现在更有依据优先研究“目标相关的局部访问与细节保留”；两种基于变化范数的选择/更新原型则得到阴性结果。** 这仍不构成 encoder 充分或不充分的普遍判断。

1. **Lane 收益已不能仅用本轮检验的归一化、df 差异解释。**  
   common-scale target 保留约 26–28% 改善，错误区域和 checkerboard 反而差，说明访问范围与目标位置的对应关系有实际价值。只去掉 `2ab` 收益较小，说明跨区域乘积不是唯一限制：即使没有这些乘积，允许读出其他区域的项仍可能妨碍当前泛化。

   但这些模型均重新拟合，不能用“27%−6%”分配模块责任；一次 df 匹配也没有排除所有正则方案。报告中的预测幅度同时上升，因此收益并非简单把输出压小。

2. **局部细节路线得到支持，但成功对象是“表示＋特权定位＋固定匹配规则”的组合。**  
   raw 明显优于两种 pooled 重建；相同载荷的 ROI bank 又显著优于 uniform/random。这说明：**16 个描述子的保存预算本身，并不必然导致 uniform bank 那样的结果，保存内容及空间配置很关键。**

   仍不能推出 pooled 没有运动信息——已有 pooled lane reader 就提供了反例。ROI 的颜色先验也承担了对象定位工作，不能把它的成功全部记到 encoder 上。

   按你补充的事后分解，约 90% 收益来自源点未 clamp 的样本，削弱了“收益全部是源点边界截断造成”的解释；它尚未排除候选终点 clamp、插值与描述子几何的影响，更不是误差来源百分比。

3. **Change-priority 的失败提供了具体机制线索。**  
   你报告的 footprint 分布说明，该规则把细节预算集中到了黄色区域，几乎没有覆盖目标。报告还显示它有 **955/1024 个对象匹配落在搜索边界**，说明失败伴随严重的匹配失配，而非仅有轻微精度损失。

   二者共同支持“变化能量与目标运动读出需求不一致”。但 footprint 不等于感受野，触边也不能证明预测直接复制了黄条速度；空间重建仍参与其中。

4. **黄色残留不只是显式访问黄条所在格子的问题。**  
   [feature-path 报告](https://github.com/XintongYang1010/predictive-memory-lab/blob/6e9286343de3158e1e285421020b5de3272b3bde/results/wm/visual-v5_6-20261010/cpu/replay.json)中，仅替换不直接覆盖黄条的外侧行，lane 的 A/B 水平响应 RMS 已约为 **0.995/0.659px**。这为黄色变化通过保留区域的上下文特征影响读出提供了计算路径证据。

   因为这是混合特征干预，不能据此给 encoder 判错或分配责任比例。它更具体地提示：**空间访问限制仍需处理所保留 token 内的干扰响应。**

5. **门控阴性结果应保留，不能靠降阈值补救本轮结论。**  
   surprise/local gate 漏掉大量真实更新，而且报告中真实变化、未变化对象及 gap 的误差都没有换来优势。这足以否定这两个固定原型在本次回放中的实用收益；不能否定所有选择性写入。Latest/FIFO 几乎持平，也不宜把微小点差写成 FIFO 胜出。

两类失败形成了一个值得研究的共同假说：**按变化范数保存内容容易保留强干扰，按范数触发写入又容易漏掉较小的目标变化。** 后续优化更应围绕对象对应产生的任务相关证据，而不是把范数直接当作重要性或更新可信度。

GPU、单轴、J/X 和 fresh64 均仍待执行；这些 CPU 结果没有提前回答组合监督或轴间串扰问题，也没有形成 memory/P0 的通过证据。

## Turn 8d55c7f9-850d-406b-8931-40200bb83c9a
### Sent context

用户已要求实际分别试消融和三个方向。V5.6 已冻结源码并从 GitHub 全114个source blob逐字节回读，开始执行；请做一次只读方法学复核，重点挑能影响结论的混杂，不要把待运行设计当结果，也不要改seed/阈值或扩大任务。协议：https://github.com/XintongYang1010/predictive-memory-lab/blob/031788e0499c242a3671f57d026f3ed71da4872e/docs/V5_6_EXECUTION_PROTOCOL.md 。四组：common-map full/additive/target + correct/opposite/checkerboard native kernels，各原α/同FIT-only df；同raw导出的pool、固定物理位移matcher和四个等字节bank；同256worlds、J旧16视图重复2 vs X独立A/B符号32视图，匹配对象边缘，共同X8192reference与α，模型发布回读后才生成fresh64；旧512episodes补Ax/Ay/Bx/By像素2×2。另已冻结8×8转移、yellow token2×2替换，以及固定lane position reader的latest/FIFO/surprise/local-change门控32B状态回放。后两原型明确不作为长期memory或novelty证据。请审查：哪一种结果组合足以支持具体模块归因/下一步优化，哪一些即便改善也应保留解释边界？通过本对话回复建议即可，代码和最终结论仍经GitHub同步。

### Response ea73d838-4f91-59cd-b2e9-1aac8bf40476

已回读[冻结协议](https://github.com/XintongYang1010/predictive-memory-lab/blob/031788e0499c242a3671f57d026f3ed71da4872e/docs/V5_6_EXECUTION_PROTOCOL.md)及关键实现。**没有发现需要因本次审查改变冻结设计的致命方法学问题；但几种改善必须限定为“特定干预下可优化的环节”，不能变成单模块责任判定。** 以下是解释建议，不追加通过门槛。

从 [J/X 实现](https://github.com/XintongYang1010/predictive-memory-lab/blob/031788e0499c242a3671f57d026f3ed71da4872e/scripts/v5_6_combination.py)看，重复行折叠后的缩放正确：J 的 `.0032` 与 X 的 `.0064` 对应相同 mean-loss alpha，不能把两者对角项不同当成正则实现错误。

1. **Reader 消融能定位函数类与估计方式，不能彻底分离访问和几何。**

   | 待观察的结果组合 | 足以支持的判断 | 仍不能推出 |
   |---|---|---|
   | additive 在原 alpha、df 匹配下都优于 full | 限制跨区域二次项具有可重复的诊断收益，值得优化读出结构 | 跨区域交互本身必然有害、所有正则解释已排除 |
   | target_common 进一步改善，correct 优于 opposite/checkerboard | 与对象位置相符的访问范围有价值，超出单纯减少维度的解释 | 已学会对象身份、已隔离全部核几何影响 |
   | native correct 改善，target_common 不改善 | lane 收益明显依赖重新归一化或惩罚几何 | 仅裁掉其他区域就能修复 |
   | df 匹配后差异明显缩小 | 总体复杂度解释得到支持 | 自由度相同意味着逐方向收缩相同 |

   特别注意：**additive 去掉的是跨空间区域的乘积，仍保留区域内部的跨帧项。** 不能将它的改善解释成“时间交互没有用”。这些仍是已见 Dev 上的固定模型比较，不能从十二个控制中挑赢家再称独立确认。

2. **单轴与 token 干预可以定位响应路径，但属于不同层面的证据。**

   单轴表最有辨识力的组合是：

   - x-only 正确、xy 错误，且 y-only 的水平响应或交互项抵消 x-only：支持这次二维运动中的轴间串扰。
   - x-only 已经弱或错：水平读出问题在没有纵向变化时就存在，不能再主要归咎于 dy。
   - y-only 很小，但交互项很大：问题是联合响应，不能用单轴主效应相加解释。

   这些都是整个 encoder→pool→reader 的有限像素干预；若某个冻结 reader 对相同特征改善，只能进一步定位到下游存在可改善因素。

   token 2×2 的强证据组合是：**真实黄色干预改变干净外侧行的特征，且替换该行会改变目标输出**。这支持“上游上下文响应经该特征路径被 reader 使用”。单有特征变化不能证明有害；单有混合 token 效应也不能证明真实图像上的自然中介关系。

   这里还有一个命名边界：top/bottom **外侧一行未必覆盖目标本体**，应叫外侧行路径，不能直接叫目标 token。现有完整预测足够比较同一输出对 top 与 bottom 替换的反应；不要直接把“上行对 A”与“下行对 B”的两个汇总值当成等大小空间对照。混合状态的分布外风险和交互项仍须保留。

3. **Matcher 与 bank 中，插值和存储内容要分开解释。**

   raw_native 与 pool_native 除了池化，也使用各自网格上的原生插值。因此 raw 更好只能支持：**池化及其空间重建方式削弱了该 matcher 可访问的线索**，还不是不可逆运动信息损失。

   本轮已经有一个很有用的内部对照：**pool_native 与 uniform_budget 存的是同一组池化描述子，只改变重建方式。** 如果二者差距很大，说明解码几何本身值得优先优化，不需要新增实验来发现这一点。

   - raw、pooled 匹配都好，而 KRR 差：支持读出方式未充分利用可访问线索。
   - raw 好、两种 pooled 重建都差：加强池化相关限制的证据，但结论仍限于这些匹配规则。
   - 都差：无法区分编码表示限制与局部平移/余弦匹配假设不适用。
   - ROI bank 优于 random：支持在相同粗细描述子配置下，带颜色先验的空间分配有价值。
   - ROI bank 优于 uniform：同时涉及分辨率配置和位置选择，不能只归因于“自适应选择”。

   change-priority 使用两帧变化，属于离线选择。等字节成立于**保存的 bank 载荷**，不包含选择时访问完整 raw、ROI 信息和工作内存，不能据此声称端到端在线内存优势。

4. **J/X 能确认组合监督扩充的作用，但并非等信息量训练。**

   对象边缘分布匹配、共同 X reference 和相同平均损失正则，使这个对照有针对性。不过 **J 的重复行与 X 的新联合视图并不提供相同的信息量**。X 同时增加了独特联合图像/监督组合，所以改善可归因于这套组合监督扩充，不能进一步宣称“方向独立性本身优于等量其他新增样本”。

   最有说服力的结果是：fresh64 主指标改善，并且分项显示收益不只来自 mixed16、自身响应保持、跨目标假运动下降。这里各分项用于解释，不改变登记的 primary：

   - 只有 mixed16 改善、old16 退化：支持覆盖收益与原区域代价并存。
   - 已见 bridge 改善、fresh64 不改善：保留探索性现象，没有取得这次组合确认。
   - fresh64 改善：确认范围仍是该组合生成器；没有确认原 bridge 分布、单轴解耦或长期记忆。

   两臂仍使用一个固定的 256-world FIT；world bootstrap 不包含重新抽取训练集的变异。

最后，**8×8 转移与 32B 回放需要防止两个容易产生“漂亮结果”的因素**：

- 重复帧零运动、反转反号可能来自已经验证的结构恒等式；64 条转移也改变了零运动样本比例。不能把其汇总 skill 与原始 `000→111` 直接当作同难度提升。
- 回放中 latest 与 FIFO 同时区别于复制最新值和两值平均；门控又引入延迟。只有在减少无变化对象误差的同时，真实变化步骤没有明显丢更新或持续滞后，才支持这个小场景中的更新策略收益。大量拒写造成的平滑不等于可信更新。

32B 是状态载荷，代码还依赖两帧特征/预计算候选；静止 hold gap 测的是保持状态，没有测试遮挡期间动力学。即便改善，也应保持协议中“小型接口压力测试”的定位。

本轮最值得优先联合阅读的是：**common-map 结果＋单轴分解＋真实黄色/特征路径响应**。它们能决定下一步应优先改读出函数、轴向对应还是上下文抑制；J/X fresh64 则独立回答组合监督是否有效。无需因此改 seed、阈值或扩大本轮任务。

