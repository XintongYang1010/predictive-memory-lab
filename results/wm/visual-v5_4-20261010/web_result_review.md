# V5.4 result review with the original web conversation

Conversation: 分支 · Saining Lec6&实验 (6ac89457-ab14-83e9-85fb-745d12e0003d). These are external interpretations, not additional experiments.

## Sent evidence

V5.4已实际完成并同步GitHub，请复核当前bad case和结论，不修改仓库或生成任何新数据。报告 https://github.com/XintongYang1010/predictive-memory-lab/blob/6cc9396/results/wm/visual-v5_4-20261010/README.md ，scores/plots在同目录；source3279bbe，N/W complete model先发布并全回读后才释放新Eval，freeze Release visual-v5_4-fit-20261010。5steps415453.35–39均COMPLETED0:0；8H800继续共用IndexMem，当前累计4.522264341GPUh。独立10k world bootstrap精确复算：fresh Dev640..895 primary N−W dxMSE=-5.9110928768e-6，95%CI[-1.6792662421e-5,5.8798499493e-6]；N MSE1.1254426433e-4，W1.1845535721e-4（点估计恶化5.25%，CI跨0，不能说显著变坏）。motion skill N.353835→W.433999，secondary NMSE improvement.080164 CI[.037491,.125275]仅辅助；W四轴skills[.26784,.51584,.40869,.54362]，两个dx均略降，dy改善。W方向错率[24.02%,14.06%,18.75%,13.87%]，幅度[.589,.668,.589,.658]；negative overall RMS/FIT .23941→.23698基本持平。wide-anchor最小非零contrast N.50813/W.53215；预注册narrow-inside anchor87worlds N motion.65273/W.57945、dxMSE8.7713e-5→1.03558e-4；outside169则W更好，不能称narrow preservation。另本地CPU只读replay自身FIT（正在独立KRR正规方程核验）：N/W训练motion skill .999493/.999508、方向错误0，幅度约.986–.994，明显train/eval gap；不新fit不改alpha，posthoc标记。我的判断：V53平衡训练修复nuisance选择性，但V54这次扩位置范围不能修复水平泛化；纵向辅助收益不能替换主指标，下一步若推进应改变泛化机制/表示或数据密度设计，不能靠当前Dev调参声称通过。请评估这一结论、是否有遗漏的bad-case机制，以及当前是否应收束此覆盖路线。原gate/Test/P0未动，正在封存全现场。

## Web response

已回读 [6cc9396 的报告](https://github.com/XintongYang1010/predictive-memory-lab/blob/6cc9396/results/wm/visual-v5_4-20261010/README.md)、原始 scores 和分组 CSV。**你的主要判断成立，应收束这轮“固定256 worlds、直接扩大范围”的路线。** 更准确的表述是：本次干预未确认总体水平误差改善，不能据此断言所有空间覆盖方案无效。

有一个值得补进结论的重要细节：**W 的水平误差并非处处变坏，而是域外改善、域内退化，最终抵消。**

以下是 fresh bridge 的逐坐标分组，MSE 单位为 \(10^{-4}\)：

| 坐标 | 原窄域内：N→W | 原窄域外：N→W |
|---|---:|---:|
| A_dx | 0.968→1.163，186 worlds | 2.086→1.752，70 worlds |
| B_dx | 0.832→1.007，191 worlds | 1.403→1.157，65 worlds |

这与“扩大范围后，有限样本的空间分配发生取舍”一致，不能把本轮解释为覆盖完全没有作用。不过这些是预登记分组的描述性点估计，没有 subgroup CI，也尚未区分密度下降、相对几何改变和读出器泛化方式。[分组原表](https://github.com/XintongYang1010/predictive-memory-lab/blob/6cc9396/results/wm/visual-v5_4-20261010/analysis/spatial_group_metrics.csv)

剩余 bad case 的机制判断，我会补充三点：

- **问题已不只是 bridge 分布差异。** 按你报告、尚待独立核验的 FIT replay，训练 motion skill 接近 .9995；但同生成器的 fresh wide anchor，W absolute motion 只有约 .51–.56。即使没有原 bridge 的反射和轨迹差异，泛化缺口仍然很大。
- **横向弱项也不只来自移动黄色物体。** 在黄色静止的 aa/bb anchor 中，W 的 dx contrast 约 .53–.58，dy 约 .80–.82。这些 anchor 没有目标反射。因此，反射或黄色运动都不足以单独解释当前横向误差。[contrast 原表](https://github.com/XintongYang1010/predictive-memory-lab/blob/6cc9396/results/wm/visual-v5_4-20261010/analysis/anchor_contrast_metrics.csv)
- **接近完美的训练拟合不证明表征足够泛化。** 4096 个 views 仍来自256个 worlds，每个 world 仅六张独立图像；高容量 quadratic reader 可能很好地插值这些布局和外观，却没有学到跨布局共享的运动规则。它也不能区分有限样本、kernel 归纳偏置、池化损失和编码器位置敏感性。直接增加特征细度，可能同时增加可用信息和样本需求，未必解决问题。

另外，position skill 从 .949 到 .987 是有意义的辅助进步，但**位置预测改善没有转化成水平位移改善**；多输出 ridge 的位置列也不会“抢走”运动列的优化能力。负控总体接近不变还掩盖了 A_dx 变差、B_dx 变好的抵消，不能宣称 nuisance 响应进一步整体消失。

目前回读内容没有显出新的致命实现矛盾。你们正在做的独立正规方程核验值得完成；FIT replay 结果在归档核验前保留为暂定 posthoc。它若成立，更支持“训练插值良好、跨 world 泛化不足”，而不是用一个固定缩放系数就能修正输出。

**我的建议是完成封存，然后停止继续微调覆盖边界。** 若开启下一轮，应明确选择一个新问题：比较具有空间共享／局部对应约束的运动读出方式，或用范围×样本数的受控设计区分覆盖与密度；不要同时更换数据、特征和正则。纵向辅助收益应保留，但不能替代失败的主端点，也不能据此改写原 gate 或宣布 P0、memory、Test 已通过。

---

## Sent evidence

上一条 V5.4 结果复核在工具中只显示已提交的 userMessage、没有 assistant 回复且状态idle。请继续给出简短分析即可（不运行实验/改代码）：主dx无改善证据 CI跨0；dy改善、narrow-inside反而变差；训练motion .9995而Dev .354/.434。CPU训练预测已独立验证 y−.0032*coef，65536个数最大差1.73e-12，结果新增 https://github.com/XintongYang1010/predictive-memory-lab/blob/c18f104/results/wm/visual-v5_4-20261010/training_generalization/README.md 。此时应怎样界定已知bad case与未证实机制，是否应收束V5.4覆盖确认？无需重述全部指标。

## Web response

**应收束并封存 V5.4 覆盖确认。** 主假设未获支持，辅助收益和退化都保留；不需要继续调整范围来追求通过。

已回读[新增 FIT replay](https://github.com/XintongYang1010/predictive-memory-lab/blob/c18f104/results/wm/visual-v5_4-20261010/training_generalization/README.md)。结合你提供的独立数值核验，可以这样界定：

- **已知问题：跨 world 泛化缺口。** 训练样本拟合近乎完美，未见样本仍有明显错向和幅度低估；同生成器的 fresh wide anchor 也明显弱于 FIT，因此不能全部归因于 bridge 的反射或生成器差异。
- **已知取舍：域外改善、域内损失。** 分组中两个 dx 的域外点估计都有改善，但较多的域内样本退化，抵消整体收益。这支持报告“本次固定样本量扩域未确认总体水平收益”，不能报告“空间覆盖完全无用”。分组尚无独立显著性结论。
- **未证实机制：** 样本密度下降、对训练布局的插值、缺乏空间共享的读出规则、池化或 encoder 的运动分辨率限制，当前仍无法分开。正规方程核验确认训练回放数值一致，不能证明表征无损，也不能证明“记住布局”就是唯一原因。
- **下一步边界：** 不再延长本轮覆盖尝试。若继续，另立协议区分“范围与样本密度”，或检验具有空间共享／局部对应约束的读出器；先选一个因素，不直接认定更细特征必然有效。

V5.3 的 nuisance 选择性修复仍成立；V5.4 没有确认水平泛化进一步修复。原 gate、P0 和 Test 的状态保持不变。
