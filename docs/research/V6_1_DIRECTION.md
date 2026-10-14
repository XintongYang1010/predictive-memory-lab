# 用贝叶斯因子教世界模型的记忆何时改写

V6 应选 **Bayes-factor 门控的残差快权重记忆**（下称 BF-Delta）。它由两个预测器和一个比值构成：一个冻结的、无记忆的世界模型预测器充当先验，这正是 Cambrian-S 的 LFP head 扮演的角色；一个测试时按 delta rule 写入的快权重矩阵充当"本 episode 的世界"这一信念。两者对下一帧潜变量的似然比给出闭式的变化后验 γ，再按 Nassar 形式 β = γ + (1−γ)τ 决定每个 key 方向上的写入强度；同一个对数比值在计数时充当"见过没有"的熟悉度否决。这个方向同时满足用户的全部约束。第一，研究对象就是记忆的保持、纠正与保护算子，不碰感知。第二，TTT 是定义而不是外挂：每次写入就是对记忆自身预测损失做一步闭式步长的 TTT。第三，它直接修复 Saining 组已公开承认的 VSC-Repeat 失败：MRA 从 42.0 跌到 20.6、3.6、0.0。第四，只用 MIT 许可的 Memory Maze 离线数据集（主）和 Apache-2.0 的 VSI-SUPER-Count（次）。第五，全程估算约 140 GPUh，8×A100 两天内跑完，需要训练的只有一个 key 投影和一个冻结前训好的小预测器。新意必须说窄。Liakoni/Nassar 的变化点代数、MOCA 的"元学习先验 + 快统计量 + 变化检测"、SEM 的记忆参照分段都已存在。截至 2026-10-13 的检索未发现两件事：有人把冻结的世界模型预测器当作 BF 先验、把快权重当作当前信念并逐 key 闭式写入；有方法论文报告 VSC-Repeat 的修复。上一版"value 侧线性生成元"草案退为 V7 候选，因为没有带明确许可的公开数据集同时提供动作、重访和视野外自主运动的逐帧真值。RSI 不进入 V6。

## 六个候选只有一个同时过七道约束

**判据有七条，全部来自用户原话和 AGENTS.md：** 聚焦 world model 记忆而非感知；TTT 或 RSI 接法自然；小范围且截至 2026-10-13 未被做过；接得上 Saining 的研究；公开数据集；≤8×A100 数天；训练不难、算法简单。下表逐条对照。"新意核验"一栏的检索有明确盲区：Semantic Scholar 引用图、OpenReview 上的 ICLR 2027 投稿和 arXiv 列表页都无法访问，所以"未发现"只表示"在可达来源中未发现"。

| 候选 | 记忆核心 | TTT/RSI 接法 | 新意核验（截至 2026-10-13） | Saining 接口 | 公开数据 | 算力/难度 | 裁决 |
|---|---|---|---|---|---|---|---|
| 旧 V6 草案：value 侧线性生成元的 predictive associative memory | 最贴 AGENTS 研究对象 | delta 写入即一步 TTT；用揭示误差调生成元为次要 | 部分占用：问题已被 LiveWorld、FloWM、Shell Game 提出；只剩"value 侧、含 shear、在无观测步运行"的窄机制 | 只有类比，不跑真实管线 | **不合格**：L0 是按 FloWM 规格自建的生成器，L1 是 Miniworld 重生成；FloWM 的 HF 数据集未声明许可 | 极小 | 退为 V7 |
| C1：TTT 预测器作 surprise 源 | 中：surprise 驱动分段 | 自然 | **机制已占**：Aakur & Sarkar 2019、STREAMER 2023、FFN 2606.26515、Titans；在 Cambrian-S/VSC-Repeat 上的应用开放 | 强 | VSI-SUPER | 小 | 并入：作计数读出 |
| C2：TTT 作潜空间 WM 记忆 | 强 | 自然 | **大体已占**：DexWorldModel、Shell Game、CoME（已用 Memory Maze）、JEPA-TTT | 弱 | Memory Maze | 小 | 并入：作基底 |
| C3：RSI 记忆时长自举 | 中 | 生硬 | 组件已占（Lee et al.、WorldCycle、RELIC），且被直接长跨度训练和按准确率门控的课程支配 | 弱 | 可 | 最贵 | 不做 |
| C5：快到慢巩固 | 中 | 半自然 | 语言模型中已做（Behrouz 2606.03979、Nested Learning）；WM 中未见；平稳设定下等价普通训练 | 弱 | Memory Maze | 小 | 留 V7 |
| BF 门控快权重（原始形式 β = γ） | 强：写、保护、熟悉度用同一量 | TTT 闭式步长 | 原理部分占用，具体实例未见；但原式有语义、冷启动、阈值三处错误 | 强：LFP 即先验 | 两者皆可 | 小 | **修正后入选** |

**旧草案输在数据上，不输在想法上。** 它的核心主张是"看不见的对象按动力学继续演化"，这需要视野外自主运动的逐帧真值。数据调研的结论是：没有公开数据集同时具备动作、重访、视野外移动对象的逐帧真值和明确许可，FloWM 的 MNIST World 与 Block World 最接近，但 HF 卡片未声明许可（[FloWM MNIST World](https://huggingface.co/datasets/flowm123/mnist-world)）。Memory Maze 的地图和物体在 episode 内静止，value 侧生成元在那里只能推进自运动，无物可演化。新意也已收得很窄：VO-RoPE/GTA 是固定 value 旋转，KDN、MbD 是 key 侧 Kalman 预测，Koopman Dreamer 已做动作条件线性潜动力学的无观测 rollout（[KDN](https://arxiv.org/abs/2609.07816)；[MbD](https://arxiv.org/html/2605.31163v4)；[Koopman Dreamer](https://arxiv.org/abs/2607.19719)）。上一轮的引理 3 还证明，它等价于"静态 delta 记忆 + 一个共享坐标系"，表达力主张站不住。如果用户之后接受"公开但未声明许可"的 FloWM 数据，它可以原样回到 V7。

**C1 与 C2 都已被占住机制，只剩应用。** "在线训练的预测器、以误差峰值切事件"在 2019–2023 年已有完整谱系。Aakur & Sarkar 的门控规则是"预测误差超过平均质量 50% 即触发"（[CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Aakur_A_Perceptual_Prediction_Framework_for_Self_Supervised_Event_Segmentation_CVPR_2019_paper.pdf)）；STREAMER 在 EPIC-KITCHENS 上用多层预测误差峰值建事件层级（[NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8f0d446441a938d9de420a8ab8d7fd36-Abstract.html)）；FFN 在长视频上以 μ+σ 阈值决定是否做 TTT 步（[2606.26515](https://arxiv.org/html/2606.26515v1)）。"TTT 快权重作世界模型记忆"同样已有先例：DexWorldModel 用 TTT 层替换预测冻结 DINOv3 特征的因果潜世界模型的 KV cache（[2604.16484](https://arxiv.org/abs/2604.16484)），Shell Game 用 LaCT 在动作条件视频 WM 上追踪隐藏状态（[2608.30692](https://arxiv.org/html/2608.30692v1)），CoME 用测试时 LoRA 微调作长期记忆，Memory Maze 上 LPIPS 从 0.209 降到 0.097（[2605.18813](https://arxiv.org/html/2605.18813v1)，会议归属未核实）。两者单独做都只剩"评测型"贡献，审稿人会称之为"TTT3R for world models"。它们留下的真正开放点是：**写入步长和计数决策应该参照什么。** BF-Delta 正好补上这一点。

**C3 是最像追热点的一项。** 在 Memory Maze 和 VSI-SUPER 上，每个时长都有真值，自标注不带来新信息；按准确率门控的课程能把固定状态递归的召回从 0.021 提到 1.000，6/6 个种子锁定，不门控的斜坡为 0/6（[2609.16183](https://arxiv.org/html/2609.16183v1)）。**C5** 的"白天快权重、夜里蒸馏进慢权重"在语言模型里已有 Behrouz 等人的 sleep 方案（[2606.03979](https://arxiv.org/html/2606.03979)）。在平稳数据上，它与直接训练慢权重拿到的信息相同，只有在不保存原始 episode 的持续学习协议下才有意义，适合在 V6 给出可用的快权重记忆之后再做。

**BF 门控的原始形式有三处必须修。** 第一，Liakoni 等人推导的 γ = mS/(1+mS) 是"整合当前信念"与"重置到先验"两个后验之间的**混合权重**，不是学习率；即使 γ = 0，整合分支仍以约 1/(n+1) 的速率学习（[Liakoni et al. 2021](https://lcnwww.epfl.ch/gerstner/PUBLICATIONS/Liakoni21NeuralComputation.pdf)）。所以 β = γ 实际是 SMiLe 式近似（[Faraji et al.](https://arxiv.org/pdf/1606.05642)），不是 Liakoni 式。第二，没写过的 key 有 Wk = 0，于是 S_BF = 1、β = p_c：每个场景第一遍都只以 hazard rate 写入，即冷启动问题。第三，"阈值为零"对应 p_c = 0.5；真正的贝叶斯判决边界是 log((1−p_c)/p_c)，p_c = 10⁻³ 时约 6.9 nats。修正后的形式见下一节和规格部分。

## BF-Delta：已做的、仍开放的、能守住的一句话

**选定理由。** 它把 C1 的"surprise 驱动分段"、C2 的"快权重即 WM 记忆"和 BF 门控合成一条递推。递推里只有一个新量，即"记忆相对世界模型的对数似然比"，而这个量同时承担三件事：写多少（保护与纠正）、哪里变了（变化检测）、是否见过（计数去重）。与 Saining 的接口不是拼贴出来的：Cambrian-S 自己写道"surprise guides attention, memory, and learning"，实现里却只用于记忆管理和分段（[Cambrian-S Sec. 1, 4.1](https://arxiv.org/html/2511.04670)）。BF-Delta 保留它的无记忆 LFP head，作为先验；让 surprise 去设定一个测试时学习器的学习率，补上缺失的"learning"那一臂；再用记忆参照的 surprise 修 VSC-Repeat。

| 已被做过（必须引用、不得声称） | 截至 2026-10-13 仍开放（检索未发现） |
|---|---|
| BF surprise 与 γ = mS/(1+mS)（[Liakoni 2021](https://lcnwww.epfl.ch/gerstner/PUBLICATIONS/Liakoni21NeuralComputation.pdf)）；Nassar 的 α = Ω + (1−Ω)/(r+1)（[Nassar 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2945906/)，公式本轮未能重读）；SMiLe | 以**冻结的无记忆世界模型预测器**为变化点先验、以**测试时快权重**为当前信念、写入强度取闭式变化后验，并**逐 key 局部**作用 |
| 元学习先验 + 快 Bayesian 统计 + BOCPD 重置，并可穿过变化检测反传（[MOCA](https://ar5iv.arxiv.org/html/1912.08866)）；网络权重向先验软重置（[Galashov 2024](https://arxiv.org/html/2411.04034v1)）；可变遗忘因子 RLS 与定向遗忘（[forgetting 文档](https://bayesianbandits.readthedocs.io/en/latest/math/forgetting.html)） | 同一个对数比值作为 Cambrian-S 管线内的熟悉度否决，修复 VSC-Repeat；没有任何方法论文报告 VSC-Repeat 结果，2511.16655 只给基准层面的建议 |
| 记忆参照分段与事件复用（[SEM 2020](https://www.biorxiv.org/content/10.1101/541607.full.pdf)）；上下文条件 surprise 切段（[EM-LLM](https://arxiv.org/html/2407.09450v3)）；with/without 上下文的对数比用于解码（[CAD](https://arxiv.org/pdf/2305.14739)）；新旧模型对数比作为 learning progress（[Achiam & Sastry](https://ar5iv.labs.arxiv.org/html/1703.01732)） | Memory Maze 离线数据上的 CHANGE/REPEAT/NEWPATH 拼接协议，配官方探针，在等预算写入规则之间画稳定性–可塑性前沿 |
| "只存世界模型预测不了的东西"（[MemoWM](https://arxiv.org/html/2610.10778v1)；[EMWM](https://memari-workshop.github.io/papers/paper_3.pdf)）；快权重学残差（[TTT-RM](https://arxiv.org/abs/2610.04701)）；按残差缩放写入步长（[T²Mem](https://arxiv.org/html/2609.36720v1)） | 闭式门控能泛化到**训练中从未出现**的变化点（Memory Maze 的 episode 内静止），并与"在变化数据上训练过的学习门控"做等预算对照 |
| TTT/快权重作 WM 记忆（DexWorldModel、Shell Game、CoME、JEPA-TTT）；Kalman 快权重（KDN、MbD、KLA、GKA） | — |

**最可辩护的一句话主张：**

> 我们把 Liakoni 等人的 Bayes-factor 变化点后验实例化在一个潜世界模型里：冻结的无记忆预测器（Cambrian-S 中 LFP head 的角色）是变化点先验，测试时快权重记忆是当前信念。在等状态、等输入、等损失的记忆对照中，所得的逐 key 闭式写入强度保护一致的内容不被改写，并在从未训练过的变化上快速重置；同一个对数比值抑制了无记忆 surprise 无法避免的重访重复计数（VSC-Repeat）。

这句话必须带四个限定。第一，主张"实例化与评测"，不主张"新的 surprise 度量"或"第一个 surprise 调制的记忆"。第二，用 Nassar 忠实形式和宽先验，β = γ 只作消融。第三，不说"无阈值"，说"一个按单位时间计的 hazard rate，跨视频长度共用"。第四，相关工作必须引 MOCA、软重置、SEM、可变遗忘 RLS 和 learning progress。审稿人最可能的一句话归并是"单一 run length 的 MOCA 加 delta-rule 记忆"，回应只能靠逐 key 局部性、分段复用和世界模型记忆指标上的结果。

## V6 规格：一条五行递推、两个公开数据集、十个等预算对照臂

### 方法：两个预测器、一个比值

**记号。** 冻结编码器给出潜变量 $z_t\in\mathbb R^D$，$a_t$ 是动作。慢模型 $f_\theta(z_t,a_t)$ 预测 $z_{t+1}$：它无记忆，在训练集上用 MSE + 余弦损失训练后冻结，与 Cambrian-S 的 LFP 损失相同（[Cambrian-S](https://arxiv.org/html/2511.04670)）。噪声方差 $\Sigma=\mathrm{diag}(\sigma_d^2)$ 用于白化。每个头的状态是 $W\in\mathbb R^{D\times d_k}$ 和每个 key 维度上的有效计数 $n\in\mathbb R^{d_k}$，初值 $W_0=0$、$n=n_0\mathbf 1$。每一步：

$$r_t=\Sigma^{-1/2}\big(z_{t+1}-f_\theta(z_t,a_t)\big),\qquad k_t=\frac{\phi(z_t,a_t)}{\|\phi(z_t,a_t)\|},\qquad m_t=W_{t-1}k_t,\qquad e_t=r_t-m_t \tag{1}$$

$$c_t=n_{t-1}^\top(k_t\odot k_t),\qquad \tau_t=\frac{1}{1+c_t} \tag{2}$$

$$\ell_t=\lambda\Big[\frac{\|e_t\|^2}{2(1+1/c_t)}-\frac{\|r_t\|^2}{2(1+1/n_0)}+\frac D2\log\frac{1+1/c_t}{1+1/n_0}\Big] \tag{3}$$

$$\gamma_t=\mathrm{sigmoid}\Big(\ell_t+\log\tfrac{p_c}{1-p_c}\Big),\qquad \beta_t=\gamma_t+(1-\gamma_t)\tau_t,\qquad v_t=\frac{(1-\gamma_t)\tau_t+\gamma_t\rho}{\beta_t}\,r_t,\quad \rho=\frac1{1+n_0} \tag{4}$$

$$W_t=W_{t-1}+\beta_t\,(v_t-W_{t-1}k_t)\,k_t^\top,\qquad n_t=\max\!\big(n_0,\;n_{t-1}+\big(1-\gamma_t(c_t-n_0)\big)(k_t\odot k_t)\big) \tag{5}$$

外层损失是写入前的序贯（prequential）残差 $\mathcal L=\sum_t\|r_t-W_{t-1}k_t\|^2$。梯度只流向 key 投影 $\phi$。$\gamma_t,\beta_t,c_t$ 和 $v_t$ 的缩放系数一律 stop-gradient，$f_\theta$ 保持冻结。门控的全部标量（$\sigma_d,n_0,\lambda,p_c$）都按下文规则闭式校准，不经梯度学习。这与 TTT3R"不加参数、不训练，用模型已经算出的统计量给出闭式 β"是同一种做法（[TTT3R](https://arxiv.org/pdf/2509.26645v4)），只是统计量换成了世界模型的预测似然。

**式 (3) 是怎么来的。** 在白化单位下（噪声方差 1），记忆沿 $k$ 的信念是 $\mathcal N(m_t,\,1/c_t)$，先验是 $\mathcal N(0,\,1/n_0)$，也就是"慢模型平均而言是对的，可解释残差方差 $\sigma_0^2$ 对应 $n_0=\sigma^2/\sigma_0^2$"。于是两个预测分布分别为 $\mathcal N(m,(1+1/c)I)$ 与 $\mathcal N(0,(1+1/n_0)I)$，$S_{\rm BF}$ 取二者之比，取对数即得式 (3)。等方差、点预测时它退化为笔记里的 $(\ell_{\rm fast}-\ell_{\rm slow})/2\sigma^2$。若改用 Cambrian-S 的余弦尺度，在共享集中度 $\kappa$ 的 vMF 似然下，$\log S_{\rm BF}=\kappa\,(s_{\rm fast}-s_{\rm slow})$，其中 $s=1-\cos$ 正是 Cambrian-S 的 surprise（本报告推导）。

**五处修正如何落在式子里：**

| 问题 | 落点 | 检查方式 |
|---|---|---|
| 语义：γ 是混合权重，不是学习率 | 式 (4)(5) 等价于沿 $k$ 的新均值 $=(1-\gamma)(m+\tau e)+\gamma\rho r$：整合分支是 Kalman 更新，重置分支是"从先验出发只看一次观测的后验"。宽先验下 $\rho\to1$，有效增益就是 Nassar 的 $\gamma+(1-\gamma)\tau$ | 单测：$\gamma\equiv0$ 时退化为对角 RLS；$\gamma\equiv1$ 时 $Wk=\rho r$ |
| 冷启动 | 未写过的 key 有 $c=n_0$、$m=0$，所以 $\ell=0$、$\gamma=p_c$，但写入后 $Wk=\rho r$，即先验后验均值，而不是只写 $p_c$ 那么多 | 消融臂 A7（β = γ）应在首访内容上写入约 $p_c$；H3 检验 |
| 阈值不是零，是 hazard | $p_c$ 是每步变化概率，由物理时间尺度事先给定（Memory Maze 取 $10^{-3}$/步，即每个 1000 步 episode 至多一次变化）。"变化"判决为 $\gamma>1/2\Leftrightarrow\ell>\log\frac{1-p_c}{p_c}\approx6.9$ nats；"熟悉"判决为 $\ell<-\log3$，即记忆的预测密度至少是先验的 3 倍 | $p_c\in\{10^{-4},10^{-2}\}$ 只作稳健性报告，不用来调参 |
| 噪声与 σ 校准 | 只用训练集，一次性：$\sigma_d^2=\tfrac12\mathrm{Var}(r_t-r_\tau)$，取自真值重访对（同一格、朝向余弦 >0.95、同动作、间隔 ≥50 步），下限 $10^{-3}V_d$，其中 $V_d=\mathrm{Var}(r_d)$；$\sigma_{0,d}^2=\max(V_d-\sigma_d^2,10^{-3}V_d)$；$n_0=\mathrm{median}_d\,\sigma_d^2/\sigma_{0,d}^2$；$\lambda=D_{\rm eff}/D$，$D_{\rm eff}=(\mathrm{tr}C)^2/\mathrm{tr}(C^2)$ 是白化残差协方差的参与率。宽先验让"两个预测器都错"时偏向重置；$\rho<1$ 把随机键上的写入向慢模型收缩，抑制噪声电视式的反复改写 | 报告 γ 直方图，以及在训练集拼接上的可靠性曲线；γ 若只取 $\{p_c,1\}$ 两值，判为饱和 |
| 并行训练 | 门控依赖状态，破坏精确的 chunkwise 形式。做法与 TTT mini-batch 一致，都在块起点状态上求值（[TTT](https://arxiv.org/html/2407.04620)；[Lattice](https://huggingface.co/papers/2504.05646.md)）：每块先用块起点的 $(W_c,n_c)$ 一次矩阵乘算出陈旧的 $m_t,c_t,\gamma_t,\beta_t,v_t$，全部 stop-gradient；再把 $(k,v,\beta)$ 送入标准 delta-rule 块核，带 initial_state 输入，Python 循环逐块调用。式 (5) 可写成 $W\leftarrow W+\beta(v-Wk)k^\top$，正是 DeltaNet 的标准形式 | 推理始终逐步精确执行。软件门槛：同一序列上 chunk = 64 与逐步扫描的 $\lvert\Delta\beta\rvert>0.1$ 比例 ≤5%。块内重复对门控不可见，这是已知代价 |

**实现上，BF-Delta 等于"对角 Kalman delta"加一行沿 $k$ 的 BF 重置。** Kalman 类快权重的增益来自协方差，与新息大小无关（[KDN](https://arxiv.org/abs/2609.07816)），对应下表的 A6；BF-Delta 只在其上多出 γ 这一项。核心代码如下，fp32 状态，L2 归一化 key：

```python
# r: [B,T,D] 白化后的慢模型残差; k: [B,T,dk] 单位 key; 状态 W:[B,D,dk], n:[B,dk]
def bf_delta_chunk(W, n, k, r, n0, rho, lam, logit_pc, D):
    with torch.no_grad():                                   # 门控在块起点状态上求值
        m  = torch.einsum('bdk,btk->btd', W, k)
        c  = torch.einsum('bk,btk->bt', n, k*k)
        e2 = ((r - m)**2).sum(-1); r2 = (r**2).sum(-1)
        ell = lam*(e2/(2*(1+1/c)) - r2/(2*(1+1/n0)) + 0.5*D*torch.log((1+1/c)/(1+1/n0)))
        g   = torch.sigmoid(ell + logit_pc)
        tau = 1/(1+c); beta = g + (1-g)*tau
        vs  = ((1-g)*tau + g*rho)/beta                      # v = vs * r
    W_new = delta_rule_chunk(W, k, vs[...,None]*r, beta)    # 标准 delta 块核, 可反传到 k
    n_new = torch.clamp(n + ((1 - g*(c - n0))[...,None]*(k*k)).sum(1), min=n0)
    return W_new, n_new, ell, g, beta
```

**每层用什么潜变量。** 三层共用式 (1)–(5)，按 AGENTS.md 第 5 条的顺序 L0 → L1 → R 推进。

| 层 | 输入 $z_t$（全部来自公开数据） | 慢模型 $f_\theta$ | key $\phi$ | 状态预算 |
|---|---|---|---|---|
| L0 | Memory Maze 自带真值组成的 token：位置（归一化，2 维）、朝向（2 维）；朝向坐标系下前方 5×5 局部格（走廊 / 墙 / 视线外未知，视线用 DDA 在 `maze_layout` 上求）；3 个物体各 [可见, 朝向坐标系下的相对位置]。9x9 时 $D=38$ | 2 层 MLP | 2 层 MLP → $d_k=32$ | $38\times32+32=1248$ 个标量 |
| L1 | L0 加观测噪声、随机 token 丢弃，以及每 episode 一个"闪烁"物体（噪声电视键）；另建一个局部重置拆分：在 token 流中途改一个物体的位置 | 同上 | 同上 | 同上 |
| R | 裁掉 2 px 边框，冻结 DINOv2 ViT-S/14 在 224 px 下的 CLS + 2×2 池化 patch token，在训练集上做 PCA 到 $D=64$ | 2 层 MLP | 2 层 MLP → 32 | $64\times32+32=2080$ |
| R2（VSI） | Cambrian-S 缓存的 SigLIP2 8×8×1152 token 均值池化，经 PCA 白化（在不相交校准集上拟合）后取固定随机正交投影到 $D=256$ | 冻结 LFP head 的预测，同样池化与投影（不新训） | 固定随机投影 → 128，不训练 | $256\times128$ ≈ 32.8K 个标量 |

L0 用的是公开数据集自带的标签，不是自建世界，符合 AGENTS.md 第 2 条"上游有疑问就换真值 token"的要求。L1 的扰动与 VSC-Repeat 一样，是对公开数据的确定性派生拆分，必须由脚本固定生成并随结果归档。R 层的编码器设置一次定死，不调；R 若失败，按第 1、2 条退回真值 token 报告，不开感知诊断轮。DINOv2 是 Apache-2.0；DINOv3 需要接受 gated 许可，因此不用（[DINOv2](https://github.com/facebookresearch/dinov2)；[DINOv3 卡片](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)）。DINO-WM 的先例是 `dinov2_vits14` 在 224 px 下的 patch token（[dino_wm 配置](https://github.com/gaoyuezhou/dino_wm/blob/0a9492f/conf/encoder/dino.yaml)）。

### 数据：Memory Maze 离线集为主，VSC-Repeat 为次

**主数据集：Memory Maze 离线数据（MIT）。** 9x9 和 15x15 各有 3 万条轨迹，每条 1000 步、1001 帧，按 29k/1k 划分训练/评测，每个约 100 GB，全部放在一个 Google Drive 文件夹（ID `1RcnkTZVwEHnAQeEuw7X8Y1RPSmrFLDFB`）（[README](https://github.com/jurgisp/memory-maze/blob/4030901cef3b7d4e7f92e62099e8be378303dc0a/README.md#offline-dataset)）。数据由脚本化策略生成，策略在动作噪声下导航到随机点，"form spatial loops"，天然带重访。以下各条坑都应写成 S0 单测：

| 坑 | 事实 | 处理 |
|---|---|---|
| `maze_layout` 语义与 README 措辞相反 | 包装器存的是 `nonwalls`，**1 = 走廊，0 = 墙**；去掉外墙，竖轴翻转，左下角为 [0,0]（[wrappers.py L159–178](https://github.com/jurgisp/memory-maze/blob/4030901cef3b7d4e7f92e62099e8be378303dc0a/memory_maze/wrappers.py#L159-L178)） | 单测：每个 $t$ 都有 `maze_layout[int(y), int(x)] == 1`，其中 $(x,y)$ = `agent_pos[t]`（[oracle.py L47–58](https://github.com/jurgisp/memory-maze/blob/4030901cef3b7d4e7f92e62099e8be378303dc0a/memory_maze/oracle.py#L47-L58)） |
| 2 px 目标色边框 | 默认开启，边框颜色为 `target_color*255*0.7`，每当达成目标就变色，每条轨迹 1–2 次（[wrappers.py L213–229](https://github.com/jurgisp/memory-maze/blob/4030901cef3b7d4e7f92e62099e8be378303dc0a/memory_maze/wrappers.py#L213-L229)） | 编码前每边裁 2 px（64→60）；否则 surprise 会在变色处出现与记忆无关的尖峰 |
| 动作对齐 | `action` 是"上一个动作"的 one-hot；`image[0]` 在第一个动作之前（[README](https://github.com/jurgisp/memory-maze/blob/4030901cef3b7d4e7f92e62099e8be378303dc0a/README.md)） | 转移 $z_t\to z_{t+1}$ 由 `action[t+1]` 驱动；单测验证 |
| 坐标 | `agent_pos` 以格为单位，左下 (0.1,0.1)；`targets_vec` 是以 agent 为中心的相对位置，但是否已旋转到朝向坐标系未确认 | S0：把 `targets_pos − agent_pos` 按 `agent_dir` 旋转后与 `targets_vec` 比较 |
| 没有可见性键 | NPZ 不含"自上次看见以来的步数"；相机 FoV 在代码里未设定 | L0 用前方 5×5 网格窗口加 DDA 视线，完全绕开 FoV；在一个文件上执行 `np.load(...).files`，看有无未记录的键 |
| 15x15 被截断 | 在线环境 4000 步，离线数据截到 1000 步，有 6 个物体（[tasks.py](https://github.com/jurgisp/memory-maze/blob/4030901cef3b7d4e7f92e62099e8be378303dc0a/memory_maze/tasks.py#L9-L75)） | 15x15 只作容量/干扰的次要对照 |
| 下载 | Drive 文件夹无法列出文件；gdown 的文件夹模式有数量限制；没有 HF 镜像 | 用 rclone 挂 Google Drive；先下载 9x9；NPZ 解压受 CPU 限制，特征只提取一次 |

**存储按算术估计。** CLS + 2×2 池化共 5 个 token × 384 维 × fp16，约 3.84 KB/帧，3000 万帧约 115 GB；PCA-64 之后约 3.8 GB。完整 16×16 patch 网格每个数据集约 5.9 TB，不可行。L0 token 只读 NPZ 中的位姿、地图与物体数组。`savez_compressed` 的各数组是独立的 zip 成员，读 L0 字段无需解压图像。

**拆分。** 训练集里固定留出 1k 条作 dev；官方 1k 评测集即 test，按第 7 条只在 FINAL 打开一次。

**次数据集：VSI-SUPER-Count，配冻结的 Cambrian-S-7B-LFP（Apache-2.0）。** 只需 `10mins.zip`，13,510,215,623 B；视频为 24 FPS；10 分钟常规拆分有 50 个视频，真值计数在 10–75 之间（[HF 文件树](https://huggingface.co/api/datasets/nyu-visionx/VSI-SUPER-Count/tree/main?recursive=true)；[supersanity 日志](https://github.com/bethgelab/supersanity/tree/296c8c3908f5e1b8e68e96beec6c8e02ee1928c2/results/vsc_repeat_exps)）。这里有三个坑。第一，supersanity 的 YAML 与脚本用的是预发布 ID `ShushengYang/VSI-SUPER-Count` 和 `ShushengYang/Cambrian-S-7B-LFP`，要换成 `nyu-visionx/...`（[vsc_repeat2.sh](https://github.com/bethgelab/supersanity/blob/296c8c3908f5e1b8e68e96beec6c8e02ee1928c2/vsc_repeat/lmms-eval/scripts/vsc_repeat2.sh)）。第二，7B-LFP 权重在 HF 上没有许可标签，其余 Cambrian-S 模型是 Apache-2.0（[HF models API](https://huggingface.co/api/models?author=nyu-visionx&limit=200&sort=createdAt&direction=-1)），只能作研究评测，不再分发派生物。第三，源视频来自 ScanNet、ScanNet++、ARKitScenes，它们自身的条款没有核对。VSI-SUPER 没有 dev 拆分，所以 dev 用 VSI-590K 中与 VSI-SUPER 不相交的房间巡游视频（Apache-2.0）自建 Repeat，只做离线熟悉度 AUROC，不调 LLM；VSC-Repeat 本身只在 FINAL 打开（[VSI-590K](https://huggingface.co/datasets/nyu-visionx/VSI-590K)）。不相交性按视频 ID 核对。

### 对照臂：只换写入规则

AGENTS.md 第 3 条要求每轮至少有一组等预算的记忆对照。主对照里，所有臂共用同一个 $\phi$ 结构、同一个冻结的 $f_\theta$ 残差目标、同一个外层损失，状态标量数相同（用不到计数向量的臂也照样携带它），只换写入规则。

| 臂 | 写入规则 | 角色 |
|---|---|---|
| A0 | 无记忆，只用 $f_\theta$ | 下界 |
| A1 | 常数 β，扫 $\{0.05,0.1,0.2,0.4,0.8,1\}$ | 普通 TTT-linear/DeltaNet；画稳定性–可塑性前沿 |
| A2 / A2s | GDN：β 与衰减由输入经学习门控给出（[fla](https://github.com/fla-org/flash-linear-attention)）；A2s 在训练集拼接上训练 | 学到的门控；A2s 是"见过变化数据的学习门控"这一公平强对手 |
| A3 | $\beta=\mathrm{sigmoid}(a\,s^{\rm slow}_t+b)$，其中 $s^{\rm slow}=1-\cos(f_\theta,z_{t+1})$ | 把 Cambrian-S 的无记忆 surprise 搬进权重 |
| A4 | $\beta=\mathrm{sigmoid}(a\,\lVert e_t\rVert^2/D+b)$，$e$ 取陈旧值 | 记忆参照误差，无先验比值（Titans、T²Mem 式）；检验"比值是否必要" |
| A5 | 二值：$\lVert e_t\rVert^2>\mu+\sigma$（最近 50 步）时 β = β_hi，否则 β = 0 | FFN 式门控 |
| A6 | $\beta=\tau_t$，即 $\gamma\equiv0$ | 对角 Kalman/RLS，对新息大小不敏感 |
| A7 | $\beta=\gamma$，等方差点预测形式 | 原始 BF 候选；冷启动消融 |
| **A8** | **BF-Delta，式 (1)–(5)** | **本方法** |
| A9 | k-NN 库（记 $z$ 与残差对），按新颖度准入，**等字节** | Surprise Forcing 式的非参数熟悉度捷径（[2607.18436](https://arxiv.org/abs/2607.18436)） |
| A10 / A11 | GRU，隐状态等于状态预算；FIFO 窗口，32 个潜变量 | 等状态、不同族的参照 |

A9 在 R 层等字节时只能存约 16 条，太弱，因此另报一个 10 倍字节的 A9×10，并明确标为"非等预算"。主臂 A8 与最强对手各跑 5 个种子，其余臂跑 3 个。按 2609.16183 的做法，报告各写入规则的"锁定率"。

### 指标：官方探针、重访、拼接、VSC-Repeat

**Memory Maze 官方探针协议（逐项复刻）。** 冻结序列模型，对其内部状态做 stop-gradient 探针：快权重臂探 $[\mathrm{vec}(W),n]$，GRU 探隐状态。探针是 4 层 × 1024 单元的 MLP，带 LayerNorm 和 ELU，Adam 学习率 3e-4、eps 1e-5。墙壁探针输入状态并拼接位置与朝向，对每格做 BCE，报告全格平均准确率；物体探针对 `targets_vec` 回归，报告各物体平均的 MSE。分数取 1k 条评测轨迹第 500–1000 步的平均（[Memory Maze 论文](https://arxiv.org/pdf/2210.13383)）。公开 pydreamer 里有探针头，但没有离线 Memory Maze 的探针配置，因此要按论文重写。原始代码还按"可见年龄"分箱记录物体 MSE，分箱为 0 / 1–5 / 6–10 / 11–50 / 51–200 / 201–1000（[probes.py L89–137](https://github.com/jurgisp/pydreamer/blob/8a4ea85a925b57ddf24a221b41dfb3257f89d63a/pydreamer/models/probes.py#L89-L137)），V6 用它作保持曲线。论文表 4 的基线如下，均值来自 3 个种子：

| 方法 | 墙 9x9 %↑ | 墙 15x15 %↑ | 物体 9x9 MSE↓ | 物体 15x15 MSE↓ |
|---|---|---|---|---|
| 常数基线 | 80.8 | 78.3 | 23.9 | 64.8 |
| VAE（无记忆） | 84.1 | 78.7 | 17.2 | 56.2 |
| RSSM | 89.0 | 80.0 | 8.3 | 39.5 |
| VAE+GRU（TBTT） | 90.0 | 80.2 | 7.8 | 34.9 |
| RSSM（TBTT） | 94.0 | 80.5 | 6.2 | 33.8 |
| 监督 oracle | 99.7 | 88.5 | 0.3 | 9.2 |

这些基线用的是像素 VAE 和 2048 维状态，与冻结 DINOv2 特征不能直接比较。V6 在同一协议下重算自己的常数基线与 A0，表 4 只作量级参照。

**重访与回环指标（在 dev/test 上由真值计算）。** 重访事件定义为同时满足三条：当前格 $c_t$ 以前到过；与上次占用的间隔 ≥ G = 50 步（4 Hz 下约 12.5 秒）；其间 BFS 距离曾超过 2 格。若再加朝向余弦 >0.7，则为视角重访。每个重访事件报告两项：一是重访后 10 步内的潜预测误差，与"在离开时清空记忆"的同间隔对照相减，得到 MemoryGain（沿用 [R2M-Bench](https://arxiv.org/abs/2608.27328) 的对照对逻辑）；二是对上次看见已过 ≥ G 步的格子和物体的探针误差。

**REPEAT/CHANGE 拼接协议。** 这是 VSC-Repeat 在 Memory Maze 上的对应物。每条离线轨迹本身就是一个随机迷宫，episode 内布局和物体不变，所以拼接可以免费得到标签。取 $T_1=400$、$t_0=600$、$T_2=400$，总长 800 步：

| 条件 | 构造 | 理想记忆的行为 |
|---|---|---|
| EXACT | A[0:400] ⧺ A[0:400] | 熟悉：ℓ ≪ 0，几乎不写，A 探针保持 |
| NEWPATH | A[0:400] ⧺ A[600:1000] | 同一世界换起点：已探部分熟悉，新探部分新颖（ℓ ≈ 0） |
| CHANGE | A[0:400] ⧺ B[600:1000] | 冲突的 key 上 ℓ ≫ 0：沿这些 key 重置，B 探针快速收敛 |
| 连续对照 | A[0:800] | 探针参照 |

三种拼接在第 400 步都有一次瞬移，拼接那一步对所有臂都不写入、不计分，因为瞬移是一个"怪球"事件。计算四个量。第一，变化检测 AUROC：拼接后 20 步内 ℓ 的均值，CHANGE 为正类，EXACT 与 NEWPATH 为负类。第二，CHANGE 恢复差额：拼接后 200 步内，B 墙探针准确率相对"拼接处清空记忆"对照的平均差。第三，REPEAT 保持差额：同一窗口内 A 探针相对连续对照的退化。第四，以第二、三项为两轴，画 A1 扫 β 得到的前沿，看 A8 落在何处。另有一组 A⧺C 对 B⧺C 的配对，用来测历史分离随 γ 驱动的重置衰减的半衰期。训练集上也用同一方式构造拼接，但只用于校准检查和训练 A2s；评测拼接只来自 dev/test。

**VSC-Repeat 的构造与基线数。** `perturb_vsc_repeat.py` 对每个 10 分钟视频执行 `ffmpeg -stream_loop N -c copy`，N = 1、2、5 时总长为 20、30、60 分钟，沿用原始真值（[perturb 脚本](https://github.com/bethgelab/supersanity/blob/296c8c3908f5e1b8e68e96beec6c8e02ee1928c2/vsc_repeat/lmms-eval/scripts/perturb_vsc_repeat.py)）。运行参数为 `--sensory_window_size 128 --surprise_threshold 0.39 --enable_visual_feature_caching True`。公开日志如下，单 GPU，型号未记录：

| 条件 | MRA | 中位 pred/GT | 50 个视频总耗时 |
|---|---|---|---|
| N = 0（10 min） | 42.0 | 0.99 | 2,481 s |
| N = 1（20 min） | 20.6 | 1.85 | 4,771 s |
| N = 2（30 min） | 3.6 | 2.85 | 6,388 s |
| N = 5（60 min） | 0.0 | 5.80 | 15,969 s |

来源：[supersanity 结果](https://github.com/bethgelab/supersanity/tree/296c8c3908f5e1b8e68e96beec6c8e02ee1928c2/results/vsc_repeat_exps)；[2511.16655](https://arxiv.org/html/2511.16655)。pred/GT 几乎正好等于 N+1，说明逐段求和把每一份拷贝都重数了一遍，失败完全出在写入与聚合，不在感知。MRA 的定义是在阈值 0.50:0.05:0.95 的 10 个点上取 $\lvert\mathrm{pred}-\mathrm{gt}\rvert/\mathrm{gt}\le1-\theta$ 的均值（[vsc utils](https://github.com/cambrian-mllm/cambrian-s/blob/058b1b4c96846a229d1c9bb69da6ecf5ceee4a6b/lmms-eval/lmms_eval/tasks/cambrians_vsc/utils.py)）。

精确重复可以被 k-NN 逐帧匹配轻易去重，所以另建两种 N = 1 的"近重复"派生拆分。一种是**相位平移副本**：重编码时丢掉前 12 帧，即 0.5 秒（`-ss 0.5` 配 `-c copy` 会在关键帧处截断，所以必须重编码），使 1 fps 采样器抽到不同的帧。另一种是**倒序副本**：先按采样器步长抽出 1 fps 帧，倒序后每帧重复 24 次重新编码，保证采样器抽到的正是倒序帧。倒序副本专门检验"转移记忆"与"内容熟悉度"的差别：从 B 回到 A 的转移是新的，A 的内容却是熟悉的。VSR 不纳入 V6，因为 NoSense 这种无记忆检索在 10/30/60 分钟上已达 98.3/98.3/96.7%（[2511.16655](https://arxiv.org/html/2511.16655)）。

**Cambrian-S 代码注入点**（[cambrians_vsc.py @058b1b4](https://github.com/cambrian-mllm/cambrian-s/blob/058b1b4c96846a229d1c9bb69da6ecf5ceee4a6b/lmms-eval/lmms_eval/models/simple/cambrians_vsc.py)）。补丁约 20 行，官方 LFP surprise 和调好的 0.39 阈值一律不动，这样新增效应只来自熟悉度否决：

| 位置 | 现状 | V6 补丁 |
|---|---|---|
| L58–80 构造函数 | `sensory_window_size=128`、`surprise_threshold` | 加 `bf_dir`、`bf_window=5`、`bf_delta=log 3`、`dump_dir` |
| L291 主循环 | 每个视频有唯一 `visual_tensor_paths` | 设了 `bf_dir` 时按路径加载 `*_bf.npy`（逐帧 ℓ_t），断言长度等于帧数；该路径能区分 repeat-N 的拷贝 |
| L428–432 | `surprisingness_score = 1 − cos(LFP 预测, 实际 SigLIP)` | 不改。设了 `dump_dir` 时保存池化后的实际特征 |
| L450 | `nfp_head` 产生下一帧预测 | dump 池化后的预测，作为慢先验 |
| L474 | 边界：surprise > 阈值，或最后一帧 | 边界触发时，取 ℓ 在 [t, t+5] 上的均值，低于 −log 3 就把新段标为"熟悉"；决策延迟 5 帧，仍然因果 |
| L476–515 | 段内帧组成 `episodic_kv_cache` | 熟悉段照常构建，保持计算一致 |
| L566–572 | `outputs = str(sum(...))` | 熟悉段的答案不计入求和 |

流程分三遍。第一遍用官方配置运行，同时 dump，这一遍就是对公开基线的复现。第二遍离线、因果地运行记忆，算出 ℓ_t，开销可以忽略。第三遍带上 `bf_dir` 重跑。这依赖一个前提：VSC 的 LFP 预测不依赖于边界决策。VSC 没有压缩，看起来成立，但 S0 必须验证：用两个阈值跑同一视频，比较两次 dump 的预测是否一致。若不成立，就把记忆直接放进 lmms-eval 循环在线运行，它只有约 3 万个标量。LFP head 吃的是 LLM 隐状态，无法脱离 7B 模型单独运行，这也是选择 dump 路线的原因。

### 预注册假设与证伪条件

以下阈值是本报告的提议。按 AGENTS.md 第 6 条，方向重定属于协议变更，需要用户批准并另起 `docs/PREREGISTRATION_V6.1.md` 与对应的 `prereg.py`；现行 V6 预注册仍是 PROPOSED，从未冻结。所有区间都是按 episode（VSI 为按视频）配对的 bootstrap，1000 次，取 95% 分位；判定每个种子分别成立才算通过。

| 编号 | 假设（层） | 通过条件 | 证伪后放弃或收窄 |
|---|---|---|---|
| H1 稳定性–可塑性（L0 主，R 复现方向） | A8 落在 A1 前沿之外 | 在与 A8 的 CHANGE 恢复差额相等的 β 处（在扫描上插值），A8 的 REPEAT 保持差额 ≤ 0.5 × A1，CI 不含 0；两个轴上同时优于 A2 | 某个 β 的 A1 支配 A8：门控无用，报告负结果；A4 与 A8 的差在 CI 内：收窄为"记忆参照误差门控"，撤回"比值必要" |
| H2 变化检测（L0、R） | 对数比值能分开 CHANGE 与 EXACT/NEWPATH | AUROC：L0 ≥ 0.90，R ≥ 0.80；A3 的慢 surprise ≤ 0.60（两类条件都有瞬移）；A8 比 A4 的 ‖e‖² 至少高 0.05 | 差距 < 0.05：比值不比记忆误差多提供信息 |
| H3 冷启动（L0、R） | Nassar 形式修复首访写入 | 未拼接 dev 轨迹上，A8 的官方物体 MSE ≤ 0.8 × A7；A7 首访步的平均 β ≈ p_c（实现检查） | 两者无差：冷启动修正在实践中无关紧要，如实报告 |
| H4 不伤存储（L0、R） | 门控在平稳情形不吃亏 | A8 的官方探针相对 {A1 最优 β, A2, A6} 中的最好者，墙准确率差 ≤ 1.0 个百分点，物体 MSE 差 ≤ 10%（非劣效）；所有快权重臂都优于 A0 | 劣于界限：按 SR-TTT 式"surprise 门控偏向丢弃不意外内容"的发现报告（[2603.06642](https://arxiv.org/abs/2603.06642)） |
| H5 一个 hazard 跨长度（L0、R） | 固定 p_c 迁移 | 用 2 段、3 段拼接（2000、3000 步）检测变化（γ > 0.5），F1 与 1000 步时相差 ≤ 0.05；p_c 改为 10⁻⁴ 或 10⁻² 时 F1 变化 ≤ 0.10 | 迁移失败：撤回"无需按长度调参"的主张 |
| H6 噪声电视（L1） | 宽先验与 ρ 收缩抑制反复改写 | 闪烁物体上的伪变化率 ≤ 2 × 干净物体；其他物体的探针 MSE 退化 ≤ 10%；A7 预期 ≥ 3 × | A8 同样反复改写：引入"变化 vs 怪球"的混合模型属于新协议，留待下一轮 |
| H7 VSC-Repeat（R2，FINAL 一次） | 熟悉度否决修复重复计数 | MRA(N = 1, 2, 5) ≥ 0.8 × A8 自身的 MRA(N = 0)；MRA(N = 0) ≥ 官方 − 5；N = 5 时中位 pred/GT ≤ 1.3；离线熟悉度 AUROC（拷贝 ≥2 的帧对拷贝 1 的帧）：精确 ≥ 0.90，相位平移 ≥ 0.80；倒序只作探索性报告 | 等字节的 A9 在精确、相位平移、倒序三项上都 ≥ A8：撤回参数化记忆主张，报告"非参数熟悉度足够" |
| H8 软件门槛 | 块内陈旧可接受 | chunk = 64 与逐步扫描下 $\lvert\Delta\beta\rvert>0.1$ 的比例 ≤ 5%；γ ≡ 0 和 γ ≡ 1 两个单测通过 | 失败：训练 chunk 降到 16 或改为逐步扫描（模型小，承受得起） |

**规则对齐。** 只有记忆指标上的坏例能驱动下一轮，每个坏例最多两轮诊断（第 4 条）。门槛失败先按 bug 排查，再作为发现报告，绝不降阈值或换种子（第 6 条）。

### 算力：约 140 GPUh，8×A100 两天

| 阶段 | 内容 | GPUh 上限 | 进入下一阶段的门槛 |
|---|---|---|---|
| S0 | CPU：下载、NPZ 单测、L0 token、重访标签、拼接索引；Cambrian-S 补丁单测；LFP dump 与边界无关的检查 | 0 | 全部单测通过 |
| S1 | 冒烟：所有臂在 16 条轨迹上过拟合；chunk 门控一致性 | 2 | H8 |
| S2 | L0 主矩阵：约 11 个臂 × 3–5 个种子，加探针 | 12 | H1–H5 能在 dev 上判定 |
| S3 | L1：噪声、闪烁物体、局部重置 | 8 | H6 |
| S4 | R：DINOv2 特征（约 5–8，主要花在 IO 上）、慢模型（≤1）、主矩阵（约 15）、探针（约 12）、拼接评测；15x15 只跑 A8 与最强对手 | 45 | H1、H2、H4 方向复现 |
| S5 | VSI dev：VSI-590K 校准集 dump、离线熟悉度 AUROC、补丁冒烟 | 6 | 补丁在 2 个校准视频上跑通 |
| FINAL | 冻结后：Memory Maze test 1k；VSC N = 0/1/2/5 的官方复现（同时 dump），约 8.2，加上 A8 与 A9 重跑各约 8.2，再加两种近重复 × 3 个臂约 8 | 40 | — |
| 预留 | 重跑与调试 | 25 | — |
| 合计 | | **≤138**（硬上限 160；第一个决策点 S0–S3 ≤ 22） | |

**估算依据。** R 层每次运行只有约 2K 个状态标量，$\phi$ 约 0.1M 参数，残差可以预先算好，所以每次运行约 0.3–1 GPUh，每卡可同时放 4 个进程。DINOv2-S/14 在 224 px 下约 12 GFLOPs/图，9x9 的 3000 万帧约 1–2 个 A100 小时，实际受 NPZ 解压限制，按 5–8 GPUh 计。这些都是本报告的算术估计。VSI 的数字由 supersanity 日志外推：约每视频分钟 5 GPU 秒，GPU 型号未知，按 A100 近似。S1 的实测 profile 若与上表偏差超过 2 倍，先更新预算表并通知用户。4×H800 的墙钟大致相同。还要单独预留下载时间：Drive 上约 100–200 GB，HF 上 13.5 GB，LFP 权重约 16 GB。

### 风险与退路

| 风险 | 早期信号 | 退路 |
|---|---|---|
| ICLR 2027 投稿抢先，或审稿人归并为 MOCA | 冻结前在 OpenReview 检索 "Bayes factor surprise memory"、"VSC-Repeat"、"test-time training world model revisit" | 主张写成"实例化 + 评测"；相关工作完整引用 |
| 比值不比记忆误差强（A4 ≈ A8） | S2 时 H2 的差距 < 0.05 | 收窄为"记忆参照误差门控 + 拼接协议"，仍是一项完整的记忆结论 |
| 见过变化的学习门控追平（A2s ≈ A8） | S2 | 结论改为"闭式门控不需要变化数据"，这是有用的限定结果 |
| 高维饱和，γ 只取 {p_c, 1} | γ 直方图 | 已用 λ = D_eff/D；仍饱和就改用 vMF 余弦形式 κ(s_f − s_s)，κ 用同样的参与率规则给定 |
| 怪球事件：瞬移、边框变色、镜头切换 | ℓ 在这些步出现尖峰 | 拼接步不写入；裁边框；把"变化 vs 怪球"留作下一版协议 |
| R 层失败而 L0 成功 | S4 | 不诊断编码器（第 1、2 条），作为发现报告，结论限定在真值 token |
| k-NN 在精确重复上追平 | H7 | 这在预期之内；主张依靠近重复、倒序和 Memory Maze 探针 |
| LFP 权重无许可标签，或 dump 路线不成立 | S0 | 改用 copy-last 先验，即 Cambrian-S 论文里的相邻帧基线；或把记忆嵌入循环在线运行 |
| VSI 统计功效有限（每个条件 50 个视频） | CI 宽 | 只主张大效应：官方在 N = 2/5 已跌到 3.6/0.0 |
| 种子锁定彩票 | 种子间结果呈双峰 | 主臂用 5 个种子并报告锁定率（[2609.16183](https://arxiv.org/html/2609.16183v1)） |
| Titans 式在线记忆在 Memory Maze 上重建崩溃 | 探针随步数恶化 | 这正是 H4 要测的现象；先例见 [On Memory](https://arxiv.org/html/2512.06983v1) |

### 代码落点

工作分支 `v6-predictive-memory` 上已有一套**未提交、状态为 PROPOSED** 的旧 V6 代码。按文件名判断：`memories.py`（delta、key 侧 GDN、FIFO、GRU、slot、LaCT 等）、`metrics.py`（bootstrap、AUROC、MRA、F1、memory_gain）、`surprise.py`（含 k-NN novelty）、`prereg.py`、`report.py` 的接口可以复用，但要从对象 token 改为向量潜变量输入。`worlds_l0.py`（自建生成器）、`worlds_l1.py`（Miniworld 重生成）和 `worlds_real.py`（nuScenes/MOT）在新约束下退役或搁置，`lfp.py` 由慢模型取代。新增模块：`mmaze_data.py`（NPZ 加载、L0 token、DDA 视线、重访标签、拼接）、`features.py`（裁边框后提取 DINOv2）、`slow_model.py`、`bf_delta.py`（式 (1)–(5) 与块循环）、`probes.py`（复刻官方探针）、`vsc_patch/`（Cambrian-S 补丁、dump 与近重复构造脚本）。新的 `BUDGET_GPUH` 按上表经用户批准后写入。

## RSI 为什么不进 V6，以及唯一允许的受限形式

**结论是信息论上的：两个数据集在每个时长上都有真值，自标注和时长自举都带不来新信息。** Lee 等人的递归自改进之所以成立，前提是更难一级的标签本来拿不到，而且不过滤就会雪崩（[Lee et al.](https://arxiv.org/html/2502.01612v2)）。Memory Maze 每一步都有地图和物体真值，VSC 有计数真值，前提不成立。直接的竞争者又很便宜。Memorizon 把监督跨度从 100 秒扩到 400 秒，步时只增加约 12%（[Memorizon](https://arxiv.org/html/2610.00544v1)）；按准确率门控的课程就能把固定状态记忆从机会水平拉到满分（[2609.16183](https://arxiv.org/html/2609.16183v1)）。2026 年的 RSI 综述总结说，已证实的收益跟踪验证器的强度，"self-evolution is not free"（[2607.07663](https://arxiv.org/abs/2607.07663)）。

**self-forcing 只是曝光偏差修正，不是 RSI，而且在 Saining 的数据上还伤记忆。** Solaris 在 KV 上反传之后，FID 变好，Memory 却从 48.96 降到 37.50，Grounding 从 72.92 降到 62.50（[Solaris 项目页](https://solaris-wm.github.io/)）。对循环信念做完整 BPTT 时，模型本来就看到了自己的信念状态，"自生成"并不是一个新变量。Talvitie 的结果说明，"自我纠正能力"比一步误差更能预测多步表现（[Talvitie](https://arxiv.org/abs/1612.06018)）。这可以支持把恢复指标当作主要指标，但不支持任何 RSI 叙事。

**唯一允许的受限形式是不带 RSI 名号的测试时自监督纠正，而 BF-Delta 本身就是它。** 记忆在 episode 内用自己可验证的预测误差改进自己对这个世界的预测，验证信号来自环境的后续观测，在综述的验证层级里属于"执行反馈"一档。论文里只写"test-time self-supervised memory correction"，不写"recursive self-improvement"。若将来真要做 RSI，只有一种形态值得考虑：持续学习协议下的快到慢巩固（C5），即不保存原始 episode，比较"蒸馏快权重"与"等 FLOPs 直接训练慢权重"。它至少要配齐八组对照：算力匹配；数据匹配（只累积不替换）；验证器消融（不过滤、同接受率的随机过滤、真值 oracle 过滤、只做 on-policy 暴露）；单轮对 k 轮；3–5 个种子跑完整流水线；与验证器不同的独立指标；copy/静态退化解检查；用 best-of-N 区分"能力扩展"与"sharpening"。这些对照的依据是 BAgger 第一轮可能变差，以及只做暴露的收益更小（[BAgger](https://arxiv.org/pdf/2512.12080v1)；[Persistent Robot WMs](https://arxiv.org/html/2603.25685)）。

## 与 Saining 研究的接口和 office hour pitch

**接口是 Cambrian-S 自己写下却没有实现的那一句。** 它的 surprise 是无记忆的一步余弦误差：预测只来自滑动窗口，不读记忆；阈值按长度分别调过，VSC 为 0.39/0.41，VSR 为 0.25–0.35，论文承认"we report the best results after hyperparameter tuning"；LFP head 离线训练、测试时冻结（[Cambrian-S](https://arxiv.org/html/2511.04670)；[vsc.sh](https://github.com/cambrian-mllm/cambrian-s/blob/058b1b4c96846a229d1c9bb69da6ecf5ceee4a6b/lmms-eval/scripts/vsc.sh)）。作者在 2511.16655 的附录 A 里承认，VSC-Repeat 是"a limitation of our Cambrian-S (w/ Surprise Seg) model"，并计划在未来的 VSC 中加入重复场景（[2511.16655](https://arxiv.org/html/2511.16655)）。在真实长视频基准 VSI-Super-Wild 上，LFP 版本的计数（VOC）是 28.36/26.17，反而低于基础版的 33.81（[VSI-Super-Wild](https://vsi-super-wild.github.io/)）。这与"门控策略设错了"的解释一致。BF-Delta 对 Cambrian-S 的改动只有三处：LFP 留作先验；surprise 改为参照记忆；surprise 改为设定一个测试时学习器的写入强度。三处都不需要重训他的模型。

**它也接上了另外两条线。** Thinking in Space 发现 MLLM 形成的是"一系列局部 world model，而不是统一的全局模型"；先画认知地图能提升约 10%，用真值地图能提升 20–32%（[Thinking in Space](https://vision-x-nyu.github.io/thinking-in-space.github.io/)）。VSC 的"切段—清空—求和"正是一串局部模型；一个跨越 surprise 边界持续存在、知道"见过没有"的记忆，是统一全局模型的最小形态。Solaris 的记忆是 6 个潜帧（24 个真实帧）的滚动窗口，在"两人都转开再转回"的 Memory 探针上与 frame-concat 基线同为 37.5（[Solaris](https://arxiv.org/html/2602.22208)）。BF-Delta 的残差快权重可以挂在任何冻结的潜预测器上，所以在潜空间里评测 Solaris 的 look-away 轨迹是自然的 V7。Cambrian-P 用位姿 token 把帧放进全局一致的 3D 空间（[Cambrian-P](https://cambrian-mllm.github.io/cambrian-p/)）；V6 的 L0（位姿进 key）与 R（视觉进 key）的差别，正好对应"按位置寻址"还是"按内容寻址"这个问题。另外，Saining 主页写明他任 AMI Labs 联合创始人兼 CSO，并在 2026 年春夏休假离开 NYU（[主页](https://www.sainingxie.com/)），约 office hour 前应先确认他本学期是否在校。

**Office hour pitch（约一分钟，英文原稿）：**

> Cambrian-S says surprise should organize memory *and learning*, but its surprise is memoryless: the LFP head compares each frame with a window-only prediction, so re-entering a room you saw ten minutes ago is as surprising as a new one. That is exactly why VSC-Repeat drops from 42% to 3.6% MRA. I want to keep your LFP head as the *prior* and add a tiny test-time fast-weight memory as the *belief*. Their likelihood ratio is a Bayes-factor surprise: it reads "familiar" when memory predicts better than the prior and "changed" when the prior beats memory. Its closed-form change posterior then sets how strongly each memory key is rewritten, which means one hazard rate instead of per-length thresholds, and each write is literally one TTT step. I test it first on Memory Maze with the official probes and change/repeat splices, then drop it into your VSC pipeline with a 20-line patch, leaving your threshold untouched.

| 想问 Saining 的问题 | 想得到什么 |
|---|---|
| 附录 A 说你们在收集带真实重访的长视频。在那之前，相位平移和倒序这两种近重复，能不能作为过渡的不变性检验？哪一种最能说服你"记忆真的必要"？ | 确认 H7 的评测设计，并拿到 V7 的真实视频目标 |
| 记忆参照的 surprise 应该放在 MLLM 内部（让 LFP 读记忆），还是作为外挂的小模块？换到 AMI 的 JEPA 潜空间，key/value 的单位应该是对象级还是 patch 级？ | 决定 V7 是否向模型内部迁移 |
| 做全局模型时，你更看好位姿寻址（Cambrian-P 式）还是内容寻址？用 Solaris 的 look-away 轨迹在潜空间评测，能不能作为下一步？ | 在 L0 与 R 两种 key 之间下方向赌注 |

## 结论

这次重定最重要的变化，是把"记忆感知的 surprise"从一个要靠手调的启发式，变成一个有生成模型来源的量：拿记忆和世界模型比，而不是拿当前帧和上一帧比。一旦参照系换成"先验对信念"，写入强度、变化检测和熟悉度就都由同一个似然比经变化点模型推出。实现上只需要定一个 hazard rate，再做几项闭式校准，不必为每种视频长度各调一个阈值。Memory Maze 的"静止"反而成了优势：episode 内从不变化，所以 CHANGE 拼接上的任何收益，都是闭式门控对训练中从未见过的变化的泛化。这是学习门控在没有变化数据时做不到的，A2s 那一臂则检验它在有变化数据时能否追上。

V6 的结果无论正负都有用。若 A8 在 H1、H2 上胜出，项目第一次得到一个把保护、纠正、熟悉度归因到同一个量的记忆算子，并能直接修复 Saining 组公开承认的失败。若 A4 追平，结论是"参照记忆就够了，不必再参照世界模型"，这本身就收窄了整个 surprise 记忆文献的设计空间。若 A9 追平，答案是在这些基准上非参数熟悉度已经足够，2511.16655 对基准的批评就延伸到了记忆方法层面。三种结局都要求实验用同样的预算、同样的输入、同样的损失，只换写入规则，而这正是 V1–V5.8 一直没有做成的那类实验。
