# 让冻结世界模型为记忆算出遗忘门

V6 应定稿为一个只回答一个问题的研究：**世界模型的有界记忆什么时候该忘、什么时候该护住旧内容**。方法仍是 BF-Delta，但本轮把它收紧成最干净的形式：在一个冻结的无记忆预测器 $f_\theta$（Cambrian-S 中 LFP head 的角色）的标准化残差上维护一个 delta-rule 快权重，用"先验预测对记忆预测"的带符号对数 Bayes 因子算出变化后验 γ，再用计数推出的 Kalman 增益 τ 写入。局部形式只有一行 $S\leftarrow S+\beta(s\,r-Sk)k^\top$，$\beta=\gamma+(1-\gamma)\tau$；全局形式恰好是**门控 DeltaNet 的两道门被闭式算出**，衰减为 $1-\gamma$，写入为 $\tau$。本轮最重要的结论是一次诚实的收缩：这条递推与 Nassar 的简化贝叶斯观察者、GPB1 切换 Kalman 滤波、计算门控的 GDN 和以 $1-\gamma$ 为遗忘门的 MesaNet 逐项等价，所以论文只能主张"门控的来源换成冻结世界模型"这一实例化和五条可证伪的经验结论，不能主张新算法。实验必须先在**注入已知观测噪声**的 Memory Maze 真值 token 上用 oracle key 定结论，因为无噪声时常数 β = 1 已经站在理想点，任何门控都赢不了；之后依次换成学到的 key、异方差 L1 和冻结 DINOv2 特征，最后在 FINAL 中一次性检验 Cambrian-S 的 VSC-Repeat。那里的熟悉度判定必须作用在**计数聚合**上（匹配到旧段就更新、不再追加），而不是作用在分段上，所以全部变体都能从一次 7B 日志 pass 在 CPU 上离线评测。阶段上限合计 131 GPUh（硬上限 160），8×A100-80GB 或 4×H800 可行。TTT 是方法的母语：每次写入是冻结世界模型残差上的一步 RLS 式测试时回归，γ 给出"何时把测试时知识重置回预训练模型"的闭式答案；RSI 整体移出 V6。截至 2026-10-14，可达来源中没有人做过这个组合，也没有 VSC-Repeat 的修复；但 World-T3（2026-10-12，用 RLS 快权重作视频世界模型记忆）的全文、ICLR 2027 投稿池的大部分和三篇种子论文的被引列表都没能读到，冻结预注册前必须由人补查。

## 七道约束收敛到一个问题：记忆何时该忘

AGENTS.md 把 V6 的研究对象定为"world-model memory transition operator"，即有界记忆如何保持、演化、纠正、保护它看不见的对象的信念，并把感知整个排除在外；它也写明了立规矩的原因：V5.1–V5.8 用八轮做感知诊断，从未跑过项目真正要做的记忆实验（仓库内 `AGENTS.md`）。四个动词里，"演化"需要视野外自主运动的逐帧真值，而唯一同时满足许可、规模和算力约束的主数据集 Memory Maze 写明"the maze layout and the locations of the objects do not change"（[Memory Maze 论文](https://arxiv.org/pdf/2210.13383)）。剩下的保持、纠正、保护由同一个量决定：每一步往记忆里写多少，何时把旧内容判为过时。V6 因此收窄为一个问题：**世界模型的记忆何时该忘**。回答方式是在同一个核、同一组冻结 key、同一状态预算、同一损失下只换写入规则，这正是 AGENTS.md 第 3 条要求、V1–V5.8 从未做成的那类对照。

| 约束 | V6 如何满足 | 剩余风险 |
|---|---|---|
| 1 只研究 world model + 记忆 | 研究对象是写入/遗忘门；主指标全部是"同编码器、同冻结 key"下两条写入规则之差；L0 用数据集自带真值；R 层编码器一次定死、不训练；VSC 只改计数聚合 | R 层失败时滑回编码器调试，靠护栏和 oracle 替换挡住 |
| 2 TTT/RSI 接得自然 | 每次写入是冻结世界模型残差上的一步测试时回归；γ ≡ 0 时是 RLS/Kalman 二阶步；γ 决定何时把测试时知识重置回预训练模型；RSI 不进 V6 | "TTT 用于世界模型"本身已被占，只能主张门控规则 |
| 3 窄、具体、未被做过 | 带符号 Bayes 因子由冻结无记忆世界模型与快权重记忆给出，闭式算出 GDN 的两道门并逐 key 作用；同一个量在计数聚合时判熟悉；修复 VSC-Repeat | World-T3 全文、ICLR 2027 大部分投稿和被引列表未能核查 |
| 4 接 Saining 的研究 | LFP head 即先验；vMF 似然下 $\log S_{\rm BF}=\kappa(s_{\rm fast}-s_{\rm slow})$，即 Cambrian-S 的 surprise 减去记忆的 surprise；VSC-Repeat 是作者承认的局限 | 7B-LFP 权重没有许可标签；Saining 2026 年春夏休假 |
| 5 只用公开数据 | Memory Maze 离线集（MIT）、VSI-SUPER-Count（Apache-2.0）、DINOv2（Apache-2.0）；所有扰动与拼接都由脚本确定性派生 | Memory Maze 只在 Google Drive 上，数据许可只有仓库的 MIT |
| 6 ≤8×A100 或 4×H800 | 阶段上限合计 131 GPUh，硬上限 160；VSC 只需约 15 GPUh 的 7B 推理 | 全部是算术估计；60 分钟 VSC 需要 80 GB 显存 |
| 7 易训练、算法简洁 | BF-Delta 没有可训练参数；只训 $f_\theta$ 和 key 网络 $\phi$（各一个两层 MLP），$\phi$ 在 γ ≡ 0 下用精确并行核训一次后冻结；唯一的标定量 $\bar s$ 用一维网格做 ML-II | 每个头一个 $\bar s$ 是否够用尚未检验 |

**被否决的方向，理由都很具体。** 上一版的"value 侧线性生成元"草案需要视野外运动的真值，退为 V7。"TTT 预测器作 surprise 源"与"TTT 快权重作世界模型记忆"作为机制都已被占：前者有 FFN 的 μ+σ 门控（[FFN 2606.26515](https://arxiv.org/html/2606.26515v3)），后者有 DexWorldModel、Shell Game、CoME 和 JEPA-TTT（[DexWorldModel](https://arxiv.org/abs/2604.16484)；[Shell Game](https://arxiv.org/html/2608.30692v1)；[CoME](https://arxiv.org/html/2605.18813v1)；[JEPA-TTT](https://arxiv.org/html/2610.00722)）。**本轮把 Solaris 否决为数据集。** 它的训练集有 332 GB、12.64M 帧、Apache-2.0，但每步标签只有动作，相机字段是偏航/俯仰的增量，没有任何玩家的绝对位置；每个 episode 只有 128–512 帧（≤25.6 秒）；记忆由 VLM 判定；模型训练只支持 TPU，推理要 ≥48 GB 显存（[HF 训练集](https://huggingface.co/datasets/nyu-visionx/solaris-training-dataset)；[Solaris 论文](https://arxiv.org/html/2602.22208v1)；[仓库](https://github.com/solaris-wm/solaris)）。它的 Memory 评测里，另一名玩家在看不见时是**暂停**的，测的是静态再观察，不是视野外运动。Solaris 只保留为论述背景：在 KV 上反传让 Memory 分数从 48.96 降到 37.50，说明训练方式本身会伤记忆（[Solaris 论文](https://arxiv.org/html/2602.22208v1)）。

## 一行递推与九条等价：没有新算法，只有新的门控来源

### 生成模型和递推都可以写在半页纸上

本节的等价关系和推导都来自本轮笔记中的代数（文献中未见逐字给出），论文必须附证明。**生成模型只有三句话。** 第一，把慢模型残差按训练集逐维标准差标准化，使每一维在训练数据上的总方差为 1。第二，观测为 $r_t=Mk_t+\varepsilon_t$，$\varepsilon_t\sim\mathcal N(0,\bar sI)$，其中 $\bar s\in(0,1)$ 是"不可约噪声占残差方差的比例"；本 episode 的修正量先验为 $Mk\sim\mathcal N(0,(1-\bar s)I)$，换成噪声精度单位就是先验精度 $n_0=\bar s/(1-\bar s)$。第三，每次写入以概率 π 让 $k_t$ 方向上的值从先验重新抽取；这是 key 局部、秩 1 的跳变，对正交 key 精确，全局变体则重抽整个 $M$。在这个切换线性高斯模型下，沿 $k$ 的一步精确后验有两个分量："没变"（概率 $1-\pi$）时做一次 Kalman 更新，增益 $\tau=1/(1+c)$；"变了"（概率 π）时从先验出发只看这一次观测，均值为 $\rho r$，$\rho=1-\bar s$。把这个两分量混合塌缩成一个高斯，就是下面的递推。

**记号。** $z_t$ 是输入（L0 为真值 token 加位姿，R 为冻结特征），$a_t$ 为动作；$f_\theta$ 冻结且无记忆；$\sigma_{\rm train}$ 是训练集残差的逐维标准差；每个头的状态是 $S\in\mathbb R^{D\times d_k}$ 和 $n\in\mathbb R^{d_k}$，初值 $S_0=0$、$n_0\mathbf 1$。

$$r_t=\big(z^{\rm val}_{t+1}-f_\theta(z_t,a_t)\big)\oslash\sigma_{\rm train},\qquad k_t=\frac{\phi(z_t,a_t)}{\lVert\phi(z_t,a_t)\rVert}\tag{1}$$

$$m_t=S_{t-1}k_t,\qquad e_t=r_t-m_t,\qquad c_t=n_{t-1}^{\top}(k_t\odot k_t),\qquad \tau_t=\frac{1}{1+c_t}\tag{2}$$

$$\ell_t=\frac{\lVert e_t\rVert^2}{2v_t}-\frac{\lVert r_t\rVert^2}{2}+\frac{D}{2}\log v_t,\qquad v_t=\bar s\Big(1+\frac{1}{c_t}\Big)\in(\bar s,1]\tag{3}$$

$$\gamma_t=\operatorname{sigmoid}\big(\ell_t+\operatorname{logit}\pi\big)\quad(\text{stop-gradient})\tag{4}$$

$$\text{局部 A8：}\quad S_t=S_{t-1}+\big[(1-\gamma_t)\tau_t\,e_t+\gamma_t(\rho\,r_t-m_t)\big]k_t^{\top},\qquad n_t=n_{t-1}+(k_t\odot k_t)\odot\big(1-\gamma_t(n_{t-1}-n_0)\big)\tag{5a}$$

$$\text{全局 A8g：}\quad S_t=(1-\gamma_t)\,S_{t-1}\big(I-\tau_tk_tk_t^{\top}\big)+\big[(1-\gamma_t)\tau_t+\gamma_t\rho\big]r_tk_t^{\top},\qquad n_t=(1-\gamma_t)n_{t-1}+\gamma_tn_0\mathbf 1+k_t\odot k_t\tag{5b}$$

其中 $n_0=\bar s/(1-\bar s)$，$\rho=1-\bar s$。(5a) 等价于 delta 形式 $S_t=S_{t-1}+\beta_t(s_tr_t-S_{t-1}k_t)k_t^\top$，$\beta_t=\gamma_t+(1-\gamma_t)\tau_t$ 是 Nassar 的学习率，$s_t=[(1-\gamma_t)\tau_t+\gamma_t\rho]/\beta_t\le1$ 是一个值收缩。

**式 (3) 的方向与 Liakoni 一致**，即先验预测密度除以信念预测密度：正值表示世界模型比记忆更能解释这次观测，也就是"变了"；负值表示记忆更好，也就是"熟悉"。三项检查直接落在式子上。未写过的 key 有 $c=n_0$，于是 $v=1$、$m=0$、$\ell=0$ 精确成立，写入后 $Sk=\rho r$，上一版的冷启动问题自然消失。$\gamma\equiv0$ 退化为对角 Kalman/RLS（A6），$\gamma\equiv1$ 退化为只记最后一次观测。当 $\lvert\ell\rvert\gg1$ 时，γ 几乎只取 0 或 1，规则读作"做 Kalman 积分，除非 $\ell>\log\frac{1-\pi}{\pi}$（π = 10⁻³ 时约 6.9 nats），否则沿 $k$ 重置"；软门与硬门有没有差别，由诊断 C4（$0.05<\gamma<0.95$ 的比例）直接回答。(5a) 的计数按坐标更新，满足 $n_i'-n_0=(n_i-n_0)(1-\gamma k_i^2)+k_i^2\ge0$，所以**不需要上一版的 clamp**。L0 里视线外的格子和不可见物体的向量是"未观测"，不是"观测到 0"：带观测掩码 $o_t$ 时，$r,e$ 都乘以 $o_t$，式 (3) 的 $D$ 换成 $D_t=\mathbf 1^\top o_t$，(5a) 的写入向量也乘以 $o_t$。

### τ 是 Kalman 增益，π 是每次写入的 hazard，标定不经梯度

**τ 不是超参数，也不是 hazard。** τ 是由计数推出的 Kalman 增益，表示记忆在 $k$ 方向上的相对不确定度。唯一的"基率"是 π，而且它是**每次写入**的 hazard，不是每单位时间的 hazard。如果世界按每步 hazard $h$ 独立变化，一个 Δ 步前写过的 key 已经变过的先验概率约为 $h\Delta$；Δ = 100 时 logit 偏移约 +4.6 nats，与 π = 10⁻³ 时 6.9 nats 的判决边界同一量级。所以上一版"一个按单位时间计的 hazard 跨视频长度共用"并不是式 (4) 实际实现的东西，这句话和对应的旧 H5 都要撤回。确有需要时，可以加一个 $d_k$ 维的年龄向量 $a\leftarrow(a+1)\odot(1-k\odot k)$，令 $\Delta_t=a^\top(k\odot k)$、$\operatorname{logit}\pi_t=\log(1-(1-h)^{\Delta_t})-\Delta_t\log(1-h)$，作为可选的稳健性设置。π 由先验物理给定：每个 1000 步 episode 至多一次变化，取 π = 10⁻³。它只以加性 log-odds 偏移的形式进入，π 变 10 倍只移动 2.3 nats，因此只要 $\lvert\ell\rvert\gg2.3$，决策就对 π 不敏感。π 不能在静态的 Memory Maze 上学习：ML-II 会把 $\hat\pi$ 推向 0，或者让它吸收模型误设；静态训练数据上拟合出的 $\hat\pi$ 只作为"误设指数"报告。

**标定分两步，都是闭式或一维搜索。** 第一步按训练集逐维标准差标准化残差，标准化后先验预测分布正好是 $\mathcal N(0,I)$。第二步对每个头做 **type-II 最大似然**：在未拼接的训练 episode 上跑 γ ≡ 0 的 A6，最大化序贯似然 $\sum_t\log\mathcal N\big(r_t;m_t,\bar s(1+1/c_t)I\big)$，在 $\bar s\in\{0.05,0.10,\dots,0.95\}$ 上做网格搜索。由链式法则，序贯似然就是逐 episode 贝叶斯线性回归的边际似然，所以这是证据最大化，不是按结果调参；它对 one-hot key 精确，对其他 key 用对角近似。上一版用重访对估计 $\sigma_d^2=\tfrac12\mathrm{Var}(r_t-r_\tau)$，本轮降为交叉检验；上一版的温度 $\lambda=D_{\rm eff}/D$ 降为后备，它的正式名字是 power likelihood / tempering，作用是让后验对模型扰动更稳健（[Miller & Dunson](https://arxiv.org/abs/1506.06101v1)），只在静态误报诊断失败时才启用。还要注意，上一版把 λ 乘在整个对数比上，这与精确的 tempering 不同：精确做法要把两个预测分布和计数增量里的噪声方差都乘以 $1/\lambda$，所以重拟 $\bar s$ 是更干净的单一旋钮。T0 注入已知噪声后，ML-II 必须找回注入值，这是一条免费的正确性单测。

**似然选标准化残差上的高斯，不选 vMF 或 Student-t。** 在共享集中度 κ 的 vMF 下，$\log S_{\rm BF}=\kappa(s_{\rm fast}-s_{\rm slow})$，其中 $s=1-\cos$ 正是 Cambrian-S 的 surprise；但 vMF 丢掉了 Occam（对数行列式）项和"记忆在这里还不确定"的方差项，只等价于等方差的 A7 形式，既没有冷启动处理，也推不出"两个预测器都错就重置"。只有在高斯残差似然下，先验预测才比信念预测更宽，这两条性质才自然成立。所以 vMF 只用来和 Cambrian-S 的余弦接口对话。Student-t（逐 key 的 Normal–Inverse-Gamma 噪声后验）是异方差"噪声电视"键的原则性修补，多 $d_k$ 个标量，留作 H3 失败后唯一允许的后续。

### 九条等价引理决定了论文能主张什么

| 引理 | 极限或变体 | 精确等价的已知算法 | 依据 |
|---|---|---|---|
| L1 | γ ≡ 0（A6） | 对角 Kalman/RLS；Longhorn 取 $\beta_t=1/c_t$ 时 $\varepsilon=1/(1+c)$；one-hot key 下即 $\Lambda=n_0I$ 的 ridge/RLS，也就是 $\gamma^{\rm Mesa}\equiv1$ 的 Mesa。key 不轴对齐时，对角形式过度自信（Jensen 不等式） | [Longhorn](https://arxiv.org/html/2407.14207v5)；[MesaNet](https://arxiv.org/html/2506.05233v2) |
| L2 | γ ≡ 1 | β = 1 的 DeltaNet，目标收缩为 $\rho r$ | — |
| L3 | (5a)，one-hot，ρ → 1 | Nassar 的 $\alpha=\Omega+(1-\Omega)/(\hat r+1)$，Ω ≡ γ，$\hat r$ ≡ c；计数递推就是 run-length 递推加先验伪计数。唯一的结构差别是变化分布：Nassar 用结果范围上的均匀分布，这里用世界模型给出的 $\mathcal N(0,I)$。$n_0\to0$ 不能在 BF 内取极限（Lindley 型）：变化分布必须有有限宽度 | [Nassar 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2945906/)（排版公式是图片，按正文给出的极限情形重建） |
| L4 | γ 的形式 | Liakoni 的 $\gamma(S_{\rm BF},m)$，$m=\pi/(1-\pi)$。(5a) 是 Liakoni 一步后验 $(1-\gamma)\pi_B+\gamma P(\theta\mid y)$ 的**精确均值**，只有塌缩回单个高斯是近似；β = γ（A7）是 SMiLe 式速率，不是 Liakoni 式 | [Liakoni 2021](https://lcnwww.epfl.ch/gerstner/PUBLICATIONS/Liakoni21NeuralComputation.pdf)；[SMiLe](https://arxiv.org/pdf/1606.05642) |
| L5 | (5a) + 计数 | 两模态（恒等动力学 / 跳回先验）切换线性高斯模型沿 $k$ 的 GPB1；均值塌缩精确，方差用精度混合代替矩匹配 | [Murphy, Switching Kalman Filters](https://www.cs.ubc.ca/~murphyk/Papers/skf.pdf) |
| L6 | 与 BOCPD/MOCA 比较 | BOCPD 保留 t 个 run-length 分量，(5a) 每步塌缩为一个；MOCA 保留全混合、元学习先验、全局重置。审稿人可预期的归并是"单 run length 的 MOCA 加 delta 记忆" | [BOCPD](https://arxiv.org/abs/0710.3742)；[MOCA](https://papers.neurips.cc/paper_files/paper/2020/file/cc3f5463bc4d26bc38eadc8bcffbc654-Paper.pdf) |
| L7 | 全局、充分统计量形式 | $H'=(1-\gamma)H+\gamma n_0I+kk^\top$、$G'=(1-\gamma)G+rk^\top$ 推出 $H_t=\sum_{t'}\zeta_{tt'}k_{t'}k_{t'}^\top+n_0I$，$\zeta_{tt'}=\prod_{s=t'+1}^{t}(1-\gamma_s)$：正是 MesaNet 的目标和 Eq. 8，取 $\gamma^{\rm Mesa}=1-\gamma$、$\beta^{\rm Mesa}\equiv1$、$\Lambda=n_0I$、$q_t=k_{t+1}$ | [MesaNet](https://arxiv.org/html/2506.05233v2) |
| L8 | 全局、矩匹配形式 (5b) | 与 fla 的递推 $S'=\alpha S(I-\beta kk^\top)+\beta vk^\top$ 逐项对齐：$\alpha=1-\gamma$，$\beta=\tau$，$v=[(1-\gamma)\tau+\gamma\rho]r/\tau$。局部 (5a) 是 $\beta=\gamma+(1-\gamma)\tau$ 的普通 DeltaNet。两者都能在不改动的 fla 核上运行 | [fla naive.py @1fe52b5](https://github.com/fla-org/flash-linear-attention/blob/1fe52b5/fla/ops/gated_delta_rule/naive.py)；[GDN](https://ar5iv.labs.arxiv.org/html/2412.06464) |
| L9 | 过程噪声分类 | 无过程噪声 → A6，增益趋于 0；高斯随机游走 → 常数稳态增益 $K=(-q+\sqrt{q^2+4q})/2$，即 A1；稀疏跳回先验 → A8/A8g；学到的门（GDN、Titans 的 $\alpha_t$）→ 未指定的、依输入的过程噪声 | [Titans](https://arxiv.org/html/2501.00663v1) |

**这些等价关闭了三种主张，换来一个更干净的主张。** L3 和 L5 让"新的更新规则""新的 surprise"无从谈起；L7 和 L8 让"新架构"无从谈起，但同时给出全文最清楚的一句话：**GDN 的两道门有贝叶斯含义，衰减门是 1 − P(变化)，写入门是 Kalman 增益；BF-Delta 用冻结世界模型把它们算出来，而不是学出来。** L6 让"第一个 BOCPD 加元学习预测器"无从谈起，留下的是分布式快权重记忆里的 key 局部重置，以及熟悉度的用法。L9 给了论文一条统一的解释轴："世界模型的记忆应该假设哪种过程噪声？"在此之后，贡献只能是经验性的、实例层面的四件事：以冻结无记忆世界模型作为跳变分布；key 局部重置与全局重置之争；闭式门与学习门在训练中从未出现过的变化上谁更好；熟悉度在计数聚合中的作用。笔记中一次定向检索没有找到"把 GDN 衰减门读作 1 − P(change)"或"用 Bayes 因子计算 GDN/MesaNet 门"的论文，但这只是一次检索，属于弱证据。

### 十种失效模式，各有一个廉价护栏

下表全部是解析推断，还没有任何关于 BF 门控快权重的实证研究可以印证或否定。排在最前的 F0 是协议风险，不是算法风险，上一版没有看到它。

| 编号 | 失效模式 | 机制（推断） | 护栏 |
|---|---|---|---|
| F0 | 确定性情形退化 | 正交 key、无噪声目标时，A1(β = 1) 在重复上写的是真值（误差为 0，写入为空操作），变化后立刻覆盖，已在理想点 (1, 1)；Nassar 的权衡只在"既有要平均的噪声、又有要重置的变化"时存在。Memory Maze 真值 token 在给定位姿与动作时几乎确定 | L0 注入预注册的已知噪声 $\bar s\in\{0.25,0.5,0.75\}$；确定性 L0（T0d）只作 sanity，预测 A8 ≈ A1(β = 1) |
| F1 | 致密 key 使计数失去特异性 | 每次写入都把总量 1 按 $k\odot k$ 摊到所有方向，$c$ 对所有方向约以 $t/d_{k,\rm eff}$ 增长；Occam 项趋于 $(D/2)\log\bar s$（D = 256、$\bar s$ = 0.5 时约 −89 nats），没见过但 LFP 预测得好的内容被判为"熟悉" | 报告非重访对的 $E\lvert k^\top k'\rvert$，< 0.1 才认可计数语义；否则否决改用等方差统计量 $d_t=(\lVert e\rVert^2-\lVert r\rVert^2)/2$ |
| F2 | 近共线 key 携带冲突值 | 两条夹角 θ 的 key 交替写 +1、−1，插值解范数约 2/θ；delta rule 不会发散，但条件数可任意差 | 监控 $\lVert S\rVert_F$ 和随机单位查询的输出范数（C5）；加大或近正交 key |
| F3 | 异方差"噪声电视" | 比值只消掉模型里有的噪声；噪声方差 $s_f\gg\bar s$ 的键上 $E[\ell]\approx0.33\,d_fs_f$（$\bar s$ = 0.5），仍会反复重置；A4 约为 $(1+\rho^2)s_fd_f$ | 预测 A8 比 A4 少触发但不归零，所以旧的"≤ 2× 干净率"绝对标准会失败；改为匹配误报率下的相对比较；唯一允许的后续是逐 key NIG/Student-t |
| F4 | R 层感知混叠 | Memory Maze 没有全局地标，单帧特征的 key 会在相似走廊之间碰撞，变化检测和熟悉度都会误触发；门控修不了寻址 | R-pose 诊断臂（R 的值配真值位姿 key）；报告混叠指数（非重访对中 key 余弦 > 0.9 的比例） |
| F5 | 高维似然饱和 | 不变时 $E[\ell]\approx(D/2)\log\bar s$，变化时 $\approx(D/2)[2/\bar s-2-\log(1/\bar s)]$；D = 38、$\bar s$ = 0.5 时约 −13 和 +25 nats，γ 实际是二值的 | ML-II 标定 $\bar s$；静态误报诊断（每个未拼接 episode 中 γ > 0.5 的次数 ≤ πT = 1）；之后才动 λ；用 log-sigmoid，ℓ 截断到 ±50 |
| F6 | 计数饱和与僵化 | 计数无界增长，τ → 0；对静态 episode 是对的，遇到缓慢漂移会出现锯齿 | 预注册稳健性设置 $n\le n_{\max}$（等价于加随机游走下限）；默认 $n_{\max}=\infty$。重置只沿 $k$，未激励方向不会饱和，这正是定向遗忘修补指数遗忘的经典问题（[forgetting 文档](https://bayesianbandits.readthedocs.io/en/latest/math/forgetting.html)） |
| F7 | 冷启动与早期内容 | 已精确处理；残余风险是 SR-TTT 式"偏向丢弃早期不意外内容"（[SR-TTT](https://arxiv.org/abs/2603.06642)） | 用官方 visible-age 分箱的物体 MSE 检查早期内容保持 |
| F8 | 穿过门控递推训练 $\phi$ | 若让梯度流过 γ，外层损失会去操纵门（比如让 key 碰撞），而 ℓ ∝ D 使 sigmoid 梯度极大 | **不在任何门控递推里训练 φ**：A6 的增益只依赖 key 的因果前缀和，所以 A6 是"只由输入决定门的 DeltaNet"，可用精确的 chunkwise 核端到端训练；每个种子训一次 φ、冻结、各臂共享；门控臂只做推理、逐步精确扫描。这样同时删掉了上一版的块内陈旧问题和 H8 |
| F9 | 容量下限 | 上一版 L0 的 $d_k=32$ 只有 1248 个标量，而 9×9 迷宫按（位姿、动作）键控有约 10³ 个上下文，所有快权重臂都可能贴着 A0 | $d_k$ 按预注册的容量规则从 {64, 128, 256, 512} 中选，冻结后各臂共用 |

## TTT 是母语，RSI 出局，Saining 的接口在 LFP 先验上

### 每次写入都是一步测试时回归

**TTT 在这里是定义，不是装饰。** 记忆的内层损失是 $\tfrac12\lVert r_t-Sk_t\rVert^2$，梯度为 $-e_tk_t^\top$。A8 的一步是沿 $k$ 用计数做预条件的 Kalman 步，外加一个按 γ 加权、把目标从 $r$ 拉向 $\rho r$ 的移动。用 Wang、Shi、Fox 的测试时回归框架来说，记忆就是"solving a weighted regression problem"，设计空间是回归权重、回归器类别和优化器（[Test-time regression](https://arxiv.org/html/2501.12352v3)）。对应到 BF-Delta：回归权重是之后各次重置的 $(1-\gamma)$ 之积，全局为 $\zeta$，局部只沿 $k$；回归器是线性的；优化器是对角 RLS（A8）或精确求解（MesaNet 形式）。可以写进论文的 TTT 句子是："BF-Delta is test-time regression on a frozen world model's residual, whose sample weights are the posterior probability that the episode has not changed since each sample; it gives TTT a closed-form answer to when test-time knowledge should be forgotten in favour of the pretrained model." 这句话回应的是一个有记录的开放问题。TTT3R 已经用了闭式、免训练的 β，但超过 1000 帧时仍要每 100 帧固定重置一次（[TTT3R](https://arxiv.org/pdf/2509.26645v4)）；ASR 则在预测集中度超过其滑动均值时向源模型重置（[ASR, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/79bdd6fe3f012befcc459ad13de65d13-Paper-Conference.pdf)）。BF-Delta 用一个后验概率回答"何时重置、重置多少、重置哪个方向"。不能声称的有三件：超出 TTT-Linear 的"TTT 层"；除 $\phi$ 之外对内循环的元学习；"TTT 用于世界模型"本身。

### RSI 不进 V6，理由不止"每个时长都有标签"

**第一个理由是语义冲突。** γ 把记忆和**固定**的先验比较。如果在 episode 内用记忆挑出的转移去更新 $f_\theta$，"变化"的参照系就会漂移，γ 不再是变化后验；先验的自我改进与 BF 门控在 episode 内互斥。**第二个理由是跨 episode 没有信息增益。** 只在高 $\lvert r\rvert$ 或低 ℓ 的转移上巩固 $f_\theta$，本质是对已标注数据做重要性重加权，必须赢过"同等 FLOPs 在全部数据上训练 $f_\theta$"，在平稳的离线 Memory Maze 和 VSI 上不太可能。**第三个理由是唯一有信息的情形不在阶梯上**：快到慢回路只在无回放的非平稳任务流里携带信息，那是持续学习，不在 L0 → L1 → R 之内。外部证据指向同一方向：已证实的收益跟踪验证器的强度，"self-evolution is not free"（[RSI 综述](https://arxiv.org/abs/2607.07663)）；不过滤的自标注递归会崩塌（[Lee et al.](https://arxiv.org/html/2502.01612v2)）；直接的长跨度监督很便宜，跨度从 100 秒扩到 400 秒只多约 12% 步时（[Memorizon](https://arxiv.org/html/2610.00544v1)）；按准确率门控的课程能把固定状态召回从 0.021 提到 1.000（[2609.16183](https://arxiv.org/html/2609.16183v1)）。论文里只写"test-time self-supervised correction"，不写"recursive self-improvement"。**这是对上一版的修正**：上一版把 BF-Delta 本身说成"唯一允许的受限 RSI 形式"，本轮整体移出。V7 可以考虑"计数门控巩固"：c 高且 ℓ 反复为负的内容视为稳定，反复重置的视为易变，在无回放的持续协议下只把稳定内容蒸进 $f_\theta$。它不叫 RSI，且需要八组对照：算力匹配；数据匹配；验证器消融（无过滤、同接受率的随机过滤、oracle 过滤、只做暴露）；单轮对 k 轮；整条流水线的多种子；独立于验证器的指标；copy/静态基线；用 best-of-N 区分能力扩展与 sharpening（[Persistent Robot WMs](https://arxiv.org/html/2603.25685)）。

### Cambrian-S 写下却没实现的那一半

**接口的事实已逐行核到代码。** Cambrian-S 的 LFP head 是两层 MLP `Linear(H,H) → GELU → Linear(H,1152)`，7B 时约 17.0M 参数；它读当前帧 72 个 token（8×8 网格加 8 个换行 token）的末层归一化隐状态，预测下一帧冻结 SigLIP2-so400m 的投影前特征（双线性缩放到 8×8，1152 维）；训练时 MSE 与余弦损失各以 0.1 的权重加在 LM 损失上，测试时冻结（[cambrian_arch.py L200–205](https://github.com/cambrian-mllm/cambrian-s/blob/058b1b4c96846a229d1c9bb69da6ecf5ceee4a6b/cambrian/model/cambrian_arch.py#L200-L205)；[7B-LFP config](https://huggingface.co/nyu-visionx/Cambrian-S-7B-LFP/raw/main/config.json)）。surprise 是 64 个 token 上的 `1 − mean cos(pred, actual)`（[L428–432](https://github.com/cambrian-mllm/cambrian-s/blob/058b1b4c96846a229d1c9bb69da6ecf5ceee4a6b/lmms-eval/lmms_eval/models/simple/cambrians_vsc.py#L428-L432)）。VSC 在 surprise > 0.39（120 分钟为 0.41）时切段，对每段的 KV 重问整道"How many different X…"，最后把各段答案相加，并且把所有帧的 KV 无压缩地保留（[cambrians_vsc.py L291–579](https://github.com/cambrian-mllm/cambrian-s/blob/058b1b4c96846a229d1c9bb69da6ecf5ceee4a6b/lmms-eval/lmms_eval/models/simple/cambrians_vsc.py#L291-L579)）。论文写了"surprise guides attention, memory, and learning"，实现只用到记忆和分段，并且邀请"a more capable world model"给出更可靠的 surprise（[Cambrian-S](https://arxiv.org/html/2511.04670)）。只有 Cambrian-S-7B-LFP 带 `nfp_head`，0.5B 的配置里没有，官方 VSC/VSR 脚本在小模型上会直接报错（[0.5B config](https://huggingface.co/nyu-visionx/Cambrian-S-0.5B/raw/main/config.json)）。BF-Delta 对 Cambrian-S 的三处改动都不需要重训他的模型：LFP 留作先验；surprise 改为参照记忆；surprise 改为设定一个测试时学习器的写入强度。

**VSC-Repeat 是一个聚合缺陷，所以修复也必须在聚合处。** Bethge 组用 `ffmpeg -stream_loop N -c copy` 把 50 个 10 分钟 VSC 视频额外循环 N = 1、2、5 次，并沿用原真值。Cambrian-S-7B-LFP 的 MRA 从 42.0 跌到 20.6、3.6、0.0，中位 pred/GT 为 0.99、1.85、2.85、5.80，几乎正好等于拷贝数（[supersanity 结果](https://github.com/bethgelab/supersanity/tree/296c8c3908f5e1b8e68e96beec6c8e02ee1928c2/results/vsc_repeat_exps)；[2511.16655](https://arxiv.org/html/2511.16655)）。Cambrian-S 作者回应"we know this is a limitation of our Cambrian-S (w/ Surprise Seg) model"。截至 2026-10-14，官方仓库自 2026-04-03 起、HF 数据自 2025-11-07 起没有变化，ICLR 2026 定稿没有提到 VSC-Repeat，检索也没有发现第三方修复（[ICLR 2026 页面](https://iclr.cc/virtual/2026/poster/10007167)；[HF datasets API](https://huggingface.co/api/datasets?author=nyu-visionx&sort=lastModified&direction=-1&limit=15)）。**本轮纠正了上一版的一个关键设计。** 只压制分段边界修不了 VSC-Repeat，因为重复拷贝会并入一个长段，照样被问、被加。熟悉度判定必须作用在**聚合**上：一段若被判为见过，就匹配到已有的槽位，并把计数更新为 $\max(\text{old},\text{new})$；只有判为没见过才追加新槽位。这只需要官方代码本来就产生的逐段计数，所以官方的边界、LFP 和 0.39 阈值一律不动，记录一次 7B pass 之后，所有记忆变体都可以在 CPU 上离线评测。本轮还从代码确认：episodic 查询会新建 `past_key_values`，patched attention 的 `torch.cat` 返回新张量，所以第 t+1 帧的流式 KV 与边界落在哪里无关（[L516–534](https://github.com/cambrian-mllm/cambrian-s/blob/058b1b4c96846a229d1c9bb69da6ecf5ceee4a6b/lmms-eval/lmms_eval/models/simple/cambrians_vsc.py#L516-L534)；[qwen2_monkey_patch.py L72–78](https://github.com/cambrian-mllm/cambrian-s/blob/058b1b4c96846a229d1c9bb69da6ecf5ceee4a6b/lmms-eval/qwen2_monkey_patch.py#L72-L78)）。上一版担心的"dump 依赖边界"并不成立。

**精确重复几乎可以被哈希解决，这一点决定了评测设计。** `-c copy` 复制的帧逐字节相同，1 FPS 采样在每份拷贝之间只差一个固定相位（推断，未核实）。所以无界最近邻去重基线是必须的，这也是 2511.16655 本身对方法提出的要求。记忆的贡献只能体现在有界预算和非完全相同的重访上，同时原始 VSC 不能掉分。致密 key 带来另一个风险（F1）：在固定随机投影 key 下，LFP 本来就预测得好的新房间可能被误判为熟悉。预注册的规则是：在校准集上，若非重访对的 key 重叠 $E\lvert k^\top k'\rvert<0.1$，用完整的 ℓ 判熟悉；否则改用去掉 Occam 项的等方差统计量 $d_t=(\lVert e_t\rVert^2-\lVert r_t\rVert^2)/2$。这打破了"一个量做三件事"，是致密 key 的诚实代价。

**其他 Saining 线索按直接程度排序。** 第二位是 Cambrian-S 的 VSR 记忆（压缩、巩固、检索），但无记忆检索 NoSense 在 10/30/60 分钟上已达 98.3/98.3/96.7%（[2511.16655](https://arxiv.org/html/2511.16655)），所以不进 V6。Thinking in Space 发现 MLLM 形成的是一串局部世界模型，而不是统一的全局模型（[项目页](https://vision-x-nyu.github.io/thinking-in-space.github.io/)）；VSC 的"切段—清空—求和"正是这样的局部模型串，一个跨越边界持续存在、知道"见过没有"的记忆是全局模型的最小形态。Cambrian-P 的位姿 token 对应 L0（位姿进 key）与 R（内容进 key）之争（[Cambrian-P](https://cambrian-mllm.github.io/cambrian-p/)）；RAE/RAEv2 只为"冻结语义编码器的潜空间"这一选择背书（[RAEv2](https://arxiv.org/abs/2605.18324)）。在真实长视频上，LFP 版本的 VOC 计数为 28.36 和 26.17，反而低于基础版的 33.81（[VSI-Super-Wild](https://vsi-super-wild.github.io/)），与"surprise 门控用错了参照系"的解释相容。一篇 NeurIPS 2026 workshop 论文发现，世界模型记忆库按内容相似度检索比随机还差（[NeurIPS 页面](https://neurips.cc/virtual/2026/166884)），这支持在每个对照里放入等字节 k-NN 或随机槽位一类的控制。Saining 的主页写明他是 AMI Labs 联合创始人兼 CSO，并在 2026 年春夏休假（[主页](https://www.sainingxie.com/)），约 office hour 前要先确认。

**更新后的一分钟 pitch（英文原稿）：**

> Cambrian-S says surprise should guide memory and learning, but its surprise is memoryless, so re-entering a room seen ten minutes ago is as surprising as a new room — which is why VSC-Repeat drops from 42 to 0 MRA. I keep your LFP head frozen as the prior and add a tiny test-time fast-weight memory on its residual. The signed log Bayes factor between the two computes, in closed form, the decay and write gates that Gated DeltaNet learns; under a von Mises–Fisher likelihood it is simply your surprise minus the memory's surprise. I first establish it on Memory Maze ground-truth tokens against budget-matched learned gates, then use the same ratio to decide whether a counted segment is a revisit — at aggregation, without touching your model or your threshold.

## 最近邻是 World-T3、MOCA 与 GDWM，新意只能窄说

**World-T3 是最需要警惕的同期工作，而且它的关键部分读不到。** 它是 2026-10-12 提交的"World-T3: Test-Time Training as Persistent Memory for Real-Time Video World Models"，同时是 ICLR 2027 投稿（OpenReview `515W3FqYxm`）。摘要说它用"a recursive least-squares rule that incrementally absorbs novel visual features via online pose-keyed memory updates"写入快权重，按相机位姿读出；在一个未具名的长时重访基准上把重访 PSNR 从 13.52 提到 18.31 dB，在单块 B200 上以 1280×704 跑到 20.6 FPS（[arXiv abs](https://arxiv.org/abs/2610.17434)；[alphaxiv 概述](https://www.alphaxiv.org/overview/2610.17434)；[OpenReview](https://openreview.net/forum?id=515W3FqYxm)）。它的增益、遗忘因子、是否有写入门或重置、key 是否只用位姿，全部未知：arXiv HTML 和 PDF 返回 429，项目页、HF Papers、OpenReview PDF 都没能打开。能确定的是，摘要、ICLR 2027 列表和各聚合站都没有提到参照（无记忆）预测器、surprise 或似然比、变化/重置/hazard、对象状态变化、潜变量预测或计数。若它用标准 RLS 加固定或学习的遗忘因子，那它已经有 BF-Delta 的两个成分（由协方差推出的逐方向写入，以及常数的全局遗忘），差别就收缩为四条：γ 由一个**独立冻结的无记忆预测器**与记忆之间的**带符号**对数 Bayes 因子闭式算出；重置沿 key 局部作用；目标是冻结潜预测器的残差；同一个量用于计数时的熟悉度（推断）。所以 V6 必须加一条"World-T3 式 RLS"臂（A6λ，带遗忘因子的 RLS，与 A6 同状态预算），并在 2026-11-05 ICLR 2027 审稿意见公开后读审稿人如何评价"RLS 快权重作世界模型记忆"的新意（[ICLR 2027 日程](https://iclr.cc/Conferences/2027/Dates)）。World-T3 解决的问题也不同：它是像素级视频生成、位姿寻址、按现有描述是静态场景；静态重访基准奖励"永不遗忘"，正好与驱动 γ 的变化检测压力相反（推断，只基于摘要）。

| 邻居 | 已有什么 | 与 V6 的确切差别 | V6 中如何回应 |
|---|---|---|---|
| **MOCA**（NeurIPS 2020） | 元学习先验在测试时固定；ALPaCA 的递归贝叶斯线性回归统计量；BOCPD run-length 后验由各 run length 的预测似然驱动（[MOCA](https://papers.neurips.cc/paper_files/paper/2020/file/cc3f5463bc4d26bc38eadc8bcffbc654-Paper.pdf)） | 保留 O(t) 混合、全局重置、元学习任务，未用于世界模型、视频或计数。它是 BF-Delta 结构上的"父方法" | A8g（单 run length、全局重置）与 A8（局部）对照；相关工作直接说明这层关系 |
| **GDWM**（2026-01） | 用两个预测似然的**无符号**对数比 $\lvert\log P(x_t\mid x_{1:t-1})-\log P(x_t\mid x_{t-n:t-1})\rvert$（n = 512）分配测试时训练步数（[alphaxiv](https://www.alphaxiv.org/overview/2601.12906)） | 无符号，分不清"熟悉"与"变了"；两个预测来自同一个 LLM；离线 softmax 分配；LoRA；只有语言 | 无符号 $\lvert\ell\rvert$ 消融：预测它在熟悉与变化之间失败 |
| **KDN / MbD** | Kalman 增益只来自协方差，"innovation … affects only the state correction, not the gain"；MbD 把 DeltaNet 视为结构上的"covariance-reset reduction"（[KDN](https://huggingface.co/papers/2609.07816)；[MbD](https://huggingface.co/papers/2605.31163)） | 对新息不敏感；没有由 surprise 触发的重置 | A6 就是对角 KDN；BF-Delta 是"对角 KDN 加沿 k 的 BF 协方差重置" |
| **Liakoni / Nassar / Wilson–Nassar–Gold 2013** | BF 变化后验；Nassar 学习率；固定速率 delta rule 的混合近似（[Wilson 2013](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1003150)） | 标量任务，没有快权重矩阵，没有冻结世界模型 | 引理 L3–L5；单测 U5 用独立的 Nassar 实现对拍 |
| **TTT3R / Longhorn** | 闭式、免训练的 TTT 学习率；闭式隐式在线更新，β 由输入学习 | 统计量是注意力对齐或学到的输入门，不是预测似然比 | A1/A2 扫描；引用而不主张"闭式学习率" |
| **TTT-RM / T²Mem** | 快权重学习有界记忆或冻结预测器解释不了的残差；按残差缩放 TTT 步（[TTT-RM](https://arxiv.org/abs/2610.04701)；[T²Mem](https://arxiv.org/html/2609.36720v1)） | "冻结预测器上的残差快权重"已被占；它们都不用似然比或变化后验 | A4（只看记忆误差）检验比值是否必要 |
| **Rho-1 / RHO-LOSS / InfoRS** | "只学参照模型预测不了的东西"（超额损失、记忆条件 surprise）（[Rho-1](https://arxiv.org/pdf/2404.07965)；[InfoRS](https://www.arxiv.org/pdf/2204.04763)） | 用于数据选择或回放缓冲，不是测试时快权重写入，也不是变化后验 | 相关工作 |
| **GauMem / TokenWeave**（ICLR 2027 投稿） | 学到的写入策略把新视图与渲染记忆比较；记忆"grows with scene novelty"（[GauMem](https://openreview.net/forum?id=Euc6zrBjEv)；[TokenWeave](https://openreview.net/forum?id=f2ar8mZH9f)，均为目录转述） | 匹配式或学习式，不是似然比；都不报告计数 | 等字节 k-NN（A9）与槽位记忆（M1）在 VSC 上必须在场 |
| **HLA-WM / WorldState / NARRA**（ICLR 2027 投稿） | GDN 世界模型按相机几何检索并重组状态；分离的擦除/写入；多频率测试时自适应（[HLA-WM](https://openreview.net/forum?id=LVhtPhBHbt)，目录转述） | 几何、学习或频率决定保留，不是变化后验 | 说明审稿人已熟悉"世界模型记忆要处理变化"，静态重访 PSNR 不够 |
| **Surprise Forcing / FFN** | 非参数新颖度准入与反馈控制预算；二值 μ+σ surprise 决定是否做 TTT 步（[Surprise Forcing](https://arxiv.org/abs/2607.18436)） | 没有比值、没有快权重或没有计数 | A9 的原型；A5（FFN 式）降为可选 |
| **"Have I Scene This Before?" / StaMina / Never Look Back / SEM** | 3D 几何去重、学到的计数状态维护、按标签和 3D 距离关联、基于似然的事件复用（[Have I Scene](https://arxiv.org/html/2610.05526)；[StaMina](https://www.alphaxiv.org/abs/2609.35416)；[SEM](https://www.biorxiv.org/content/10.1101/541607.full.pdf)） | 都不用 surprise，都没评测 VSC-Repeat | "避免重访重数"不能主张为首创；只主张用记忆自己的带符号预测对数比来判熟悉，且不需要检测器、3D 重建或身份跟踪 |

**可以守住的一句话主张（英文原稿，供论文用）：**

> Building on the Bayes-factor change-point posterior (Liakoni et al., 2021; Nassar et al., 2010) and its run-length-mixture use in meta-learning (MOCA; Harrison et al., 2020), we instantiate a single-state, key-local switching-Kalman memory for world models. A frozen, memoryless latent predictor (the role of Cambrian-S's LFP head) supplies the change distribution, a test-time delta-rule fast weight on its residual is the belief, and the signed log Bayes factor between them computes in closed form the two gates that Gated DeltaNet learns: decay 1 − P(change) and write = Kalman gain. The same signed log-ratio decides at count aggregation whether a segment is a revisit. In budget-matched comparisons against constant-gain, innovation-blind Kalman, forgetting-factor RLS, globally reset, learned-gate and retrieval memories, we test whether it protects consistent content, resets on change types absent from training, and removes the revisit over-counting of memoryless surprise on VSC-Repeat.

这句话要带五个限定。第一，写"to our knowledge, as of 2026-10-14"，并把 GDWM、RHO-LOSS/Rho-1 列为"比值门控学习"，把测试时回归列为框架。第二，明确说**带符号**，并给出无符号消融。第三，不说"世界模型中的 Kalman 快权重"是贡献，因为有 World-T3 和 DexWorldModel。第四，不说"按单位时间的 hazard"，只说"每次写入的 hazard"。第五，VSC-Repeat 的主张必须在近重复拆分上扛过 k-NN 臂，否则改报"非参数熟悉度已足够"。

**核验状态需要说清楚到底查到了哪里。** 本轮能看到的 ICLR 2027 投稿池只有一个公开目录：它在 2026-10-10 从 AcceptOdds 取回 1,383 篇含"world model"短语的候选，筛出 133 篇并逐篇对过 OpenReview 页面。对这 133 篇的标题、转述摘要和关键词做关键词扫描，"surpris""kalman""change-point""hazard""cambrian""supersens""familiar""titans"全部零命中（[目录仓库](https://github.com/hexmSeeU/iclr-2027-world-models)）。但目录的摘要是转述，只写在方法节里的门控会漏掉；不含"world model"短语的投稿，例如线性注意力架构论文或视频 LLM 计数论文，根本不在目录里。OpenReview 论坛页被浏览器验证拦住，API 请求的权限没有得到回复；Semantic Scholar 和 OpenAlex 的被引接口报错或被出口策略拒绝；NeurIPS 2026 的录用列表没有找到；Pith 上 Cambrian-S 的 57 篇引用只显示了 25 篇。因此，"没有人做过"的准确表述是：**在可达来源中，没有发现同时具备冻结无记忆预测器作参照、快权重作信念、带符号闭式变化后验逐 key 控制写入的工作，也没有发现 VSC-Repeat 的修复方法。**

## 实验：先在加噪真值 token 上定结论，再逐层外推

### 两个公开数据集，各有必须写成单测的坑

**主数据集是 Memory Maze 离线集，只有 9x9 和 15x15 两种，没有 11x11 和 13x13。** 每种 30,000 条轨迹，每条 1000 步（1001 帧），按 29k/1k 划分训练与评测，每种约 100 GB，放在一个 Google Drive 文件夹里；仓库 LICENSE 是 MIT，README 没有单独的数据许可；HF 上没有镜像（[README](https://github.com/jurgisp/memory-maze/blob/4030901cef3b7d4e7f92e62099e8be378303dc0a/README.md#offline-dataset)；[LICENSE](https://github.com/jurgisp/memory-maze/blob/4030901cef3b7d4e7f92e62099e8be378303dc0a/LICENSE)；[HF 搜索](https://huggingface.co/api/datasets?search=memory-maze&limit=50)）。每条轨迹是一个 NPZ，README 列出 11 个键，全部带长度 1001 的时间轴；位姿、物体的全局与自我中心位置、迷宫布局每步都给，**可见性不给，必须派生**。数据由脚本化策略生成，策略在动作噪声下导航到随机点，"form spatial loops"，天然带重访；每条轨迹都是一个新的随机迷宫，所以测试表现只能来自 episode 内记忆。代码层面本轮确认了四件事：`maze_layout` 中 1 = 走廊、0 = 墙，去掉了外墙并翻转竖轴（[wrappers.py L159–178](https://github.com/jurgisp/memory-maze/blob/4030901cef3b7d4e7f92e62099e8be378303dc0a/memory_maze/wrappers.py#L159-L178)）；格子索引是 `maze_layout[int(y), int(x)]`（[oracle.py L47–58](https://github.com/jurgisp/memory-maze/blob/4030901cef3b7d4e7f92e62099e8be378303dc0a/memory_maze/oracle.py#L47-L58)）；`targets_vec` 由 `add_egocentric_vector` 生成，**已经旋转到 agent 朝向坐标系**，上一版说"未确认"的这一点现已确认，只剩轴约定要用一个文件核对（[maze.py L78–98](https://github.com/jurgisp/memory-maze/blob/4030901cef3b7d4e7f92e62099e8be378303dc0a/memory_maze/maze.py#L78-L98)）；图像带 2 px 的目标色边框，达成目标时变色（[wrappers.py L213–229](https://github.com/jurgisp/memory-maze/blob/4030901cef3b7d4e7f92e62099e8be378303dc0a/memory_maze/wrappers.py#L213-L229)）。移动由物理驱动（带阻尼的滚球），每个动作的位移不固定；Shell Game 发现只给离散动作时所有骨干都明显退化，给全局位姿时"retrieval is not the bottleneck"（[Shell Game](https://arxiv.org/html/2608.30692v1)）。所以 **L0/L1 必须把真值位姿作为输入**，否则失败会把路径积分和记忆混在一起。官方离线基准冻结状态、训练 4×1024 的 MLP 探针，在评测轨迹第 500–1000 步上计分；RSSM-TBTT 在 9x9 上为墙 94.0%、物体 MSE 6.2，监督 oracle 为 99.7% 和 0.3，之后没有论文再报告这组离线探针数（[Memory Maze 论文](https://arxiv.org/pdf/2210.13383)）。15x15 的离线轨迹被截到 1000 步（在线 episode 为 4000 步），oracle 只有 88.5% 和 9.2，动态范围很小，只作次要的容量对照。

**次数据集是 VSI-SUPER-Count，配冻结的 Cambrian-S-7B-LFP，只在 FINAL 用一次。** 数据集在 HF 上 Apache-2.0、不 gated，400 行、311 GB、24 FPS，10 分钟常规拆分有 50 条，列为 `video_path, question, answer, split, query_times, answers`（[HF 数据集页](https://huggingface.co/datasets/nyu-visionx/VSI-SUPER-Count)）。MRA 是在阈值 θ ∈ {0.50, 0.55, …, 0.95} 上 $\lvert\text{pred}-\text{gt}\rvert/\text{gt}\le1-\theta$ 的平均，乘 100（[utils.py L45–80](https://github.com/cambrian-mllm/cambrian-s/blob/058b1b4c96846a229d1c9bb69da6ecf5ceee4a6b/lmms-eval/lmms_eval/tasks/cambrians_vsc/utils.py#L45-L80)）；pred/GT = 2 时 MRA 正好为 0。有三个坑。第一，若 `$HF_HOME/cambrians_vsc` 不存在，lmms-eval 会把整个 311 GB 的仓库 snapshot 下来（[task.py L936–990](https://github.com/cambrian-mllm/cambrian-s/blob/058b1b4c96846a229d1c9bb69da6ecf5ceee4a6b/lmms-eval/lmms_eval/api/task.py#L936-L990)），所以只下 `10mins.zip`（13.5 GB）和 parquet，并在第一次运行前解压到该目录；任务配置写了 `token: True`，即使仓库不 gated 也需要 HF 登录。第二，7B-LFP 在 HF 上没有许可标签（[HF API](https://huggingface.co/api/models/nyu-visionx/Cambrian-S-7B-LFP)），只能作研究评测、不分发派生物。第三，7B 的 KV 每帧约 4.13 MB，VSC 不设预算地保留全部帧，60 分钟约 14.9 GB，加上约 16 GB 的权重，所以 repeat5 必须用 80 GB 的卡（推断，由配置计算）。按 supersanity 日志，7B-LFP 约每视频分钟 5 GPU 秒，base 加 repeat1/2/5 全部约 8.2 GPUh，GPU 型号未记录（[supersanity 结果](https://github.com/bethgelab/supersanity/tree/296c8c3908f5e1b8e68e96beec6c8e02ee1928c2/results/vsc_repeat_exps)）。

### 六条轨道，按 AGENTS.md 第 5 条的顺序推进

| 轨道 | 输入 | key | 目的 | 运行时机 |
|---|---|---|---|---|
| T0d | 确定性 L0 真值 token | oracle one-hot（格子 × 8 个朝向八分区） | 协议 sanity：预测 A8 ≈ A1(β = 1) ≈ oracle（F0） | S2 |
| T0 | L0 + 残差空间注入噪声，$\bar s_{\rm inj}\in\{0.25,0.5,0.75\}$ | oracle one-hot | 在切换模型精确成立处检验门控；ML-II 必须找回 $\bar s$ | S2 |
| T1 | L0 + 噪声 $\bar s_{\rm inj}$ = 0.5 | 学到的 φ（在 A6 下训一次、冻结、共享） | **主结果** | S3 |
| T2 | L1：观测空间噪声、可见性丢失与误报、每 episode 一个闪烁物体、局部移动拆分 | 在 L1 上重训的 φ | 比值是否必要；局部性 | S4 |
| T3 | R：DINOv2 ViT-S/14，224 px，裁 2 px 边框，CLS + 2×2 池化，PCA-64 | 同样配方的 φ；仅在需要时跑 R-pose 诊断 | 只复现方向 | S5 |
| T4 | VSC-Repeat，Cambrian-S LFP 作先验 | 固定随机投影 | Saining 线的演示 | S6 校准，FINAL 一次 |

每层的结论确立之后才进入下一层。T0 的 H1 失败就停止 BF 这条线，作为负结果报告，不进入 L1 或 R。T0 通过而 T1 失败时，只允许一轮关于 key 容量（F9）的诊断，然后报告。T3 失败而 T0–T2 通过时，跑 R-pose 诊断：通过则报告"内容寻址/混叠限制了 R 层门控"，失败则报告"门控结论未迁移到 R 特征"，两种情况都停止，不做任何池化、分辨率、层、PCA 维数或编码器扫描。T4 要求 T1 的 H1 通过；T3 的成败不阻塞 T4，但要一并报告。

### 十三个对照臂只换门，核、key 和状态预算都相同

所有臂都用同一个 $f_\theta$、同一个冻结的 φ、同一个状态 $S\in\mathbb R^{D\times d_k}$ 加 $n\in\mathbb R^{d_k}$（用不到 $n$ 的臂也照样携带），同一个序贯损失。

| 臂 | 门（衰减 α，写入 β，值 v） | 角色 |
|---|---|---|
| A0 | 无记忆，只用 $f_\theta$ | 下界 |
| A1(β₀) | α = 1，β = β₀ ∈ {0.05, 0.1, 0.2, 0.4, 0.8, 1}，v = r | 普通 TTT-Linear/DeltaNet；随机游走 Kalman（L9）；画稳定性–可塑性前沿 |
| A1+BF(β₀) | β = γ + (1−γ)β₀，v 按 (5a) 收缩，γ 用固定 $c\equiv1/\beta_0-1$ 计算 | 无计数的 BF：前沿平移消融 |
| A2 | GDN：(α, β) = 学到的 MLP(输入)，在未拼接数据上训练 | 标准学习门 |
| A2⁺ | (α, β, s) = MLP(‖e‖²/D, ‖r‖²/D, log c)，在**拼接**训练数据上训练 | 最强的公平对手：同样的充分统计量，外加变化监督 |
| A3 | β = σ(a·s_slow + b)，s_slow = Cambrian-S 式的 1 − cos | 把无记忆 surprise 放进写入 |
| A4 | β = σ(a·‖e‖²/D + b) | 只看记忆误差（Titans、T²Mem、FFN 一族） |
| A6 | β = τ（γ ≡ 0） | Kalman/Longhorn 计数；"永不重置"的归一化端点 |
| A6λ | β = τ，计数带遗忘 $n\leftarrow\lambda n+(1-\lambda)n_0+k\odot k$，λ ∈ {0.99, 0.999} | World-T3 式 RLS |
| A7 | β = γ，v = r | SMiLe 式消融，作单元级检查 |
| **A8** | **(5a)** | **方法（局部）** |
| **A8g** | **(5b)**，即 α = 1−γ、β = τ 的 GDN | **方法的同胞（全局）**；即新意笔记中的 A6g |
| Oracle C | 在拼接处清空，之后按 A6 | 完美检测的恢复端点 |

A3 与 A4 的 (a, b) 是两个标量，在未拼接训练数据上用外层损失拟合。A2、A2⁺、A3、A4 每个跑 5 个种子；闭式臂在给定 φ 时是确定性的，所以比较都在每个 φ 种子内部配对。T4 另加 A9（等字节 k-NN，按新颖度准入）、A9×10（10 倍字节，明确标为非等预算）、M1（等字节槽位记忆）、M3（无界最近邻去重，非等预算参照）和无符号 $\lvert\ell\rvert$ 否决。预算允许时再加次要臂：全协方差 Kalman/MesaNet（缩小 $d_k$ 以对齐预算）、带动量的 Titans（$d_k$ 减半）、GRU 和 FIFO。上一版的 A5（FFN 式二值门）、A10/A11 降为可选。

### 六条假设，每条都写好了证伪后的去向

主指标是**序贯标准化预测误差** MPE：写入前的 $\lVert o_t\odot(r_t-S_{t-1}k_t)\rVert^2$ 在一个窗口内求和再除以观测维数之和。窗口有三个：POST 是拼接后 200 步，STAT 是未拼接 episode 的第 500–1000 步，REV 是真值重访事件后 10 步。两个归一化端点把每个臂放到单位正方形里：恢复分数 $\nu_C=[\mathrm{MPE}(A6)-\mathrm{MPE}(\text{arm})]/[\mathrm{MPE}(A6)-\mathrm{MPE}(\text{Oracle C})]$ 在 CHANGE-POST 上计算，保持分数 $\nu_R=[\mathrm{MPE}(\text{Oracle C})-\mathrm{MPE}(\text{arm})]/[\mathrm{MPE}(\text{Oracle C})-\mathrm{MPE}(A6)]$ 在 EXACT-POST 与 NEWPATH-POST 上计算。理想点是 (1, 1)，A1 的 β₀ 扫描画出稳定性–可塑性前沿。以下阈值全部是提议，没有外部锚点，按 AGENTS.md 第 6 条需要用户批准并另起预注册版本。判定规则：按 episode（T4 按视频）配对 bootstrap，1000 次，95% 分位区间；点估计达到阈值，且区间不跨过 0 的错误一侧；在 5 个 φ 种子中至少 4 个通过。注意旧的 PROPOSED 预注册要求"每个种子都通过"，两者如何取舍由用户决定。

| 编号 | 轨道 | 主张 | 通过条件（提议） | 证伪后 |
|---|---|---|---|---|
| H1 前沿 | T0（每个 $\bar s$）、T1；T3 只看方向 | BF 门控把稳定性–可塑性前沿向外推 | (i) A8 不被任何 A1(β₀) 点支配；(ii) 在 A8 的 $\nu_C$ 处，A8 的 $\nu_R$ 比 A1 前沿的插值高 ≥ 0.2；(iii) 6 个 β₀ 中至少 5 个满足 A1+BF(β₀) 在两轴上都不差于 A1(β₀)；(iv) T0 上 A8 的 $\nu_C\ge0.8$ 且 $\nu_R\ge0.8$ | 门控无用：报告负结果，停止 BF 线，不进 L1/R。T0 过而 T1 不过：一轮 key 容量诊断（F9）后报告 |
| H2 闭式门能泛化 | T1 | 算出来的门处理训练中从未出现的变化，不差于见过变化数据的学习门 | $\nu_C$(A8 与 A8g 中较好者) − $\nu_C$(A2) ≥ 0.3；且 ≥ $\nu_C$(A2⁺) − 0.1；$\nu_R$ ≥ $\nu_R$(A2⁺) − 0.1 | A2 ≈ A8：学习门也能泛化，撤回主张。A2⁺ ≫ A8：收窄为"免训练基线，并量化它损失了多少" |
| H3 比值必要 | T2 | 以先验为参照的比值在异方差噪声下优于瞬时误差 | 把 A4 的阈值设在 A8 于干净数据上的静态误报率处，A8 在闪烁物体上的伪变化率 ≤ 0.5 × A4；以真实变化为正类、闪烁与静态为负类，A8 的 AUROC ≥ A4 + 0.05 | 收窄为"参照记忆的误差就够了"；唯一允许的后续是逐 key NIG/Student-t（F3） |
| H4 局部性 | T2 局部移动拆分（单头）；T1 全局 CHANGE | key 局部重置保护没变的内容 | 局部移动拆分上，未变内容的 $\nu_{R,\rm unch}$(A8) ≥ $\nu_{R,\rm unch}$(A8g) + 0.3，同时被移物体的揭示纠正误差不超过 A8g 的 1.1 倍；全局 CHANGE 上只描述性报告 A8 与 A8g | 局部性不必要：把 A8g（即"GDN 门由世界模型算出"）作为方法 |
| H5 静态无代价 | T1、T3 | 门控在未拼接数据上没有代价 | 只读 S 的官方探针：墙准确率与 {A1 最优 β、A6、A2} 中最好者相差 ≤ 1.0 个百分点，物体 MSE 相差 ≤ 10%；所有快权重臂都优于 A0 | 按 SR-TTT 式代价报告 |
| H6 熟悉度 | T4，FINAL 一次 | 记忆参照的对数比在聚合处阻止 VSC 重复计数 | MRA(N = 1, 2, 5) ≥ 0.8 × A8 自身的 MRA(N = 0)；MRA(N = 0) ≥ 官方复现值 − 5；N = 5 时中位 pred/GT ≤ 1.3；N = 0 视频中被否决的段 ≤ 5%；逐帧熟悉度 AUROC（拷贝 ≥ 2 的帧对拷贝 1 的帧）精确重复 ≥ 0.90、相位平移 ≥ 0.80；倒序只作探索 | 等字节 A9 在精确、相位平移、倒序三项上都 ≥ A8：报告"非参数熟悉度已足够" |

**这张表相对上一版做了五处删改。** 删除旧 H2"L0 上 A8 的 AUROC ≥ A4 + 0.05"，因为两者在 L0 上都会饱和；旧 H6"闪烁 ≤ 2 × 干净率"的绝对标准按构造很可能失败（F3），改为 H3 中匹配误报率的相对比较；旧 H3 冷启动降为单测；旧 H5"一个 hazard 跨长度"随 π 的语义修正一起撤回；旧 H8 软件门槛随"φ 只在 A6 下训练、门控臂逐步精确扫描"一起删除。新增 H4 局部性。有两点推断需要用户在冻结前拍板。第一，在全局 CHANGE 上，局部 A8 会在每个旧 key 第一次被重访时付出一次"陈旧值"误差，所以预测 $\nu_C$(A8) < $\nu_C$(A8g)，H1(iv) 的 0.8 对 A8 可能偏紧；是否改为"A8 与 A8g 中较好者"属于协议选择。第二，H6 中"N = 0 上否决 ≤ 5%"可能因为合法的房间内合并而超标，同时 MRA 反而提高。

**诊断不是假设，但失败时先按 bug 排查。** C1：每个未拼接 dev episode 中 γ > 0.5 的次数 ≤ πT = 1。C2：ML-II 在 T0 找回注入的 $\bar s$，容差 ±0.05；由于格内连续位姿还带来自然方差，期望值应是 $\bar s_{\rm inj}+(1-\bar s_{\rm inj})\hat s_{\rm nat}$，$\hat s_{\rm nat}$ 取 T0d 上的 ML-II 值（推断）。C3：静态数据上 $\hat\pi\approx0$。C4：$0.05<\gamma<0.95$ 的比例。C5：$\lVert S\rVert_F$ 与随机单位查询的输出范数。C6：R 层混叠指数。C7：T4 的 key 重叠 $E\lvert k^\top k'\rvert$。

### 护栏：所有主终点都是两条写入规则之差

**防止滑回感知的第一道护栏是指标设计本身。** 每个主终点都是同一编码器、同一冻结 key 下两条写入规则的对比：MemoryGain（完整记忆对离开时清空）、相对 Oracle C 的恢复、相对 A6 的保持、同 β₀ 下 BF 与非 BF 之差。编码器质量在一阶上相互抵消。第二道护栏是 oracle 阶梯：先在 one-hot key 上把"门错了"和"寻址错了"分开，再引入任何学到的 φ。第三道护栏是 R 层只允许一种编码器配置（DINOv2 ViT-S/14、224 px、裁 2 px、CLS + 2×2 池化、在训练集上 PCA-64；DINOv2 为 Apache-2.0，参照 DINO-WM 用 `dinov2_vits14` 的先例，[dino_wm 配置](https://github.com/gaoyuezhou/dino_wm/blob/0a9492f/conf/encoder/dino.yaml)），并且只跑一次预注册的 sanity 门槛：单帧 PCA-64 特征到可见物体自我中心位置的线性探针 MSE ≤ 0.5 × 常数基线，A0 的 dev 下一潜变量 R² ≥ 0.3（两个数都是占位提议）。不过门槛就把 R 报告为"编码器不足以做这个记忆测试"，V6 在 L0/L1 上下结论。第四道护栏是上游问题只用 oracle 替换来诊断（换真值 token、真值 key、真值位姿），从不换编码器变体，这正是 AGENTS.md 第 2 条的做法。第五道护栏是预算：感知相关的 GPU 时只允许用于一次性特征提取，任何其他编码器 GPU 时都要用户批准并升预注册版本。第六道护栏在 VSC 上：LFP head、SigLIP、0.39 阈值和 128 帧感官窗口都不动，逐段计数的误差作为下限报告、不去修，不微调 MLLM、不重训 LFP；学到的 key 投影只作消融。第七道护栏是探针只读 S，不读计数 n，因为计数记录的访问历史在所有非 BF 臂中相同，会给出平凡的泄漏。最后，按 AGENTS.md 第 4 条，只有记忆指标上的坏例能驱动下一轮，每个坏例最多两轮诊断。

### 阶段与 GPU 时：上限合计 131，首个决策点 30

| 阶段 | 内容 | 上限 GPUh | 进入下一阶段的门槛 |
|---|---|---|---|
| S0 | CPU：下载；NPZ 单测；派生标签（可见性、visage、已见格、重访）；拼接索引；FOV 确定；BF-Delta 单测 U1–U9；Cambrian-S 日志补丁的 CPU 部分 | 0 | 全部单测通过；可见性与图像校准一致率 ≥ 95% |
| S1 | 试点与测速：$f_\theta$、A6 下的 φ、逐步扫描、探针的吞吐；ML-II 冒烟；学习门臂 300 步内过拟合到首损失的 50% 以下 | 2 | 实测与估算偏差 ≤ 2 倍，否则先更新预算并告知用户 |
| S2 | T0d + T0：3 个噪声水平 × 3 个种子；闭式臂只做推理；A2/A2⁺/A3/A4 的门拟合 | 4 | C2；T0d 上 A8 ≈ A1(β = 1)；H1 在 T0 上决定继续或停止 |
| S3 | T1：容量规则选 $d_k$；5 个种子训 φ；学习门臂 × 5 种子；全部臂的 dev 评测；H5 的探针 | 16 | H1、H2、H5 在 dev 上可判定 |
| S4 | T2：L1 扰动、闪烁物体、局部移动拆分 | 8 | H3、H4。**首个决策点（S0–S4 ≤ 30 GPUh）向用户汇报** |
| S5 | T3：DINOv2 特征（约 5–8，主要花在 IO）、$f_\theta$、φ × 5、学习门 × 5、探针、拼接评测；R sanity 门槛；必要时 R-pose 诊断 | 40 | R 门槛只跑一次；H1、H5 的方向 |
| S6 | T4 开发：用与 VSI-SUPER 不相交的校准流跑 7B-LFP 日志 pass；离线标定否决阈值、PCA 白化与残差标准差；补丁回归测试 | 6 | 补丁在 2 个视频上输出逐位一致；阈值冻结 |
| FINAL | 预注册冻结后：Memory Maze test 1k（全部臂推理加探针，约 8–10）；VSC base/rep1/rep2/rep5 日志 pass（约 8.2）、相位平移与倒序 N = 1（约 2.8）、30 分钟 base 作变化对照（约 2.1）；可选 15x15 复现（约 6） | 30 | — |
| 预留 | 重跑与调试 | 25 | — |
| 合计 | | **131**（硬上限 160） | |

这些数都是算术估计，没有实测。估算依据如下。L0 的 φ 和 $f_\theta$ 各约 0.1M 参数，每次 φ 训练约 0.5 GPUh；学习门臂每个种子约 0.3 GPUh；闭式臂只做推理，状态 ≤ 20K 个标量，T = 1000，开销可忽略；H5 的探针每臂每种子约 0.2–0.3 GPUh。DINOv2-S/14 在 224 px 下约 12 GFLOPs/图，9x9 的 3000 万帧约 1–2 个 A100 小时，实际受 NPZ 解压限制，按 5–8 GPUh 计。VSC 按约每视频分钟 5 GPU 秒外推，因为否决在聚合处，**比上一版的约 40 GPUh 降到约 15 GPUh**。GPUh 按仓库现有办法记账（阶段墙钟 × 占用 GPU 数，写入 `$OUT/budget_ledger.json`）。新的 `BUDGET_GPUH` 必须经用户批准后才能替换 `prereg.py` 里旧 V6 的表。4×H800 时上限不变，墙钟约翻倍。

## 本轮对上一版报告的二十处修正

上一版《世界模型记忆 V6 方向重定》的方向（BF-Delta、Memory Maze 为主、VSC-Repeat 为次）保持不变，下表列出本轮按新笔记改掉的地方。凡有冲突，以本轮为准。

| # | 上一版 | 本轮 | 依据 |
|---|---|---|---|
| 1 | VSC 熟悉度在分段边界（L474）用 [t, t+5] 的 ℓ 均值把新段标为熟悉，再从求和中剔除 | 否决作用在**计数聚合**（L564、L572）：熟悉段匹配到旧槽位并取 max(old, new)，否则追加；官方边界与阈值不动；全部变体离线在 CPU 上评测 | 只压边界修不了：重复拷贝会并成长段照样被数 |
| 2 | L0 用确定性真值 token | L0 必须注入预注册的观测噪声；确定性 L0 只作 sanity | F0：A1(β = 1) 已在理想点，旧 H1 不可能通过 |
| 3 | 方法被描述为"新的实例化"，未列等价 | (5a) = Nassar 简化观察者 / GPB1；(5b) = α = 1−γ、β = τ 的 GDN；充分统计量形式 = Λ = n₀I、遗忘门 1−γ 的 MesaNet；A6 = β = 1/c 的 Longhorn；全部写成引理 | 本轮推导，L1–L9 |
| 4 | τ 与 hazard 混称；"一个按单位时间的 hazard 跨长度共用"（旧 H5） | τ 是计数推出的 Kalman 增益；π 是每次写入的 hazard；撤回旧 H5，年龄向量作为可选项 | D7 |
| 5 | σ_d 由重访对估计，λ = D_eff/D 作温度 | 标准化残差加每头一个 ML-II 标量 $\bar s$；重访对估计降为交叉检验，λ 降为 tempering 后备 | D5；Miller & Dunson |
| 6 | 计数更新带 `max(n0, …)` 的 clamp | 逐坐标更新，天然非负，不需要 clamp | D4(a) |
| 7 | 块起点陈旧门控 + chunk 训练，软件门槛 H8 | φ 只在 A6 下用精确并行核训一次、冻结共享；门控臂逐步精确扫描；删除 H8 | F8 |
| 8 | 十个臂（A0–A11） | 新增 A8g、A2⁺、A1+BF(β₀)、A6λ（World-T3 式 RLS）；A5、GRU、FIFO 降为可选 | 新意与算法批判 |
| 9 | 旧 H2：L0 上 A8 ≥ A4 + 0.05 AUROC | 删除（两者都会饱和）；比值必要性移到异方差 L1、在匹配误报率下比较（新 H3） | F3、Q6 |
| 10 | 旧 H6："闪烁 ≤ 2 × 干净率" | 改为相对 A4 的 ≤ 0.5 倍 | F3：绝对标准按构造会失败 |
| 11 | 冷启动作为假设 H3 | 冷启动已被 (3)(5a) 精确处理，降为单测 U3 | D9 |
| 12 | 没有局部性检验 | 新增 H4：局部移动拆分上 A8 对 A8g | Q6 |
| 13 | L0 的 $d_k$ = 32 | 按容量规则从 {64, 128, 256, 512} 选 | F9 |
| 14 | 探针读 [vec(W), n] | 主探针只读 S，[S, n] 作次要 | Q4 |
| 15 | 直接用学到的 φ | 先跑 oracle one-hot key 的 T0 | Q4 |
| 16 | VSC 用完整 ℓ 判熟悉 | 致密 key 下计数失去特异性：key 重叠 ≥ 0.1 时改用等方差统计量 | F1 |
| 17 | VSC 只有 A8 与 A9 | 必须加无界最近邻去重 M3、等字节槽位 M1、无符号 $\lvert\ell\rvert$；各臂阈值统一按校准集 5% 误熟悉率设定 | VSC-Repeat 精确重复可被哈希解决 |
| 18 | `targets_vec` 是否旋转"未确认"；LFP dump 是否依赖边界"需验证" | 两者都已由代码确认（已旋转；不依赖边界），只剩轴约定和回归单测 | maze.py L78–98；cambrians_vsc.py L516–534 |
| 19 | RSI：允许"受限形式 = BF-Delta 本身" | RSI 整体移出 V6；只用"test-time self-supervised correction"措辞；V7 的计数门控巩固不叫 RSI | Q5 |
| 20 | 最近邻为 MOCA、软重置、SEM 等；Solaris 作为 V7 候选 | 新增 World-T3、GDWM、Wilson–Nassar–Gold 2013、ICLR 2027 的 GauMem/TokenWeave/HLA-WM；Solaris 被否决为数据集；VSC 预算从约 40 降到约 15 GPUh，总上限从 138 降到 131 | 新意复核；数据集核验 |

## 冻结预注册前必须由人核对的事项

这些项目在本轮因访问限制没能完成，或者是必须由用户拍板的协议选择。前三项决定新意主张能否成立，必须在冻结之前完成；预注册冻结之前不能开始 S2。

| 事项 | 为什么重要 | 怎么查 | 阻塞 |
|---|---|---|---|
| World-T3 全文 | 若它已有似然比或变化触发的门，主张要大改 | 浏览器打开 arxiv.org/html/2610.17434 或 OpenReview `515W3FqYxm` 的 PDF，搜 forgetting、λ、gain、P_t、covariance、reset、novel、gate、Kalman、delta rule；看 key 是否只有位姿；11-05 之后读审稿意见 | 冻结 |
| ICLR 2027 投稿池（不限"world model"短语） | 目录只覆盖 133 篇含该短语的投稿，转述摘要会漏掉方法节的门 | 登录 OpenReview 检索 "Bayes factor"、"change-point"、"hazard"、"surprise" + "delta rule"、"fast weights" + "reset"、"VSI-SUPER"、"Cambrian-S"、"supersensing"、"repeated video counting"；读 GauMem、TokenWeave、HLA-WM 的原始摘要 | 冻结 |
| 三篇种子论文的被引列表 | 引用者最可能先做了组合 | Google Scholar 或 Semantic Scholar 网页查 Liakoni 2021、MOCA、2511.16655 的"Cited by"；Pith 上 Cambrian-S 隐藏的 32 篇引用 | 冻结 |
| 尚未打开的同期条目 | 标题相关 | 2608.27763（Fast Weight Attention for Continual Learning）、SURGE（OpenReview `rCFPSIjqMf`）、2610.07355（Tracking Is Not Permanence）、2609.37096、OSDN 2605.13473、Erase-then-Delta 2606.26560 | 冻结 |
| 自适应控制里的"误差触发协方差重置" | 若已有 BF 触发的定向重置，局部性也有先例 | 文献检索 | 冻结 |
| KDN 公式、GDWM 细节、Nassar 2010 排版公式 | KDN 公式只来自机器生成的评审页；GDWM 的 Δ_t 用基座还是适配后的模型未知；Nassar 的公式是图片 | 读 arXiv PDF | 冻结（论文写作） |
| Memory Maze 下载与字段 | Drive 文件夹对自动抓取 robots 封禁，文件名、分片和训练/评测子目录未知；实际 NPZ 键列表（是否另有 `reset` 等）；`targets_vec` 轴约定；相机 `fovy`；数据许可只有仓库 MIT | 集群出口须放行 drive.google.com 与 *.googleusercontent.com；先 `gdown --folder`，退路为 rclone；下载一个文件跑 S0 单测；在 pip 安装的环境里读 `env.physics.model.cam_fovy`（MuJoCo 默认 45°，但 XML 可能覆盖）；需要时向作者确认数据许可 | S0 |
| VSI-SUPER 与校准集 | 10 分钟以外拆分的行数未确认；VSC 源视频是否与 VSI-590K 不相交未核；房间片段边界不在 parquet 里；7B-LFP 无许可标签；lmms-eval 要求 transformers 4.39.2 而顶层要求 4.37.0；ScanNet/ScanNet++/ARKitScenes 条款未核 | 查 VSI-590K-MetaInfo；若无法证明不相交，退路是从 50 个 VSC 视频中划出 10 个只作校准、不进 FINAL 计分（需用户批准）；安装时解决版本冲突 | S6 |
| L0 噪声与 L1 扰动设计 | 改变 L0 定义，属于协议变更 | 用户确认：T0 在残差空间混合噪声；L1 的噪声幅度、丢失率、闪烁方差、局部移动规则（见实现规格） | 冻结 |
| 主配置：单头还是因子化头 | 因子化头（布局一头、每个物体一头）让全局重置在因子之间天然局部，会使 H4 在构造上失去意义 | 本报告推荐单头（H = 1、带观测掩码、共享计数）为主，因子化头作次要配置（推断） | 冻结 |
| 阈值总表与种子规则 | 全部是提议 | 用户逐项确认 H1–H6、R 门槛、$d_k$ 规则的 80%、"5 中 4"对"每个种子" | 冻结 |
| 预算表 | 旧 `BUDGET_GPUH` 属于旧 V6 阶段 | 用户批准后写入 `prereg.py`；S1 实测偏差 > 2 倍时更新并告知 | S2 |
| Saining 是否在校 | 主页写 2026 年春夏休假 | 约见前确认 | — |

**哪些结论是推断而不是核实的事实。** 等价引理 L1–L9 是笔记中的推导；文献中没有逐字出现，Nassar 的形式是按正文的极限情形重建的。失效模式 F0–F9 全是解析预测。所有阈值都是按理论预测设定的提议，没有外部锚点。全部算力都是算术估计，没有实测。"精确重复可被哈希解决"依赖于 1 FPS 采样相位的计算，未核实。Memory Maze NPZ 是否由 pydreamer 的 `savez_compressed` 写出未核实。对 World-T3 的定位只基于摘要。ICLR 2027 各邻居的描述来自目录转述（HLA-WM 来自新闻摘要）。C2 的期望值、单头优先的选择、全局 CHANGE 上 A8 劣于 A8g 的预测、VSC 的 π 取值，都是本报告的推断。

## 结论

这一轮改变的不是方向，而是论文能诚实主张什么。BF-Delta 的每一部分都退化为一个已知算法，这看上去削弱了新意，实际上给出了一个比"新的 surprise 门控记忆"更清楚、也更难反驳的问题：线性注意力社区一直在**学习** GDN 的衰减门和写入门，而这两道门有精确的贝叶斯含义，可以由一个冻结的世界模型**算出来**。那么在世界模型的记忆里，算出来的门什么时候胜过学出来的门？尤其是面对训练中从未出现过的变化时。这个问题以前没有人用等预算对照回答过，V6 的五条假设各自切掉它的一个侧面，每条都有失败后的明确去向。

第二个启示是方法论上的。F0 说明，无噪声的静态记忆基准在原理上分辨不了门控设计，因为常数 β = 1 已经最优；注入已知噪声并用 ML-II 找回它，是让"保护与纠正之间的权衡"在实验里真正存在的前提。第三个启示针对 Saining 的线索：VSC-Repeat 的失败出在一个无记忆计数器的聚合环节，所以修复也该放在聚合环节。这让测试变得便宜（一次 7B pass，之后全在 CPU 上），但也意味着哈希就能解决精确重复，科学价值只存在于有界预算和近重复拆分上。三种可能的结局都有信息量：A8 胜出，项目第一次得到一个由世界模型算出遗忘门的记忆算子；A2⁺ 追平，结论是"有变化数据时学习门就够了"；A9 追平，2511.16655 对基准的批评就延伸到了记忆方法本身。

## 实现规格：编码代理可直接照做

本节给出 V6.1 的可执行规格。所有数值阈值都是**提议**，用户批准后写入 `docs/PREREGISTRATION_V6.1.md` 和 `pmr/predictive_memory_v6/prereg.py`（新命名空间 `V61`），旧 V6 的 PROPOSED 表保留不删。遵守 AGENTS.md：修 bug 要单独提交，附回归测试，并在提交信息里写明错在哪里；不在 bug 修复提交里改科学内容；结果归档并只核验一次。

### A. 代码组织（工作分支 `v6-predictive-memory`）

| 文件 | 内容 | 状态 |
|---|---|---|
| `pmr/predictive_memory_v6/mmaze_data.py` | NPZ 惰性加载、dev 划分、派生标签、L0 token、T0 噪声、L1 扰动、拼接 | 新建 |
| `features_r.py` | DINOv2 特征提取与 PCA-64（只跑一次） | 新建 |
| `slow_model.py` | $f_\theta$ 训练与残差标准化 | 新建（取代 `lfp.py`） |
| `bf_delta.py` | `bf_step`、各臂门函数、ML-II、A6 并行训练 φ | 新建 |
| `probes.py` | 官方墙/物体探针 | 新建 |
| `metrics.py` | 复用 `bootstrap`、`auroc`、`mra`、`memory_gain`、揭示纠正；新增 MPE 窗口、$\nu_C$、$\nu_R$、前沿插值、误报、visage 分箱、逐帧熟悉度 AUROC | 扩展 |
| `vsc/`（补丁、构造脚本、离线聚合器） | 对 cambrian-s @058b1b4 的日志补丁；相位平移/倒序构造；离线否决与聚合 | 新建 |
| `experiments.py`、`report.py` | 复用阶段运行器与 `budget_ledger.json` 记账；阶段名换成下表 J | 修改 |
| `worlds_l0.py`、`worlds_l1.py`、`worlds_real.py`、`lfp.py` | 自建生成器、Miniworld、nuScenes/MOT | 搁置，不删除 |

### B. Memory Maze 加载

| 键 | 每步形状 | 类型 | 语义（已由代码核实） |
|---|---|---|---|
| `image` | (64, 64, 3) | uint8 | 第一人称图像；**只有 R 层读取**；裁掉 2 px 边框 → 60×60 |
| `action` | (6,) | one-hot | **上一个**动作：(noop, forward, left, right, forward_left, forward_right)；`action[0]` 无定义 |
| `reward` | () | float | 上一步奖励 |
| `maze_layout` | (9, 9) / (15, 15) | binary | 1 = 走廊，0 = 墙；无外墙；左下为 [0, 0]；索引 `[int(y), int(x)]` |
| `agent_pos` | (2,) | float | 格单位，左下角约 (0.1, 0.1) |
| `agent_dir` | (2,) | float | 单位朝向向量（`absolute_orientation[:2, 1]`） |
| `targets_pos` | (G, 2) | float | 物体全局位置；9x9 时 G = 3，15x15 时 G = 6；episode 内不变 |
| `targets_vec` | (G, 2) | float | 物体相对 agent 的位置，**已旋转到朝向坐标系**，格单位 |
| `target_pos`, `target_vec`, `target_color` | (2,), (2,), (3,) | float | 当前目标；V6.1 不使用 |

```python
LABEL_KEYS = ("action", "reward", "maze_layout", "agent_pos", "agent_dir", "targets_pos", "targets_vec")
def load_labels(path):                      # np.load 对 NPZ 按键惰性读取；L0/L1 永远不解压 'image'
    with np.load(path) as f:
        ep = {k: f[k] for k in LABEL_KEYS}
    assert all(len(v) == 1001 for v in ep.values())
    return ep
# 划分：dev = 训练文件中 int(sha1(basename), 16) % 29 == 0 的约 1k 条；train = 其余；
# test = 官方 1k 评测集，只在 FINAL 打开一次（AGENTS.md 第 7 条）。
# 转移 t -> t+1 由 action[t+1] 驱动；t ∈ [0, 999]。
```

**S0 数据单测**（在约 20 个文件上运行，`python -I`）：(1) 记录 `np.load(f).files` 的实际键列表；(2) 所有键长度为 1001，`image.dtype == uint8`；(3) 对所有 t 有 `maze_layout[int(agent_pos[t,1]), int(agent_pos[t,0])] == 1`；(4) `maze_layout`、`targets_pos` 不随 t 变化；(5) `targets_vec ≈ R(agent_dir)ᵀ (targets_pos − agent_pos)`，确定哪一分量是"前方"（预期分量 [1]，待核实）；(6) 前进动作平均使 `agent_pos` 沿 `agent_dir` 增加，以此确认动作对齐。

### C. 派生标签

| 标签 | 定义 |
|---|---|
| 视线 LOS(p, q) | 在 `maze_layout` 上从 p 到 q 做 DDA，经过任何值为 0 的格（终点格除外）即被挡 |
| 物体可见 $\mathrm{vis}_{t,g}$ | $\lvert\angle(\texttt{targets\_vec}_{t,g})\rvert<\mathrm{FOV}/2$ 且 LOS(agent_pos_t, targets_pos_g)；FOV 由 `cam_fovy` 读出（64×64 图像宽高比为 1，水平与竖直 FOV 相等），并用约 50 条轨迹上的物体颜色像素做校准，报告一致率（门槛 ≥ 95%） |
| visage | 物体 g 自上次可见以来的步数；分箱 [-1, 0, 5, 10, 50, 200, 1000]（与 pydreamer 的 `log_ranges` 一致） |
| 已见格 | 到 t 为止曾进入 LOS 与 FOV 的格，用于 `acc_map_seen` |
| 重访事件 | 当前格以前到过；距上次占用 ≥ 50 步；其间 BFS 距离曾 > 2 格。再加朝向余弦 > 0.7 即为视角重访 |
| oracle key（T0） | `idx = (int(y)*S + int(x))*8 + octant`，`octant = floor(atan2(dir_y, dir_x) mod 2π / (π/4))`；9x9 时 $d_k$ = 648，15x15 时 1800 |

### D. 三层输入与扰动

**L0 token。** 输入 $z_t$ = [pose_t（`agent_pos/S·2−1` 与 `agent_dir`，共 4 维），$a_t$ = `action[t+1]`（6 维），$o_t\odot y_t$，$o_t$]。值向量 $y_t$ 共 D = 25 + 3G 维（9x9 为 34，15x15 为 43）：前方 5×5 局部格（行为前方 0–4 格，列为横向 −2..+2，坐标系取量化到 4 个方向的朝向，迷宫外记为墙；观测掩码为 LOS）；每个物体 [$\mathrm{vis}_{t,g}$（总是观测），$\mathrm{vis}\cdot$`targets_vec`（2 维，仅 vis = 1 时观测）]。位姿只作输入，不作值，因为记忆帮不了位姿的物理噪声。

**T0 噪声（在残差空间，精确可控）。** 先在干净训练数据上标准化残差，再在观测维上做 $\tilde r_t=\sqrt{1-\bar s_{\rm inj}}\,r_t+\sqrt{\bar s_{\rm inj}}\,\varepsilon_t$，$\varepsilon_t\sim\mathcal N(0,I)$，$\bar s_{\rm inj}\in\{0.25,0.5,0.75\}$；$f_\theta$ 在干净数据上训练，输入保持干净。T1 用 $\bar s_{\rm inj}$ = 0.5。

**L1 扰动（观测空间；数值是提议）。** `targets_vec` 与位姿输入加高斯噪声，幅度设为使标准化残差中的噪声占比约为 0.5；可见性随机丢失 p = 0.05；误报可见 p = 0.01；每个 episode 随机选一个**闪烁物体**，它的向量观测加方差为 4（标准化单位）的噪声。**局部移动拆分**：在 $t_m\in[400,600]$ 中选一个时刻，要求物体 g 此前已连续 ≥ 50 步不可见、此后还会再不可见 ≥ 20 步；把 `targets_pos_g` 移到一个 BFS 距离 ≥ 3 的随机走廊格中心，然后对 $t\ge t_m$ 按 S0 单测 (5) 确认的轴约定重算 `targets_vec` 和可见性。揭示时刻 $t_r$ = 移动后该物体第一次可见的步。

**R 特征。** 裁 2 px（64 → 60）后双三次缩放到 224，送入 `dinov2_vits14`，取 CLS（384）和 16×16 patch token 均值池化成的 2×2 网格（4×384），拼接成 1920 维；在训练集上取 1M 帧拟合 PCA-64 后，对每帧做投影，以 fp16 存储（9x9 约 3.8 GB）。$z_t$ 与值均为这 64 维，没有掩码。R 层的 $f_\theta$ 用 MSE 加余弦损失。R-pose 诊断的 key 输入为 [pose_t, $a_t$]。

### E. 拼接（只用 dev/test 构造评测；train 上构造的只用于 A2⁺ 和校准检查）

| 条件 | 构造（$t_s$ = 400） | 理想行为 |
|---|---|---|
| EXACT | A[0:400] ⧺ A[0:400] | 熟悉：几乎不写，保持 A |
| NEWPATH | A[0:400] ⧺ A[600:1000] | 同一世界换起点：已探部分熟悉 |
| CHANGE | A[0:400] ⧺ B[600:1000]，B 为哈希顺序中的下一条 | 冲突的 key 上重置 |
| 连续对照 | A[0:800] | 参照 |

拼接那一步（瞬移）对所有臂都**不写入、不计分**。dev 每种条件 500 条，test 每种 1000 条（每条 test 轨迹在三种条件下都作 A）。

### F. VSC-Repeat：下载、日志补丁、构造、离线聚合

```bash
# 只下 10 分钟拆分，避免 311 GB 的全仓 snapshot；需要 HF token
huggingface-cli download nyu-visionx/VSI-SUPER-Count --repo-type dataset \
    --include "10mins.zip" "test_*.parquet" --local-dir $TMP_VSC
unzip $TMP_VSC/10mins.zip -d $HF_HOME/cambrians_vsc/          # 必须在第一次 lmms-eval 运行之前
# 重复视频（supersanity @296c8c3；脚本里的 ShushengYang/* 预发布 ID 一律换成 nyu-visionx/*）
python3 scripts/perturb_vsc_repeat.py --splits 10mins           # N = 1, 2, 5 -> $HF_HOME/cambrians_vsc_repeat{N}/
export CAMBRIANS_VSC_CACHE_NAME=cambrians_vsc_repeat{N}
# 官方参数：--pretrained nyu-visionx/Cambrian-S-7B-LFP --miv_token_len 64 --si_token_len 729
#           --sensory_window_size 128 --surprise_threshold 0.39 --num_frames -1
```

parquet 列：`video_path, question, answer, split, query_times, answers`；常规拆分用 `split == "10mins"`、`answer`。近重复构造（N = 1，确定性脚本，归档；未测试）有两种。**相位平移**：把副本重编码并丢掉前 12 帧（0.5 秒），再与原视频拼接，因为 `-ss` 配 `-c copy` 会截在关键帧上，所以必须重编码。**倒序**：按采样器步长抽出 1 FPS 帧，倒序后每帧重复 24 次重新编码，再与原视频拼接。

**日志补丁**（cambrian-s @058b1b4 的 `cambrians_vsc.py`，单独提交，并附"补丁前后输出逐位一致"的回归测试）：

| 位置 | 现状 | 补丁 |
|---|---|---|
| L381–383 | 每个视频初始化三个 KV 缓存 | 初始化 `log = {}` |
| L428–432 | `surprisingness_score = 1 − cos` | 记录 `s[t]`；记录 `d_pool[t] = vit_visual_features[t].mean((0,1))`（1152，fp16） |
| L450 | `nfp_head` 预测下一帧 | 记录 `p_pool[t+1]` = 去掉换行列后对 64 个位置取均值（1152，fp16）；`p_pool[0] = NaN` |
| L474 | 边界规则 | **不改**；记录边界下标 |
| L564 | `outputs.append(decoded)` | 另外记录原始字符串和 `to_float` 的结果 |
| L572 | `str(sum(...))` | **不改**；每个视频把 `log` 写成 `<dump>/<video_path 中 / 换成 _>.npz` |

repeat 变体可以关闭 `enable_visual_feature_caching`，避免约 200 GB 的特征缓存（要用 2 个视频回归验证结果不变）。

**离线记忆（CPU）。** 在校准集上拟合 PCA 白化到 D = 256：$x_t=W(d_t-\mu)$，$\hat x_t=W(p_t-\mu)$；残差 $r_t=(x_t-\hat x_t)\oslash\sigma_{\rm calib}$；key $k_t=\mathrm{normalize}(Qx_{t-1})$，Q 是固定随机 128×256（种子固定）。BF 记忆的状态为 256×128 + 128 = 32,896 个标量。**写入延迟一个感官窗口**：在第 t 帧写入 $(k_{t-128},r_{t-128})$，使记忆只装窗口外的内容；第 t 帧用当前状态读 $m_t=Sk_t$，算出 $\ell_t$ 和 $d_t=(\lVert e_t\rVert^2-\lVert r_t\rVert^2)/2$。π 取 5×10⁻³（先验估计：约每 200 帧换一次房间；提议，稳健性 {10⁻³, 10⁻²}）。

```python
def aggregate(seg_frames, seg_counts, X, fam, thr):
    """所有否决臂共用同一聚合规则；fam[j] 越小越熟悉，在段关闭时因果地计算。"""
    slots = []                                    # [mu, count, n_frames]
    for j, (fr, c) in enumerate(zip(seg_frames, seg_counts)):
        mu = X[fr].mean(0)
        if slots and fam[j] < thr:                # 熟悉：更新已有槽位，不追加
            i = max(range(len(slots)), key=lambda i: cos(mu, slots[i][0]))
            s = slots[i]; s[1] = max(s[1], c)
            s[0] = (s[0] * s[2] + mu * len(fr)) / (s[2] + len(fr)); s[2] += len(fr)
        else:
            slots.append([mu, c, len(fr)])
    return sum(s[1] for s in slots)
```

| 否决臂 | 段熟悉度 fam[j] | 预算 |
|---|---|---|
| M0 官方 | 不否决（总是追加） | — |
| A8 | 段内 $d_t$ 的均值（若校准集上 $E\lvert k^\top k'\rvert<0.1$，改用 $\ell_t$ 的均值） | 32,896 |
| A4 | 段内 $\lVert e_t\rVert^2/D$ 的均值 | 同一状态 |
| $\lvert\ell\rvert$ 消融 | −段内 $\lvert d_t\rvert$ 的均值（GDWM 式无符号） | 同一状态 |
| M1 槽位 | −max 余弦(μ_j, 槽位)；K = 127 个槽位 × (256 + 2)，LRU 淘汰 | 等字节 |
| A9 k-NN | −段内 [与库的 max 余弦] 的均值；128 条 256 维，新颖度准入、淘汰最旧 | 等字节 |
| A9×10 | 同上，1280 条 | 10 倍，非等预算 |
| M3 | −段内 [与所有窗口外历史帧的 max 余弦] 的均值 | 无界参照 |

每个臂的阈值 thr 在校准集的非重复流上设为"5% 的段被误判为熟悉"，在 S6 冻结。所有臂都只用窗口外（> 128 帧之前）的内容判熟悉。

### G. 模型与对照臂

**$f_\theta$**：两层 MLP（隐层 256，GELU），输入 $z_t$，输出 D 维；L0/L1 用带掩码的 MSE，R 用 MSE 加余弦；训练后冻结；$\sigma_{\rm train}$ 取训练集上观测项残差的逐维标准差。**φ**：两层 MLP（隐层 256），T1/T2 的输入与 $f_\theta$ 相同，R 的输入为 [$z_t$, $a_t$]，输出 $d_k$ 维并做 L2 归一化。φ **只在 A6 下训练**：$c_t=(n_0\mathbf 1+\sum_{s<t}k_s\odot k_s)^\top(k_t\odot k_t)$ 是因果前缀和，所以 $\beta_t=\tau_t$ 只依赖输入；用 fla DeltaNet 的 chunk 核，取 $v_t=r_t$、$q_t=k_{t+1}$，使输出正好是写入前的预测 $m_{t+1}$（核的函数签名要在固定的 fla 提交 1fe52b5 上核对）；损失为 $\sum_t\lVert o_t\odot(r_t-m_t)\rVert^2$；$c_t$ 和 $\tau_t$ 截断梯度。默认 AdamW 3e-4、500 步 warmup、余弦衰减、weight decay 0.1、梯度裁剪 1.0、batch 32 个 episode × 1000 步、10k 步（S1 测速后可调整）；在 dev 上选 checkpoint；每个 φ 种子训一次，之后冻结共享。

```python
def bf_step(S, n, k, r, o, sbar, logit_pi, arm, beta0=None):
    """S:[B,D,dk] n:[B,dk] fp32；k:[B,dk] 单位向量；r:[B,D] 标准化残差（未观测处为 0）；o:[B,D] 观测掩码。"""
    n0, rho = sbar / (1 - sbar), 1 - sbar
    kk = k * k
    m = torch.einsum('bdk,bk->bd', S, k) * o              # 写入前预测
    e = r - m
    Dt = o.sum(-1).clamp_min(1)
    if beta0 is None:                                     # 计数推出的 Kalman 增益
        c = (n * kk).sum(-1); tau = 1 / (1 + c)
    else:                                                 # A1+BF：固定稳态 c = 1/beta0 - 1
        c = torch.full_like(Dt, 1 / beta0 - 1).clamp_min(1e-6); tau = torch.full_like(Dt, beta0)
    vb = sbar * (1 + 1 / c)
    ell = ((e * e).sum(-1) / (2 * vb) - (r * r).sum(-1) / 2 + 0.5 * Dt * torch.log(vb)).clamp(-50, 50)
    g = torch.zeros_like(ell) if arm == 'A6' else torch.sigmoid(ell + logit_pi)   # 不回传梯度
    G, T = g[:, None], tau[:, None]
    if arm in ('A6', 'A8', 'A1BF'):                       # 局部 (5a)；A6 时 g = 0
        u = o * ((1 - G) * T * e + G * (rho * r - m))
        S = S + torch.einsum('bd,bk->bdk', u, k)
        if beta0 is None:
            n = n + kk * (1 - G * (n - n0))
    elif arm == 'A8g':                                     # 全局 (5b) = alpha=1-g, beta=tau 的 GDN
        S = (1 - G)[..., None] * (S + torch.einsum('bd,bk->bdk', o * T * e, k)) \
            + torch.einsum('bd,bk->bdk', G * rho * r, k)
        n = (1 - G) * n + G * n0 + kk
    elif arm == 'A7':                                      # SMiLe 式：beta = gamma
        S = S + torch.einsum('bd,bk->bdk', o * G * e, k)
        n = n + kk * (1 - G * (n - n0))
    return S, n, dict(m=m, e=e, ell=ell, g=g, tau=tau, c=c, Dt=Dt)
```

| 臂 | 实现 | 可训练参数 | 训练数据 |
|---|---|---|---|
| A0 | $m_t\equiv0$ | 无 | — |
| A1(β₀) | $S\leftarrow S+\beta_0\,(o\odot e)k^\top$ | 无 | — |
| A1+BF(β₀) | `bf_step(arm='A1BF', beta0=β₀)`；β₀ = 1 时与 A1(1) 相同 | 无 | — |
| A2 | $(\alpha,\beta)=\sigma(\mathrm{MLP}(z_t))$；$S\leftarrow\alpha S(I-\beta kk^\top)+\beta\,(o\odot r)k^\top$（fla chunk GDN，$g=\log\alpha$） | 约 1 万 | 未拼接 train |
| A2⁺ | $(\alpha,\beta,s)=\sigma(\mathrm{MLP}([\lVert e\rVert^2/D_t,\lVert r\rVert^2/D_t,\log c]))$；$S\leftarrow\alpha S+\beta\,o\odot(s\,r-\alpha Sk)k^\top$；$n\leftarrow\alpha n+(1-\alpha)n_0+k\odot k$；门依赖状态，用截断 250 步的 BPTT 逐步训练 | 约 1 千 | 拼接 train（EXACT : NEWPATH : CHANGE : 未拼接 = 1 : 1 : 1 : 1） |
| A3 | $\beta=\sigma(a\,s^{\rm slow}_t+b)$，$s^{\rm slow}_t=1-\cos(o\odot\hat z_{t+1},\,o\odot z_{t+1})$ | 2 | 未拼接 train |
| A4 | $\beta=\sigma(a\lVert e\rVert^2/D_t+b)$ | 2 | 未拼接 train |
| A6 | `bf_step(arm='A6')` | 无 | — |
| A6λ | 同 A6，但 $n\leftarrow\lambda n+(1-\lambda)n_0+k\odot k$，λ ∈ {0.99, 0.999} | 无 | — |
| A7 / A8 / A8g | `bf_step(arm='A7' / 'A8' / 'A8g')` | 无 | — |
| Oracle C | 在 $t_s$ 处 $S\leftarrow0$、$n\leftarrow n_0$，其余按 A6；局部移动拆分另报 Oracle L（只把被移物体的行清零） | 无 | — |

所有学习门臂都在冻结的 φ 上、用同一个序贯损失训练；门控臂的推理一律用逐步精确扫描。**ML-II**：对 $\bar s$ 网格中的每个值，在未拼接 train（1k 条即可）上跑 A6，累加 $-\tfrac12\sum_t[\lVert e_t\rVert^2/v_t+D_t\log(2\pi v_t)]$，取最大者。**$d_k$ 容量规则**：在 {64, 128, 256, 512} 中取最小的 $d_k$，使 A6 在 dev STAT 窗口上相对 A0 的 MPE 改善达到 oracle-key A6 改善的 80% 以上；冻结后各臂共用。主配置为单头（H = 1，带观测掩码，所有行共享计数；在静态世界里，给定位姿 key 时掩码基本由位姿决定，所以共享计数的近似较温和，此为推断）。因子化头（布局一头、每个物体一头，各有自己的 n 和 γ）作次要配置，只在 T2 报告。

### H. 指标公式

```text
MPE(arm, W)    = Σ_{(i,t)∈W} ‖o_{i,t} ⊙ (r_{i,t} − S_{i,t−1} k_{i,t})‖² / Σ_{(i,t)∈W} D_{i,t}     # 写入前；A0 时预测为 0
窗口 W          POST = [t_s+1, t_s+200]；STAT = 未拼接 episode 的第 500..1000 步；REV = 每个重访事件后 10 步
ν_C(arm)       = [MPE(A6) − MPE(arm)] / [MPE(A6) − MPE(OracleC)]                    在 CHANGE-POST 上
ν_R(arm)       = [MPE(OracleC) − MPE(arm)] / [MPE(OracleC) − MPE(A6)]                在 EXACT-POST ∪ NEWPATH-POST 上（另分别报告）
ν_R,unch       = 同 ν_R，只计未被移动的行（布局 + 其他物体），用于局部移动拆分
A1 前沿 F(x)    = 6 个点 (ν_C(β₀), ν_R(β₀)) 按 ν_C 排序后对 ν_R 做分段线性插值；x 超出范围时取最近端点
被支配          = 存在 β₀ 使 ν_C(β₀) ≥ ν_C(A8) 且 ν_R(β₀) ≥ ν_R(A8)，并至少一项严格大于
变化检测 AUROC  = 正类 CHANGE，负类 EXACT ∪ NEWPATH；得分为 [t_s+1, t_s+20] 上的均值：
                 BF 臂用 ℓ_t，A4 用 ‖e‖²/D_t，A3 用 s_slow，A2/A2⁺ 用 1−α_t
静态误报 FA     = 未拼接 dev episode 中 γ_t > 0.5 的步数的均值（C1：≤ π·T = 1）
闪烁伪变化率    = 闪烁物体可见且未被移动的步中事件所占比例；A8 的事件为 γ>0.5；A4 的事件为 ‖e‖²/D_t > θ_A4，
                 θ_A4 取使 A4 在干净未拼接 dev 上的 FA 等于 A8 的值
揭示纠正        = 被移物体向量误差（格单位，ŷ = f_θ + σ⊙m）在 t_r（写入前）、t_r+1、+3、+10 的值；
                 never-recovered = t_r+10 时误差 > 1.0 格的 episode 比例（复用仓库中 V4 的定义）
官方探针        = MLP 4×1024，LayerNorm + ELU，Adam lr 3e-4、eps 1e-5，输入截断梯度；
                 墙：输入 [vec(S_t), map_coord_t]，map_coord = [agent_pos/S·2−1, agent_dir]，对 maze_layout 做 BCE，
                     报告全格准确率与已见格准确率；
                 物体：输入 vec(S_t)，回归 targets_vec（G×2），报告各物体平均 MSE，并按 visage 分箱；
                 只在第 500..1000 步计分；A0 的输入为 [o_t⊙y_t, o_t]；次要报告 [S, n] 的探针
MemoryGain     = REV 窗口内"离开时清空记忆"的 MPE − 完整记忆的 MPE（R2M-Bench 对照对）
key 重叠        = 非重访对（|t−t'| ≥ 50 且格子不同）上 E|k_tᵀ k_t'|；混叠指数 = 其中余弦 > 0.9 的比例
软门比例 C4     = 0.05 < γ_t < 0.95 的步所占比例
MRA            = 100 · mean_items mean_{θ∈{0.50,…,0.95}} 1[|pred − gt|/gt ≤ 1 − θ]   # 与官方代码一致
重复不变性差    = ΔMRA_N = MRA(repN) − MRA(base)
逐帧熟悉度 AUROC = 正类：拷贝 ≥ 2 中的帧；负类：拷贝 1 中的帧；得分为 −d_t（A8）或 max 余弦（k-NN 类）
统计            = 按 episode（VSC 按视频）配对 bootstrap，1000 次，95% 分位区间（复用 metrics.bootstrap）
```

### I. 单测（CPU，任何 GPU 运行之前）与诊断

| 编号 | 内容 | 容差 |
|---|---|---|
| U1 | γ ≡ 0、one-hot key：$Sk_j=\sum r/(n_0+N_j)$，且等于 A6 | 1e-6（fp64） |
| U2 | γ ≡ 1：$Sk=\rho r$ | 1e-6 |
| U3 | 冷启动：$\ell=0$，写入后 $Sk=\rho r$ | 1e-6 |
| U4 | (5a) 的均值等于暴力两分量后验（留下做 Kalman / 变化后从先验开始）的均值，随机抽样检验 | 1e-6 |
| U5 | 一维变化点流上，逐 key 速率等于一个独立实现的、变化分布为高斯的 Nassar 观察者 | 1e-6 |
| U6 | A8g 经未修改的 fla GDN naive 递推（g = log(1−γ)，β = τ，v = [(1−γ)τ+γρ]r/τ）复现 (5b) | 1e-5 |
| U7 | A6 的并行 chunk 核（β = τ，$q_t=k_{t+1}$）等于逐步扫描 | 1e-4（fp32） |
| U8 | 合成 one-hot 数据上 ML-II 找回注入的 $\bar s$ | 0.05 |
| U9 | 局部写入不改动未观测行 | 精确 |
| U10 | MRA 四例：(10, 10) → 1.0，(15, 10) → 0.1，(10.5, 10) → 1.0，(0, 10) → 0.0（沿用现有 GATES） | 精确 |
| U11 | Memory Maze NPZ 单测 (1)–(6) | 见 B 节 |
| U12 | Cambrian-S 日志补丁：补丁前后 2 个视频的输出逐位一致；关闭特征缓存后结果不变 | 精确 |

诊断 C1–C7 的定义见正文。任何单测或诊断失败，都先按实现错误排查，再作为发现报告（AGENTS.md 第 6 条）。

### J. 阶段表与 GPU 时上限（提议，需用户批准后写入 `BUDGET_GPUH`）

| 阶段 | 轨道与内容 | 上限 GPUh | 门槛 |
|---|---|---|---|
| S0 | 下载；U1–U12；派生标签；拼接索引；FOV | 0 | 全部通过；可见性校准一致率 ≥ 95% |
| S1 | 测速与冒烟 | 2 | 偏差 ≤ 2 倍，否则更新预算并告知用户 |
| S2 | T0d、T0 | 4 | C2；T0d 预测成立；T0 上的 H1 |
| S3 | T1 | 16 | H1、H2、H5（dev） |
| S4 | T2 | 8 | H3、H4；首个决策点 ≤ 30，向用户汇报 |
| S5 | T3 | 40 | R 门槛只跑一次；H1、H5 的方向 |
| S6 | T4 校准 | 6 | U12；阈值冻结 |
| FINAL | Memory Maze test；VSC 日志 pass（base、rep1/2/5、相位平移、倒序、30 分钟 base）；可选 15x15 | 30 | — |
| reserve | 重跑与调试 | 25 | — |
| 合计 | | 131（硬上限 160） | |

运行器沿用 `experiments.py` 的 `run_stage` 与 `_ledger`：拒绝在没有 `--prereg-commit` 的情况下启动 S2 及之后的阶段；按"阶段墙钟 × 占用 GPU 数"记账到 `$OUT/budget_ledger.json`；不触碰其他用户的作业和目录（AGENTS.md 第 12 条）。7B-LFP 的 60 分钟视频必须跑在 80 GB 显存的卡上。

### K. 预注册阈值（提议，镜像到 `prereg.py` 的 `V61`）

```python
V61 = {
  "status": "proposed",
  "rules": {"bootstrap_draws": 1000, "ci": 0.95, "unit": "episode (VSC: video)", "paired": True,
            "pass_requires": "point estimate meets threshold AND CI on the correct side of 0",
            "seed_rule": "pass in >= 4 of 5 phi seeds",          # 旧 PROPOSED 为每个种子；由用户决定
            "phi_seeds": 5, "learned_gate_seeds": 5, "t0_noise_seeds": 3},
  "calib": {"pi": 1e-3, "pi_robustness": (1e-4, 1e-2), "pi_vsc": 5e-3,
            "sbar_grid": [round(0.05 * i, 2) for i in range(1, 20)],
            "t0_sbar_inj": (0.25, 0.5, 0.75), "t1_sbar_inj": 0.5,
            "dk_grid": (64, 128, 256, 512), "dk_frac_of_oracle_gain": 0.80,
            "ell_clip": 50.0, "n_max": None, "lambda_temper_fallback": "only if C1 fails"},
  "l1": {"obs_noise_frac": 0.5, "vis_dropout": 0.05, "vis_false_pos": 0.01, "flicker_var": 4.0,
         "move_window": (400, 600), "move_hidden_before": 50, "move_hidden_after": 20, "move_min_bfs": 3},
  "H1": {"not_dominated": True, "nuR_margin_at_matched_nuC": 0.20, "a1bf_ge_a1_min_count": 5,
         "t0_nuC_min": 0.80, "t0_nuR_min": 0.80, "applies_to": "A8"},   # 是否改为 max(A8, A8g) 由用户决定
  "H2": {"nuC_gap_vs_A2_min": 0.30, "nuC_vs_A2plus_min_diff": -0.10, "nuR_vs_A2plus_min_diff": -0.10,
         "applies_to": "better of A8, A8g"},
  "H3": {"flicker_false_change_ratio_max": 0.50, "auroc_gap_vs_A4_min": 0.05},
  "H4": {"nuR_unch_gap_vs_A8g_min": 0.30, "moved_reveal_err_ratio_max": 1.10, "config": "single head"},
  "H5": {"walls_pts_tol": 1.0, "objmse_rel_tol": 0.10, "ref_arms": ("A1_best", "A6", "A2"),
         "all_fast_beat_A0": True},
  "H6": {"repro_tol_mra": 2.0, "mra_rep_frac_of_own_base_min": 0.80, "mra_base_drop_max_pts": 5.0,
         "median_pred_gt_N5_max": 1.30, "veto_rate_N0_max": 0.05, "key_overlap_max": 0.10,
         "fam_auroc_exact_min": 0.90, "fam_auroc_phase_min": 0.80, "false_familiar_calib": 0.05,
         "kill_if": "A9 equal-byte >= A8 on exact, phase-shift and reversed"},
  "DIAG": {"C1_fa_per_episode_max": 1.0, "C2_sbar_tol": 0.05, "C4_soft_band": (0.05, 0.95)},
  "R_GATE": {"probe_mse_frac_of_const_max": 0.50, "A0_next_latent_r2_min": 0.30, "runs": 1},
  "GATES": {"s0_vis_agreement_min": 0.95, "overfit_ratio_max": 0.50, "unit_tol_fp64": 1e-6,
            "gdn_vs_fla_max_abs": 1e-5, "chunk_vs_scan_max_abs": 1e-4, "profile_dev_max_factor": 2.0},
  "BUDGET_GPUH": {"S1": 2, "S2": 4, "S3": 16, "S4": 8, "S5": 40, "S6": 6, "FINAL": 30, "reserve": 25,
                  "first_decision_point": 30, "hard_cap": 160},
}
```

