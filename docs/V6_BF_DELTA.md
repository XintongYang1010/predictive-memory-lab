# V6.1 BF-Delta：让冻结世界模型为记忆算出遗忘门

> 状态：代码与预注册已就绪，**尚无任何 GPU 结果**。阈值见 [PREREGISTRATION_V6.1.md](PREREGISTRATION_V6.1.md)，执行步骤见 [CODEX_V6_HANDOFF.md](CODEX_V6_HANDOFF.md)，调研依据见 [research/V6_1_FINAL_REVIEW.md](research/V6_1_FINAL_REVIEW.md)。

## 1. 只回答一个问题：世界模型的有界记忆什么时候该忘

V1–V5.8 的教训是：一旦把感知设成串行门槛，自动研究就会一直在感知里打转（V5.1–V5.8 八轮都没跑到记忆实验）。V6.1 把研究对象收窄为**世界模型记忆的写入/遗忘规则**，并且所有主终点都是"同一编码器、同一冻结 key、同一状态预算、同一损失下，两条写入规则之差"，感知质量在一阶上相互抵消。

记忆的保持、纠正、保护都由同一个量决定：**每一步往记忆里写多少，以及何时把旧内容判为过时**。线性注意力社区一直在*学习*这两道门（Gated DeltaNet 的衰减门 α 与写入门 β）。V6.1 的主张是：在世界模型里，它们有精确的贝叶斯含义，可以由一个**冻结的无记忆世界模型**直接算出来：

- 衰减门 = 1 − P(世界已变)，写入门 = Kalman 增益；
- P(世界已变) 来自"无记忆预测器 vs 记忆"的带符号对数 Bayes 因子；
- 在 vMF 似然下，这个对数比恰好是 **Cambrian-S 的 surprise 减去记忆自己的 surprise**。

## 2. 方法（`pmr/predictive_memory_v6/bf.py`，无可训练参数）

记号：$z_t$ 为输入，L0 为真值 token 加**被预测帧的位姿**，R 层为冻结特征；$f_\theta$ 为冻结的无记忆一步预测器，相当于 Cambrian-S 的 LFP head；$\phi$ 为 key 网络。

$$r_t=\big(y_{t+1}-f_\theta(z_t)\big)\oslash\sigma_{\rm train}\ (\text{未观测维为 }0),\qquad k_t=\phi(z_t)/\lVert\phi(z_t)\rVert$$

$$m=Sk,\ e=r-m,\ c=n^\top(k\odot k),\ \tau=\tfrac1{1+c},\ v=\bar s(1+\tfrac1c),\ \ell=\tfrac{\lVert e\rVert^2}{2v}-\tfrac{\lVert r\rVert^2}{2}+\tfrac{D_t}{2}\log v,\ \gamma=\sigma(\ell+\operatorname{logit}\pi)$$

- **A8（局部）**：$S\mathrel{+}=[(1-\gamma)\tau e+\gamma(\rho r-m)]k^\top$，$n\mathrel{+}=(k\odot k)\odot(1-\gamma(n-n_0))$
- **A8g（全局）**：$S=(1-\gamma)S(I-\tau kk^\top)+[(1-\gamma)\tau+\gamma\rho]rk^\top$，$n=(1-\gamma)n+\gamma n_0+k\odot k$

其中 $n_0=\bar s/(1-\bar s)$，$\rho=1-\bar s$。$\bar s$ 是"不可约噪声占标准化残差方差的比例"，在未拼接的训练 episode 上用 ML-II 一维网格标定，这是证据最大化，不按结果调参。π 是**每次写入**的 hazard，取 10⁻³，由先验给定。γ 不回传梯度。

生成模型：$r=Mk+\varepsilon$，$\varepsilon\sim\mathcal N(0,\bar sI)$，先验 $Mk\sim\mathcal N(0,(1-\bar s)I)$；每次写入以概率 π 让 $k$ 方向上的值从先验重抽。A8 是这个切换线性高斯模型沿 $k$ 的一步精确后验均值（两分量混合塌缩为一个高斯）。

**已由单测验证的性质**（`tests/predictive_memory_v6/test_bf.py`，float64，同一份代码在 GPU 上以 torch 运行）：

| 测试 | 内容 |
|---|---|
| U1 | γ≡0 与 one-hot key 时退化为 Kalman/RLS 的后验均值 |
| U2 | γ≡1 时 $Sk=\rho r$ |
| U3 | 冷启动 ℓ=0 精确成立，写入后 $Sk=\rho r$ |
| U4 | 等于暴力两分量后验均值 |
| U5 | 与独立实现的 Nassar 观察者逐步一致 |
| U6 | A8g 与 α=1−γ、β=τ 的门控 DeltaNet 递推逐项一致 |
| U7 | chunk 核等于逐步扫描 |
| U8 | ML-II 找回注入的 $\bar s$ |
| U9 | 局部写入不动未观测行 |

## 3. 能主张什么，不能主张什么

这条递推与 Nassar 简化观察者、GPB1 切换 Kalman 滤波、计算门控的 GDN、以 1−γ 为遗忘门的 MesaNet 逐项等价。因此**不主张新算法、新 surprise 或新架构**。可以主张的是一个实例化加四条经验结论：

1. 以冻结无记忆世界模型作为变化分布，闭式算出 GDN 的两道门；
2. key 局部重置与全局重置之争（H4）；
3. 闭式门在训练中从未出现过的变化上，是否不差于学出来的门（H2，包括见过变化数据的最强对手 A2⁺）；
4. 同一个对数比在计数聚合时判定"见过没有"，修复 Cambrian-S 的 VSC-Repeat（H6）。

最近邻工作有 World-T3（RLS 快权重作视频世界模型记忆）、MOCA、GDWM（无符号对数比）、TTT3R、KDN、Titans。逐条差别见调研报告第 4 节；**冻结 FINAL 前必须由人补查 World-T3 全文**。

## 4. 与 TTT、RSI、Saining 研究的关系

- **TTT 是母语**：每次写入是冻结世界模型残差上的一步测试时回归（对角 RLS）。γ 给 TTT 一个闭式答案：何时把测试时学到的知识遗忘、退回预训练模型。
- **RSI 不进 V6**：γ 把记忆和*固定*先验比较，episode 内自我改进先验会破坏这个参照系。V7 可以考虑"计数门控巩固"，但不称 RSI。
- **Saining 线**：
  - Cambrian-S 写了"surprise guides attention, memory, and learning"，但它的 surprise 是无记忆的，所以重访房间和新房间一样"意外"，VSC-Repeat 的 MRA 从 42 降到 0。
  - V6.1 保留 LFP head 作先验，只在**计数聚合**处用记忆参照的对数比否决重复计数，官方边界、LFP 和 0.39 阈值一律不动。
  - Thinking in Space 发现 MLLM 形成的是一串局部世界模型；VSC 的"切段—清空—求和"正是这种局部模型串，一个跨越边界、知道"见过没有"的记忆是全局模型的最小形态。

## 5. 数据（全部公开）

**Memory Maze 离线集**（MIT，Google Drive，9x9 约 100 GB；15x15 只作可选复现）。

- 每条轨迹一个 NPZ，所有键长度均为 1001。`action[t]` 是**导致**第 t 帧的动作。
- `maze_layout` 中 1 为走廊，已去外墙，左下为 [0,0]；`targets_vec` 已旋转到 agent 坐标系。
- 第一人称相机 fovy=80°（dm_control `jumping_ball_with_head.xml`）。
- 可见性不随数据提供，由 FOV + 视线派生，S0 用图像颜色校准，一致率门槛 ≥95%。
- 轴约定在 S0 由数据拟合确定。
- 划分：dev = 训练文件中 `int(sha1(basename)) % 29 == 0` 的约 1k 条；train 取 sha1 序前 12000 条；test = 官方 1k 评测集，只在 FINAL 打开一次。

**三层输入**：

| 层 | 值向量 y（D 维） | 输入 / key | 用途 |
|---|---|---|---|
| L0（T0d、T0、T1） | 前方 5×5 局部格（走廊=1，视线可见才算观测）+ 每个物体 [可见, 可见×自我中心向量]；9x9 时 D=34 | 上一帧位姿、**被预测帧的位姿**、动作、上一帧观测；oracle key = 格子 × 8 个朝向八分区 | 定结论 |
| L1（T2） | 同 L0，加观测噪声（占标准化残差的 0.5）、可见性丢失 5%、误报 1%、每个 episode 一个闪烁物体（方差 4）、位姿噪声 0.1 格、朝向噪声 0.05 rad | 同上（带噪位姿） | H3、H4 |
| R（T3） | 冻结 DINOv2 ViT-S/14：CLS + 2×2 池化，PCA-64 白化 | 上一帧特征 + 动作 | 只复现方向 |

T0 在残差空间注入已知噪声：$\tilde r=\sqrt{1-\bar s}\,r+\sqrt{\bar s}\,\varepsilon$，$\bar s_{\rm inj}\in\{0.25,0.5,0.75\}$。T1 取 0.5。不注入噪声时，常数 β=1 已经站在理想点，任何门控都赢不了（失效模式 F0）。

**评测条件**（拼接点 $t_s=400$；瞬移那一步对所有臂都不写入、不计分）：

| 条件 | 构造 | 理想行为 |
|---|---|---|
| STATIC | A[0:1001] | 用于 STAT 窗口（500–1000）、探针、误报 |
| CONT | A[0:800] | 参照 |
| EXACT | A[0:400] ⧺ A[0:400] | 熟悉：保护 |
| NEWPATH | A[0:400] ⧺ A[600:1000] | 同一世界换起点：保护 |
| CHANGE | A[0:400] ⧺ B[600:1000] | 变了：在冲突的 key 上重置 |
| MOVE（仅 L1） | 一个已隐藏 ≥50 步的物体在 $t_m\in[400,600]$ 被移到 BFS ≥3 格外 | 只纠正这个物体 |

**VSI-SUPER-Count**（HF，Apache-2.0）搭配冻结的 Cambrian-S-7B-LFP（cambrian-s @058b1b4，打日志补丁）：

- 只下载 10 分钟拆分；
- 50 个视频中 sha1 最小的 10 个只用于校准（白化、残差尺度、$\bar s$、各否决臂阈值按 5% 误判熟悉率设定），其余 40 个只在 FINAL 计分；
- 变体有 repeat1/2/5（`-stream_loop N -c copy`）、相位平移 0.5 s、倒序。

## 6. 对照臂（只换门：同一核、同一冻结 key、同一 $S\in\mathbb R^{D\times d_k}$ 加 $n\in\mathbb R^{d_k}$）

| 臂 | 门（衰减 α，写入 β） | 角色 |
|---|---|---|
| A0 | 无记忆 | 下界 |
| A1(β₀) | α=1，β=β₀ ∈ {0.05,0.1,0.2,0.4,0.8,1} | TTT-Linear/DeltaNet；画稳定性–可塑性前沿 |
| A1BF(β₀) | 固定 c=1/β₀−1 的 BF 门 | 无计数的 BF：前沿平移消融 |
| A2 | GDN：(α,β)=σ(MLP(z))，在未拼接 train 上训练 | 标准学习门 |
| A2P | (α,β,s)=σ(MLP(‖e‖²/D, ‖r‖²/D, log c))，在拼接 train 上训练 | 最强公平对手：同样充分统计量，外加变化监督 |
| A3 | β=σ(a·s_slow+b)，s_slow 为 Cambrian-S 式 1−cos | 无记忆 surprise 进入写入 |
| A4 | β=σ(a·‖e‖²/D+b) | 只看记忆误差（Titans、T²Mem 一族） |
| A6 | β=τ（γ≡0） | Kalman/Longhorn 计数；"永不重置"端点 |
| A6L(λ) | 计数带遗忘，λ ∈ {0.99, 0.999} | World-T3 式 RLS |
| A7 | β=γ | SMiLe 式消融 |
| **A8 / A8g** | (5a) / (5b) | **方法（局部 / 全局）** |
| OC / OL | 在拼接 / 揭示处清空（OL 只清被移物体的行） | 完美检测端点 |

VSC 另有 7 个否决臂：M0 官方、A8、A4、无符号 |ℓ|、M1 等字节槽位、A9 等字节 k-NN、A9×10、M3 无界最近邻。

## 7. 指标

- **MPE**：写入前标准化预测误差 $\sum\lVert o\odot(r-Sk)\rVert^2/\sum D_t$。窗口有 POST（拼接后 200 步）、STAT（未拼接第 500–1000 步）、REV（重访后 10 步）、MPOST（揭示后 200 步）。
- **两个归一化端点**：
  - 恢复分数 $\nu_C=\frac{\mathrm{MPE}(A6)-\mathrm{MPE}(arm)}{\mathrm{MPE}(A6)-\mathrm{MPE}(OC)}$，在 CHANGE 上计算；
  - 保持分数 $\nu_R=\frac{\mathrm{MPE}(OC)-\mathrm{MPE}(arm)}{\mathrm{MPE}(OC)-\mathrm{MPE}(A6)}$，在 EXACT∪NEWPATH 上计算；
  - 理想点为 (1,1)；A1 的 β₀ 扫描画出前沿。
- **其他**：
  - 变化检测 AUROC；静态误报（γ>0.5 的步数，每个 episode 应 ≤πT=1）；
  - 闪烁伪变化率（在与 A8 相同的静态误报率下比较 A4）；
  - 揭示纠正误差（格单位）；
  - 官方式探针（只读 S：墙准确率、物体 MSE，第 500–1000 步）；
  - MemoryGain（离开时清空 vs 完整记忆）；
  - VSC：MRA、中位 pred/GT、否决率、逐帧熟悉度 AUROC。
- **统计**：按 episode（VSC 按视频）做配对 bootstrap，1000 次，95% 分位区间。

## 8. 假设与阶梯（阈值见预注册）

| | 主张 | 失败后去向 |
|---|---|---|
| H1 | 在 T0、T1 上，BF 门把稳定性–可塑性前沿向外推（判 A8g；A8 同时报告） | T0 失败：停止整条 BF 线，报告负结果。T0 过而 T1 不过：做一轮 key 容量诊断（F9）后报告 |
| H2 | 闭式门处理训练中没见过的变化，不差于学习门 | A2≈A8：撤回。A2⁺≫A8：收窄为"免训练基线" |
| H3 | 异方差噪声下，以先验为参照的比值优于瞬时误差 | 收窄为"参照记忆的误差就够了" |
| H4 | key 局部重置保护没变的内容 | 把 A8g 作为方法 |
| H5 | 门控在静态数据上没有代价（探针） | 按 SR-TTT 式代价报告 |
| H6 | 记忆参照的对数比在聚合处阻止 VSC 重复计数 | A9≥A8：报告"非参数熟悉度已足够" |

**阶梯**（AGENTS.md 第 5 条；运行器读 `report/gates.json` 强制执行）：

1. S2（T0）的 H1 通过，才能进入 S3。
2. S3（T1）的 H1 通过，才能进入 S4、S5、S6。
3. S0–S4 合计 ≤30 GPUh 是首个决策点，要向用户汇报。
4. R 层只允许一种编码器配置，门槛只跑一次；不通过就把 R 报告为"编码器不足以做这个记忆测试"，不做任何编码器扫描。

## 9. 阶段与 GPU 时（上限，硬上限 160）

| 阶段 | 内容 | 上限 GPUh |
|---|---|---|
| S0 | CPU：单测、数据单测、可见性校准、token cache | 0 |
| S1 | 测速、过拟合、冒烟 | 2 |
| S2 | T0d、T0（oracle key；3 个噪声水平 × 3 个噪声种子；学习门只在 0.5 拟合一次，作描述） | 4 |
| S3 | T1：$d_k$ 容量规则、5 个 φ 种子、学习门 × 5、全臂评测、探针 | 16 |
| S4 | T2：L1、MOVE、闪烁物体 | 8 |
| S5 | T3：DINOv2 特征、R 门槛、5 个种子 | 40 |
| S6 | VSC 官方 base pass 加日志（1 次 7B），用 10 个校准视频标定 | 6 |
| FINAL | Memory Maze test、VSC 变体，一次性 | 30 |
| 预留 | 重跑与调试 | 25 |

全部是算术估计。S1 实测偏差 >2 倍时先更新预算并告知用户。8×A100-80GB 或 4×H800 均可；60 分钟或 repeat5 的 VSC 需要 80 GB 卡。

## 10. 失效模式与护栏（摘要，详见调研报告"十种失效模式"一节）

| 编号 | 失效模式 | 护栏 |
|---|---|---|
| F0 | 无噪声时 β=1 已最优 | T0 注入已知噪声 |
| F1 | 致密 key 使计数失去特异性 | 报告 E\|kᵀk′\|，VSC 上若 ≥0.1 改用等方差统计量 |
| F3 | 异方差"噪声电视" | H3 用相对比较 |
| F4 | R 层感知混叠 | 混叠指数；R-pose 诊断臂 |
| F5 | 高维似然饱和 | ML-II；静态误报诊断 C1 |
| F8 | 不在任何门控递推里训练 φ | φ 只在 A6 下用精确 chunk 核训练一次后冻结 |
| F9 | 容量下限 | $d_k$ 容量规则 |

## 11. 代码地图（`pmr/predictive_memory_v6/`）

| 模块 | 内容 |
|---|---|
| `bf.py` | 全部臂、chunk 核、ML-II（numpy/torch 通用） |
| `maze.py` | 几何、可见性、局部格、key、重访 |
| `prepare.py` | NPZ → token cache、S0 校准 |
| `data.py` | 评测条件、L1 扰动、MOVE、转移张量 |
| `models.py` | $f_\theta$、φ、门、探针 |
| `train.py` | fit_slow、ML-II、φ、学习门 |
| `evaluate.py` | 全臂评测、行级统计 |
| `probes.py` | H5 |
| `features_r.py` | DINOv2、PCA、R 门槛 |
| `vsc.py` | 离线熟悉度与聚合 |
| `experiments.py` | 阶段 DAG 与预算运行器 |
| `report.py` | 假设判定与 gates.json |
| `fake_maze.py` | 仅供测试与 SMOKE 的合成数据 |

## 12. 相对调研报告规格的实现选择（均已写入预注册）

1. **H1 判 A8g**：报告原写 A8。依据是解析预测加合成 one-hot 世界的结果：A8g 为 (0.88, 0.96)，A8 为 (0.30, 1.00)，后者在全局 CHANGE 上必然付出每个旧 key 一次的陈旧误差。局部性单独由 H4 检验。
2. **被预测帧的位姿进入 $f_\theta$ 与 key**：报告说位姿是输入，这里明确为下一帧位姿（真值里程计），使路径积分永远不进记忆问题。
3. **学习门的训练方式**：一律用 TBPTT 250 逐步训练，不依赖 fla 核；φ 用自写的精确 chunk 核，U7 验证它与逐步扫描一致。
4. **T0 的学习门**：只在 $\bar s_{\rm inj}=0.5$ 拟合一个种子，作描述，以守住 S2 的 4 GPUh；H2 只在 T1 判定。
5. **$d_k$ 选定后 φ 的种子**：5 个种子都在 $d_k^*$ 上重训，不复用扫描时的种子 0。
6. **残差尺度下限**：σ 下限为 max(0.05·std(y), 1e-3)，避免近乎常数的维度放大数值噪声。
7. **物体探针输入**：加入 map_coord，因为 S 本身不含当前位姿；墙和物体两个探针同步训练。
8. **MOVE 条件的端点**：OC 在揭示处全局清空；OL 用复位运行的被移物体行替换 A6。
9. **VSC 校准源**：用 50 个 VSC 视频中的 10 个作校准（报告中的退路），FINAL 前需用户确认。
