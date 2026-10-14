# V6：让记忆在看不见时继续推演世界

依据：`docs/V6_RESEARCH_REPORT.md`（2026-10-13 深度调研 + 补充核验，含全部引用）。
本文是 V6 的方法、协议与执行规格；阈值见 [PREREGISTRATION_V6.md](PREREGISTRATION_V6.md)，
集群执行步骤见 [CODEX_V6_HANDOFF.md](CODEX_V6_HANDOFF.md)。代码在 `pmr/predictive_memory_v6/`。

## 1. 一句话

**问题。** 世界有一部分看不见时，world model 的记忆能不能保持"世界此刻是什么样"的信念：
被遮挡的对象按动力学继续演化（而不是暂停）；重新看见时只纠正这个对象、不扰动其他记忆；
surprise 只在真正违背预期时出现。

**方法：predictive associative memory。** 一个多头 delta-rule fast-weight 矩阵 $S\in\mathbb R^{d_v\times d_K}$，每步三件事：

| 步骤 | 公式 | 含义 |
|---|---|---|
| predict | $S\leftarrow K S,\;K=\exp(\Delta t\,L_{dyn}+L_{ego}(u))$ | 学到的、含 shear 的线性生成元在 **value 侧** 推进全部存储内容；无观测时也执行 |
| compare | $r=Sk,\;\hat y=h(\hat p),\;e=[1-\hat\pi,\,-\hat\tau,\,J_h(y\ominus\hat y)]$ | 记忆自己的现在时预测与观测之差；同一个残差就是 surprise |
| correct | $S\leftarrow S+[e_{book};\,\beta\,G\,J_h e_{pos}]\,k^\top$ | delta 写入 = 对记忆自身观测预测损失做一步（value 空间预条件的）TTT |

TTT 不是外挂：当 $G=C^\top$、$\beta=1$、$h$ 为恒等时，写入严格等于 $-\nabla_S\tfrac12\|y-CSk\|^2$
（单测 `test_ttt_equivalence*`）。V6 对 TTT 层只加了一样东西：inner step 之间的 predict 算子——
世界模型记忆要拟合的映射本身随时间运动。RSI 不进入 V6（生成器在任意时长都给真值，自标注不增加信息；见第 11 节）。

## 2. 为什么从 V5.x 转到这里

V1–V3 编辑冻结 RealMotion 的预测 bank，没有可靠部署收益；V5–V5.8 在"冻结 I-JEPA 特征能否读出两帧位移"
这一前置门槛上打转，原定的 memory 主实验一次也没跑。V6 **完全拿掉感知**：只在对象 token 上研究记忆的转移算子。
"失败也许来自特征"这一歧义在结构上不可能出现。V4 已给出方向：window/FIFO 遗忘早期历史；GRU/slot 在 4× 长度漂移；
恒速解析跟踪始终准确——任务能被线性 predict/correct 解决，学到的非线性记忆却在外推时漂移。

V6 的问题因此是：**一个学到的、叠加的、没有 slot 的联想记忆，能否同时充当许多对象的那个线性滤波器，并在 9 倍训练长度上不漂移？**

## 3. 实现细节（相对调研报告的落地决定）

| 项 | 实现 | 理由 |
|---|---|---|
| value 通道 | $d_v=8$：$[\pi,\tau,p_x,p_y,q_x,q_y,z_1,z_2]$ | presence 兼作齐次坐标；age 由 $\dot\tau=\pi$ 计时 |
| 多头 | 4 个 32 维单位键拼接后除以 $\sqrt4$，等价于 $d_K=128$ 的单个记忆；状态 $8\times128=1024$ 标量 | 与全部对照对齐 1024 标量预算 |
| 结构化生成元 | $L_{\tau\pi}=1$；$L_{pp}=\omega_pJ$；$L_{pq}=B$（初值 $I$，即 shear）；$L_{qq}=\omega_qJ-\mathrm{diag}(\mathrm{softplus}\,d_q)$；$L_{zz}$ 同形 | 谱横坐标 $\le0$，$e^{\Delta tL}$ 至多多项式增长 |
| 写入增益 | $G_{eff}=\hat\pi_c G_{track}+(1-\hat\pi_c)G_{init}$；$G_{init}=C^\top$（只写 presence/age/位置）；presence/age 通道单位增益 | 新对象首次出现时不能把"位置"误写成"速度"；等价于随新旧程度变化的 Kalman 初始化 |
| 写入门 | $\beta=\nu+(1-\nu)\sigma(w_s s+w_b)$，$\nu=1-\hat\pi_c$ | 新对象覆盖写；已追踪对象由学到的门控 |
| fold Jacobian | 反弹世界写入乘 $J_h=\pm1$ | 不乘时，折返相位上的速度会被往反方向纠正（单测 `test_fold_jacobian_needed_in_box`） |
| 同步写入 | 一步内所有可见 token 相对同一个 $S^-$ 计算残差后求和写入 | 与引理 4 的干扰律一致 |
| 坐标系 | 主矩阵用 allocentric token；`frame=ego` 子检查让动作通过 $L_{ego}(u)$ 搬运记忆 | 避免把"不会自运动补偿"混进"会不会演化"的比较 |
| 训练目标 | 只用可见、且此前见过的 token 的写入前误差（prequential）；隐藏真值永不入损失 | 隐藏期间的演化只能靠架构保存 |

## 4. 四条引理（全部有单测）

1. **叠加可交换**：与叠加可交换的逐对象演化律只能是 value 侧左乘 $\Phi(S)=KS$；predict 步不引入串扰（`test_lemma1_superposition`）。
2. **key 侧转移只是重新寻址**：$S\leftarrow SA$ 后读出落在已写 value 的张成内；只写过一次的对象无法外推。
   所以 GDN/DeltaProduct/Mamba-3 一类在自主连续运动上应暂停，而负特征值在置换任务上反而成功。
3. **坐标系等价**：$P_t=K_t\cdots K_1$、$\tilde S_t=P_t^{-1}S_t$ ⇒ predictive memory = 静态 delta 记忆 + 一个共享的 $d_v\times d_v$ 坐标系。
   不能声称比 transform-aware readout 更有表达力；正交 $P_t$ 造不出 shear ⇒ 异速对象上固定旋转必然失败（可证伪）。
   `ours_frame` 与 `ours` 在 fp64 下逐步相等（`test_lemma3_frame_form_equivalence`）。
4. **干扰律**：纠正对象 $i$ 使对象 $j$ 的读出改变 $\beta\,C G J e_i(k_i^\top k_j)$；predict 不增加干扰 ⇒ 干扰斜率可预先注册（评测中逐对象计算预测值）。

必须承认的限制：可逆良态动力学下与"时间戳静态记忆 + transform-aware readout"等价；表达不了对象间相互作用（Block World 本身无块-块碰撞）；
置换需要 key 侧；外观相同实例会混叠（ρ 扫描报告混叠墙）；常数增益忽略协方差随隐藏时长增长；对象数接近 $d_K$ 时串扰增大；
shear 不收缩，有限精度下表现为数值漂移（S1 做 fp64 回放检查）。

## 5. 新意边界

问题本身不是贡献（LiveWorld 2603.07145 几乎逐字提出"视野外对象被冻结"；WRBench、StEvo-Bench、2610.07355 已大量诊断）。V6 只主张：

1. value 侧、受动作/Δt 条件控制、含 shear 的线性生成元，在 delta-rule 记忆的无观测步上运行；
2. 等价分析（演化记忆 = 静态 delta 记忆 + 共享坐标系；异速需要 shear，旋转不够）；
3. 等预算联合持久性协议（BJP；为避开 2609.16183 的 "matched-state" 同名而改名），把遗忘/暂停/演化/纠正/非干扰/外推分开测；
4. 用演化记忆的残差修复 Cambrian-S 式重复计数（VSC-Repeat 至今无修复），并区分"预期中的演化"与"意外变化"。

揭示误差驱动的 episode 内生成元 TTT 只作次要消融（AdaJEPA、JEPA-TTT、Sandwich-Residuals、iSEE 各占一角）。
必须作为显式对照引用：LiveWorld、Koopman Dreamer（2607.19719）、MbD（2605.31163，key 侧 Kalman）、KLA、KDN、Surprise Forcing（2607.18436）、SPACE（2609.32592）、FloWM、Shell Game。
冻结预注册前，按 PREREGISTRATION 第 6 节再做一次 OpenReview/全文检索。

## 6. 数据（全部公开）

| 级别 | 来源与许可 | 生成/加载 | 用途 |
|---|---|---|---|
| L0 | FloWM MNIST World `dynamic po` 公开规格（Table 8：50×50 torus、32×32 视窗、5 对象、整数速度 −2..2、每步自运动 ±10）；官方生成器未公开 → `worlds_l0.py` 自建，未声明项写入 config | numpy 即时生成；训练池每 5000 步刷新 | 主世界与受控变体 |
| L1 | Miniworld fork（Apache-2.0）@ `b28988d`，FloWM 原命令（`biased_walk_v2 --forward-prob 0.90`、room 16、cardinal 速度、仅墙反弹） | `scripts/setup_v6_miniworld.sh` + `prepare-l1`（CPU 渲染，48 进程）；train 1 万 / dev 1000 / test 1000 × 500 步；另有 peekaboo / back_and_forth 重访评测 split | 复现：反弹、6–10 个块 |
| R | nuScenes 元数据（只下载表格；CC BY-NC-SA 4.0，下载时保存许可原文）；MOT17/MOT20 训练集 GT（纯标签包） | `prepare-real`；nuScenes 2 Hz、保留无点框、不插值；MOT 静止相机序列、可见度 <0.1 为隐藏、按序列留一 | 真实运动下"不暂停"的 token 级验证 |

L0 世界注册表（`data.L0_WORLDS`）：`l0_main`（FloWM 视窗）、`l0_sched`（静止相机、逐对象遮挡日程）、`l0_ctrl`（隐藏期间 50% 概率意外变速或瞬移）、
`l0_blackout`（关灯）、`l0_drift`（逐 episode 速度旋转/阻尼，H6）、`l0_tour_static`/`l0_tour_dynamic`（计数巡游，`copies=k` 拼接）。
覆盖项写法：`l0_ctrl:rho=0.5,n_objects=8`、`l0_main:blind_after=50`（FloWM 协议：50 步上下文 + 盲推）、`l0_main:frame=ego`。
评测长度 3T/9T 时 schedule 世界的隐藏时长按 T/64 等比放大。

R 阶段硬限制：隐藏段真值由标注员推断、近乎线性，线性插值 oracle 是按构造的天花板；只报告 gap closure，不主张胜过恒速 KF。

## 7. 变体与参考

| 名称 | 机制 | 级别 |
|---|---|---|
| `static_delta` | $L_{dyn}=0$ | 一级 |
| `gdn` / `gdn_neg` | key 侧标量衰减 + rank-1 擦除（FLA naive 递推）；$\beta\in[0,2]$ 负特征值版 | 一级 |
| `vo_rotation` | VO-RoPE 式固定频率 value 旋转（取最有利的 (p,q) 配对），不学习、无 shear | 一级 |
| `tpa_readout` | 静态记忆 + 通用时间感知 MLP 读出（输入 $\hat q,\hat\tau,\hat\pi$，平移不变） | 一级（H2 的关键对照） |
| `ours` / `ours_GCt` | 结构化 $L$ + 学到的 value 空间增益；$G=C^\top$ 纯 SGD 消融 | 一级 |
| `ours_frame` | 引理 3 坐标系形式（等价性检查） | 一级 |
| `fifo` / `gru` / `slot_known_id` | 有界事件检索；全局 GRU(1024)；已知 ID 的逐对象 GRU（结构参考，不对齐预算） | 一级 |
| `lact` | LaCT 快权重（官方 `a648340` 更新规则：SwiGLU、点积 inner loss、apply-then-update、更新后按初始行范数重归一化；为对齐预算关闭动量与滑窗注意力） | L1 一级、L0 一级 |
| `ours_ttt` / `amortized_dL` / `oracle_L` | 揭示缓冲上的 TTT（学习率 {0.1,0.3,1.0} 在 dev 上选）；GRU 产生 ΔL；真值生成元 | H6（次要） |
| `m2rnn` / `kdn_diag` / `ours_dense` / `flowm_token` | tanh 内 value 转移；对角 key 衰减 Kalman-delta；无约束生成元；FloWM 式速度通道栅格（地址侧搬运，结构参考） | 二级 |
| `pause` / `cv_kf` / `cv_kf_oru` / `analytic_kf` / `interp_oracle` | 复制最后观测；已知 ID 的恒速 KF；OC-SORT 式揭示后回放；已知墙面的 KF；用未来的线性插值上界 | 参考 |

预算：一级与二级中标注对齐的变体持久状态 ≤1024 标量（`state_scalars`，单测检查）；参数量、FLOPs 与显存另报。

## 8. BJP 协议指标（`metrics.py`）

- **ER**（演化比）$=\sum\langle\hat x-x_{last},x-x_{last}\rangle/\sum\|x-x_{last}\|^2$；只统计速度可辨识（最后一次可见段 ≥2 步）、未发生意外变化、真实位移非零的隐藏步；torus 上只用 $\|d\|_\infty\le$ 周期/4 的位移。0 = 暂停，1 = 正确演化。`er_all` 为不限可辨识的版本。
- **NHE** $=\mathrm{RMSE}(\hat x,x)/\mathrm{RMSE}(x_{last},x)$（copy-last 参照）。
- **遗忘**：history pair 分离误差（V4 定义）。**纠正**：揭示时（写入前）、+1/+3/+10 步误差，never-recovered 率（阈值 0.01×世界尺度），仅成功样本的恢复延迟；按"可预测 / 意外变化"分组。
- **非干扰**：change pair 中其他对象读出的位移，相对被纠正量；ours 族同时输出引理 4 的预测位移并回归斜率。
- **外推**：按"距上次看见的步数"分箱（1–8、9–32、33–64、65–192、193–576、>576）的 RMSE 与 NHE。
- **MemoryGain**：关灯世界中，"关灯时清空记忆"的误差减去完整记忆的误差。
- **surprise 侧**：k 份拷贝计数的 MRA（VSI-Bench 定义，10 个阈值硬编码、`<=`）；揭示事件上意外变化 vs 可预测演化的 AUROC；T 上选的单一阈值迁移到 3T/9T 后的 F1。
- **R 阶段**：按遮挡长度分箱的隐藏 ADE、揭示误差、+k 纠正、ER 与 gap closure。

统计：每个训练种子内以 episode 为单位配对 bootstrap（1000 次，95% percentile）；3 个种子分别报告，不合并；test split 仅在 FINAL 打开。

## 9. 阶段、预算与命令

| 阶段 | 内容 | GPUh 上限 | 门槛 |
|---|---|---|---|
| S0 | CPU：单测（含代数性质、torch 对拍）、生成器真值、KF oracle、MRA 四例、数据准备与遮挡直方图、L1 冒烟渲染 | 0 | 全部 gate 通过 |
| S1 | 全部变体 16 episode 过拟合；吞吐 profile；谱横坐标；fp32/fp64 9T 回放 | 2 | 全部可过拟合；偏差 >2× 先改预算并告知用户 |
| 冻结 | 用户确认阈值，提交 PREREGISTRATION_V6.md 并记录 commit hash | — | — |
| S2 | L0 主矩阵：12 个一级变体 × 3 种子 × {l0_main, l0_ctrl}，20k 步；参考基线 | 30 | H1/H2/H4 可判定 |
| S3 | ρ/N/噪声扫描、计数与 surprise 类比、ρ 训练变体 | 8 | H3/H5 |
| S4 | 漂移世界与 H6 | 18 | — |
| S5 | L1 Block World（含 lact、两个重访 split） | 35 | L1 复现 |
| S6 | 二级基线 | 20 | — |
| S7 | R：nuScenes（主）、MOT17/20 七折（次） | 6 | H7 |
| 预留 | 重跑与调试 | 25 | — |
| 合计 | 首个决策点 S0–S3 ≤40；全程 ≤144，硬上限 160 | ≤160 | — |

GPUh 记账 = 阶段墙钟 × 占用 GPU 数（独占计费，偏保守）；记录在 `<out>/budget_ledger.json`。换成 4×H800 时上限不变，墙钟约翻倍。

```bash
python -m unittest discover -s tests/predictive_memory_v6 -t . -v        # 软件检查
bash scripts/run_v6.sh S0 /abs/runs/v6_<id>
bash scripts/run_v6.sh S1 /abs/runs/v6_<id>
PMR_V6_PREREG_COMMIT=<hash> bash scripts/run_v6.sh S2 /abs/runs/v6_<id>
python -m pmr.predictive_memory_v6 report --out /abs/runs/v6_<id>
```

输出：每个作业目录有 `job_spec.json`、`status.json`、日志；训练有 `manifest.json`（参数量、状态标量、GPU、配置）、`training.jsonl`、`best.pt`/`final.pt`；
评测每个条件有汇总 JSON、逐 episode `*_episodes.jsonl.gz`、揭示事件 `*_events.npz`。根目录 `metrics.csv`、`hypotheses.json`、`REPORT.md`、`figures/`（学到的 $L$、$G$ 热图与谱）。

## 10. 主假设失败时的退路

- H1 中 ours 失败而 gdn_neg/lact 成功：转向 key 侧或非线性 TTT，在其上研究记忆感知 surprise。
- H2 中 `tpa_readout` 在 9T 也追平：结构化生成元对外推不必要，主张改为参数效率、可解释谱与 surprise。
- 混叠墙过早：转向按位置编码的键 + 地址侧搬运（FloWM/SSP 式），或双侧转移 $S\leftarrow KSA$。
- L1 反弹处理失败：换学到的非线性读出；仍失败则把反弹列为限制，以 slot 结构作对照。
- H6 中 amortized 不弱于 TTT：如实报告"测试时训练不必要"。
- 撞上稀疏监督"干扰墙"（训练长度内就学不出隐藏演化）：对全部学习型变体统一启用 success-gated 隐藏时长课程（`curriculum=True`），绝不调低门槛。
- R 事件不足或真实 ER 崩溃：主指标转向 R-b 伪遮挡，或把结论限定在合成世界。
若 H2 与 H5 都无差异，论文收窄为 BJP 协议 + "记忆感知 novelty 修复重复计数"的机制演示。

## 11. 与 Saining 研究的接口

Cambrian-S 的 predictive sensing 中，surprise = 1 − 冻结的两层 MLP 预测下一帧 SigLIP2 特征的余弦相似度；长期记忆是静态 KV 快照；VSC 计数按 surprise 切段后 `sum()`。
surprise 从不查询记忆。2511.16655 的 VSC-Repeat：视频与自身拼接 2 次（共 3 份、30 分钟），MRA 从 42.0% 掉到 3.6%，拼接 5 次归零；Cambrian-S 作者承认这一局限。
V6 的 H5 是它在 token 世界的机制级缩影：memoryless（LFP 类比）、静态记忆残差 / 最近邻 novelty（Surprise Forcing 类比）、演化记忆残差三者对比。
其余连接：Thinking in Space（局部 world model vs 统一全局模型）、Cambrian-P（位姿 token ↔ 动作驱动的 $K(u)$）、Solaris（转身期间另一名玩家仍在移动；论文写明缺少持久记忆）、VSTAT、AMI Labs 的 JEPA 路线。

Office hour pitch：*Predictive sensing needs a predictive memory: surprise should be measured against what memory says the world is now.*
（完整中文 pitch 与三个提问见调研报告第 4 节。）

**RSI 不进 V6**：生成器在任意时长给真值，自标注无信息增益；可信的 RSI 主张至少需要算力匹配、数据匹配、verifier 消融、单轮对多轮等八组对照，代价超过 V6 全部预算。
RSI 唯一合理的位置是 V7 的真实长视频（长时程标注稀缺时，用循环闭合/重访一致性做 verifier 的时长自举）。
