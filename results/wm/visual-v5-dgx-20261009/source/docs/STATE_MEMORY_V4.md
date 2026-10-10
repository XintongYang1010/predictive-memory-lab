# V4：遮挡、状态更新与历史记忆的机制实验

V4 要回答的是：**一个对象离开视野后，模型能否依靠过去观察持续估计其状态；新证据到达时，能否修正该对象，同时保留其他对象仍有效的历史？**

这是对原先 world model + history memory 目标的一次小规模、可解释检查。它使用合成数值状态，没有图像编码器、动作条件视频生成或 JEPA 训练，也没有实现 RSI。结果只支持这个任务内的机制判断，不代表视觉 world model 的效果或新方法的成立。

## 为什么从 V3 转到这里

V3 编辑的是冻结 RealMotion 在两次更新时间传递的旧预测 bank；它没有检验长历史下的持续世界状态。
已审计的 temporal 候选在 Dev 的 ES 配对收益约为 +0.01033，但 Eval 为 −0.00875，95% CI 为 [−0.01778, +0.00022]，没有证明部署收益。
这不能推出“memory 没用”，也不能证明继续给该 controller 加特征就能回答原问题。

V4 将记忆的保留、隐藏期间的演化、新证据修正和对象间干扰分开测量。它首先建立研究诊断，五个模型都作为对照，不把其中某个结构包装成已验证的新方法。

## 环境与模型能看见的内容

每段默认含 3 个对象，时间、位置和速度采用任意单位，步长为 1。环境分段恒速：

1. 第 0、1 步全部对象可见；之后，一个待变化对象和一个保留对象被遮挡，背景对象持续可见。角色在每个 episode 随机分配。
2. 待变化对象在序列中点（长度 24 时约第 12 步，抖动 ±1）发生位置跳变和速度变化，同时重新可见 3 步，随后再次遮挡。变化没有预告。
3. 保留对象从第 2 步起始终不可见，继续原来的运动。模型必须保存并推进其早期信息。

模型唯一输入为 `[x, y, vx, vy, visible]`。速度是当前传感器读数，不由未来位置作差获得；可见坐标有标准差 0.01 的独立高斯噪声，不可见坐标严格清零。对象对应关系已知，模型拿不到真实状态、对象角色、未来或评价掩码。

模型输出当前的四维状态估计。未来位置按估计的 `xy + h * velocity` 推算，默认 `h = 0, 1, 4`；其中 `h=0` 是重建，不能将其称为未来预测。跨越尚未观测到的随机跳变的目标被排除，避免把无法预知的变化当作模型失败。掩码仅用于训练损失和评价，不作为模型输入。

这比视觉任务简单很多：没有目标检测、身份匹配、相机变化和复杂物理；还直接提供速度，恒速动力学可以解析求解。长序列同时增加遮挡时间、积分误差与位置范围，不能将长度外推下降全部归因为“记忆遗忘”。

## 实验矩阵

| 对照 | 实际机制 | 默认持续状态预算 | 应如何解释 |
|---|---|---:|---|
| `window` | 最近 4 帧，学习 attention reader | 60 个标量 | 短历史对照；窗口之外的信息确实丢失 |
| `retrieval` | 全局 FIFO 保留最近 8 个可见对象事件，使用与 window 相同的 reader | 56 个标量 | 8 个事件不是 8 帧；没有学习写入或淘汰，不是 FAR 复现 |
| `gru` | 全局 GRU 递归状态 | 192 个标量 | 通用可学习状态对照 |
| `slot` | 每对象 64 维状态，共享 GRU 写入与潜在残差动力学；遮挡时仍推进 | 192 个标量 | 已知身份、对象之间不互相写入，是强结构先验 |
| `analytic` | 保留每对象最新可见位置/速度，恒速推进；重新可见时覆盖 | 12 个标量，0 个参数 | 任务可解性参考；带噪下不是性能上界或 hindsight oracle |

状态预算不包含模型参数、瞬时激活、返回的整段输出或训练反向图，也不是显存字节数。
GRU 与 slot 只匹配持续状态标量数，参数量不同；window 与 retrieval 共享 reader 结构，但不与其他模型匹配参数或存储预算。报告中的真实参数量、持续状态量、时间和显存必须一起看。
FIFO 同帧事件按对象索引插入，容量截断时较高索引保留；角色随机化减轻固定角色偏差，但不能据此声称检索策略已被优化。

默认训练长度为 24，评价长度为 24、48、96，种子为 42、43、44。每个种子 Train/Dev/Eval 为 4096/512/512 个 episode，学习模型训练 1500 步，batch size 64。三种种子顺序运行，不启动多 GPU 作业。

数据按 split 和 episode ID 分隔；配对分析必须使用一致的 episode ID。三个训练种子使用同一批数据，用于检查初始化/优化差异；不同评价长度使用 `eval_T24`、`eval_T48`、`eval_T96` 独立 split，不是同一 episode 的截断。具体划分与潜变量关系保存在 `datasets_manifest.json`。
默认数据种子为 20261010，与本地 120 步试跑的 20261009 分开；smoke 使用额外的 `smoke_` split 前缀。本轮仍是探索性机制实验。

## 四类干预如何判断是否真的使用历史

| 检查 | 改变什么 | 保持什么 | 能回答什么 |
|---|---|---|---|
| 正常输入 | 无 | 完整合法观察 | 基准状态估计能力 |
| 初期历史 delete / shuffle / stale | 仅第 0、1 步删除、跨 episode 错配，或用第 0 步重复第 1 步 | 后续观察与原世界标签 | 初期信息缺失、错误或陈旧时的敏感性 |
| 无变化配对世界 | 取消待变化对象的跳变 | 其他对象及共享观测噪声 | 改一个对象是否意外影响保留对象 |
| 不同早期历史配对世界 | 只改变保留对象的初始位置/速度 | 第 2 步起全部可见输入相同 | 当前观察完全相同时，预测能否反映不同的真实历史 |

`stale` 只改变初期的一次测量，不是“拒绝接受新证据”的实现。无变化和不同历史配对都重新使用各自世界的真值，不给错配世界沿用原标签。
删除历史后的损失增大只能说明依赖该信息；还需确认正确历史下预测准确，且不同历史带来的预测差与真实差相符。
历史损坏条件沿用正常世界的自然遮挡区间计算 `hidden_position_rmse`，不把被删除的前两帧额外计入，保证条件间评分区间相同。

slot 的对象独立结构使跨对象改写天然被阻止。因此它在配对干扰指标上为零是结构性质，不是“学会局部修正”的独立证据；仍需检查 changed 对象恢复与 preserved 对象长期准确性。

## 读结果的判据

本轮的主对比是 **长度 48、正常历史下，slot 对 GRU 的 `hidden_position_rmse`**，因为两者持续状态大小相同；这是机制对照，不是新方法胜负榜。
同时检查下列约束，不能只挑一项有利指标：

| 指标 | 含义及注意事项 |
|---|---|
| `hidden_position_rmse` | 对不可见对象位置的欧式 RMSE，检查遮挡期间状态追踪 |
| `changed_post_position_rmse` | 新证据到达后的变化对象位置误差，检查修正效果 |
| `preserved_post_position_rmse` | 同期未变化且持续隐藏对象的误差，检查保留代价 |
| `never_recovered_rate` + `recovery_latency_success_only_mean` | 恢复要求连续 2 帧位置误差 ≤0.15；失败率必须与仅成功样本的恢复时间同时读 |
| `unsolicited_update_rmse` | 实际世界与无变化世界中，保留对象预测的差异；衡量跨对象干扰，不替代准确率 |
| `history_pair_difference_rmse` | 末步两世界的预测位置差与真实差之间的欧式误差；不是两预测之间有差就算成功 |
| `future_mse` | 按有效坐标汇总的预测 MSE，默认含 h=0/1/4；不同于欧式位置 RMSE |

每个指标先得到 episode 级数值，再等权汇总。配对 bootstrap 以 episode 为重采样单位，默认 1000 次、95% percentile CI；正收益定义为 reference error − candidate error。
同一训练种子内做配对区间，不把三个训练种子重复的 episode 当作独立样本拼接。分别报告三个种子、缺失指标的实际分母和失败次数；三个种子不足以声称广泛稳定性。

支持继续的证据是：主比较的收益在种子间方向一致，区间与效应大小具有解释力；新证据后的恢复没有用持续破坏其他对象换来；不同历史的预测差符合真实状态差。长度 96、历史损坏和短窗口对比用于解释机制及失败，不用于事后重选主指标。

若 analytic 的无噪声测试不能通过，先查环境/时序/评价。若学习模型连长度 24 都拟合不好，先查训练，不急于讨论长度外推。若 slot 只凭结构先验避免干扰，却无法推进隐藏状态或接受新证据，则不能说核心问题已解决。
若所有收益都可由已知身份、更多参数或恒速先验解释，这只是有效诊断，不是论文新颖性；后续需再做参数/计算匹配、未知身份和视觉任务验证。

## 在已经跑通的 A100 镜像启动

进入新下载的仓库，在已分配到 GPU 的作业内执行：

```bash
bash scripts/retest_v4.sh /share/home/s2026316084/yxt_pmr_20261002
```

该路径只用于寻找先前工作的 Python，优先使用 `source_852f57d/.venv/bin/python`，再检查旧根目录的 `.venv/bin/python`；也可显式指定：

```bash
PMR_PYTHON=/path/to/working/python bash scripts/retest_v4.sh
```

两者都没有指定时使用当前 `python`。脚本检查 CUDA，执行专门 V4 测试、小规模 GPU smoke、正式训练与评价，最后打包。
它不需要 AV2、RealMotion、预训练 checkpoint 或额外数据下载，也不会自动安装或覆盖 PyTorch。
若工作环境缺少 V4 的 Python 依赖，可用该环境的解释器安装：

```bash
/path/to/working/python -m pip install -r requirements-state-memory.txt
```

目标镜像仍为 Python 3.11 / PyTorch 2.9.1 / CUDA 12.8。依赖文件不包含 torch，保留现有 CUDA wheel。脚本在首次 CUDA 调用前设置确定性矩阵运算所需的 `CUBLAS_WORKSPACE_CONFIG`，已有设置则保留。

初始 wall-time 预算为 3600 秒，从 wrapper 启动计时；正式 runner 获得扣除检查与 smoke 后的剩余时间。GPU smoke 最多分到 120 秒，并标记 `smoke_only=true`，其效果数值不能当作实验结果。
runner 按阶段检查预算并保存 partial；单个正在执行的操作可能稍超时，这不是调度器硬截止。完整耗时需依据实际设备 profile，不能承诺一小时一定跑完。

结果路径自动包含 UTC 时间、Slurm 作业号和进程号。已有输出会被拒绝覆盖；可用 `PMR_V4_OUT` 指定全新路径，或用 `PMR_V4_MAX_SECONDS` 显式调整本次预算：

```bash
PMR_V4_OUT="$PWD/runs/state_memory_v4_fresh" \
PMR_V4_MAX_SECONDS=3600 \
bash scripts/retest_v4.sh /share/home/s2026316084/yxt_pmr_20261002
```

runner 超出预算会记录 `budget_exceeded` 并非零退出；内部其他异常记录 `failed`。若在启动前的环境检查就失败，查看 wrapper 日志。保留已有结果和控制台日志，不自动转 CPU、不把不完整结果当成功，不自动扩大预算。
**V4 目前没有 resume。** 重跑必须使用新目录；先阅读 partial 的训练速度和已完成组，再决定是否调整下一次预算。V1–V3 的 `--resume` 不适用于本模块。

本地 CPU 只检查软件的最小命令：

```bash
python -m pytest -q tests/test_state_memory_*.py
python -m pmr.state_memory --device cpu --smoke \
  --steps 3 --train-episodes 32 --dev-episodes 8 --eval-episodes 8 \
  --seeds 42 --eval-lengths 24 --max-seconds 120 \
  --out runs/state_memory_v4_cpu_smoke_fresh
```

## 返回哪些文件

正式套件全部完成、且 `run_stats.json` 为 `status=complete, smoke_only=false` 时，脚本输出：

```text
RETEST_V4_COMPLETE: <输出目录>.tar.gz
SHA256: <输出目录>.tar.gz.sha256
```

返回这两个文件即可。打包器将正式结果与 `<输出目录>.log` 一起打包，并加入逐文件 SHA256 清单。完整包中应包含：

| 文件 | 用途 |
|---|---|
| `REPORT.md`、`summary.json`、`metrics.csv` | 先看状态、主指标、各长度和种子的对比 |
| `episode_metrics.jsonl` | 复算逐 episode 差异与配对区间 |
| `training.jsonl`、`training_curves.png`、`checkpoints/` | 检查训练、Dev 选择与保存的模型 |
| `datasets_manifest.json`、`source_manifest.json`、`environment.json` | 追溯数据配置、split、源码和环境 |
| `run_stats.json` | 完成状态、运行预算与实际开销 |
| `bad_cases/`、`trajectories/` | 对照预测与真值，定位恢复失败、漂移和干扰 |

失败或超时时没有 complete 结果包；检查日志及保留的目录，不要只回传一张均值截图。

## 这一轮对应的论文与 Saining 问题

| 阅读顺序 | 原文和章节 | 已有研究与你要回答的问题 |
|---|---|---|
| 1 | [Can Video World Models Track Unobserved World States? (2026)](https://arxiv.org/abs/2608.30692)，[官方项目](https://joonghyuk.com/stateful-vwm-web/) 的 *Mechanisms That Enable State Tracking* 与 *Beyond the Canonical Shell Game* | 已研究可修改的持续状态、非线性 fast weights、观察纠正与视野外动力学。V4 的 learned state update 不能宽泛声称新颖；需要说明测试了它之外的哪种失败。 |
| 2 | [Learning What to Recall: Adaptive Multi-Cue Episodic Memory for World Models / FAR (2026)](https://arxiv.org/html/2609.34677v1)，§3.1、§4.3、§5、Appendix B.3/C.4 | 已用训练时未来预测效用监督推理时不看未来的检索，也研究变化世界中的陈旧记忆。V4 的 FIFO 不代表该方法；若研究重点转向检索，必须加入更强的选择对照。 |
| 3 | [Cambrian-S: Towards Spatial Supersensing in Video (2025)](https://arxiv.org/html/2511.04670v1)，§4.1 *Predictive Sensing via Latent Frame Prediction*、§4.2、Appendix F.2 *Memory Framework Design for VSI-Super Recall* | Saining 合著。surprise 来自下一帧 latent 预测误差，用于 LLM KV 记忆压缩/整合；它能指导保存新信息，但并不直接给出旧状态应如何局部修正。 |
| 补齐 WM 基础 | [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning (2025)](https://arxiv.org/html/2506.09985v1)，§3–4；[Solaris: Building a Multiplayer Video World Model in Minecraft (2026)](https://arxiv.org/html/2602.22208v1)，§5.4、§6 | V-JEPA 2 用于理解 latent、动作条件预测与规划的关系；Solaris 是 Saining 合著，阅读生成历史训练与记忆评价。V4 当前都没有复现，不能从此轮成功直接推出换用 JEPA 即有效。 |

FAR 和 stateful video world model 并非 Saining 的论文。Solaris 的 checkpointed self forcing 解决训练与生成历史及反向显存问题，不能把显存节省直接等同为语义长期记忆。

带去 office hour 的问题应落到结果上：

1. 在已知身份的数值状态任务里，历史保留、动态推进、局部纠正究竟哪一个仍失败？该失败有没有视觉任务对应，而不是由我们的弱对照制造？
2. Cambrian-S 的 surprise 存储与持续隐状态更新之间，是否需要区分“保留某次证据”和“修正当前状态”？什么观察能判别，而不是只比较最终平均分？
3. 如果 analytic 已足够强，下一步最小视觉验证该引入什么：未知身份、局部遮挡、非恒速隐藏变化，还是观察噪声？一次只引入一种，保留同历史配对与对象级指标。

每次新结果后更新阅读顺序：检索丢失先读 FAR；跨对象干扰或长度外推失败先读 stateful world model；保存何种视觉证据先读 Cambrian-S；进入动作条件或视频生成后再深入 V-JEPA 2 / Solaris。先用本轮证据缩小问题，再决定新的方法与算力预算。
