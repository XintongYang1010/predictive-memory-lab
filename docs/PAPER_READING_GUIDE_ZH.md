# 从已有 bad case 出发的论文阅读顺序

2026-10-24 更新；三篇指定版本的作者、章节及公式已重新核对。这次仅整理已有案例和阅读路线，没有新增 PM 实验。

先看[四张 PM 案例卡](PM_CASE_GALLERY_ZH.md)，再配着新版
[CV 案例图册源码及媒体](https://github.com/JasonLIN0226/CV_project/tree/phase2/independent-interfaces-20261020/results/literature_diagnosis/visual-casebook-20261024)
阅读。下载 CV 仓库后打开该目录的 `index.html`；GitHub 文件页不直接执行 HTML。
[CV 完整配套阅读指南](https://github.com/JasonLIN0226/CV_project/blob/phase2/independent-interfaces-20261020/docs/CASE_STUDY_READING_GUIDE_ZH.md)
列出逐例数字和统一分析模板。

本轮按 **Worth Remembering → PaFu-KV → Gated DeltaNet** 阅读：先理解 WR 启动时的事件接纳；
再把 CV 内容失败与成功放在一组，时间戳只作辅助消融；最后回到 PM 的旧值改写与保护。
接纳失败和保存后利用失败发生在不同环节，不能把它们都归为“淘汰错了”。

## 1. Worth Remembering：把“接纳事件”与“有限容量保留”分开

[Worth Remembering: Surprise-Gated Robot Episodic Memory，v3](https://arxiv.org/html/2606.03787v3)
—— Nicolas Gorlo、Derek K. Wise、Alberto Speranzon、Luca Carlone。
必读 **§3.2 的式1–3、§5**；略读 §3.3，附录A选读以核查近似关系。

实际触发分数是局部 Gaussian 标准化的 L1 偏差，结合 median/MAD 阈值、局部峰值和
NMS，存储事件片段供后续检索。不要把式1直接当成普通预测 MSE 或精确计算的KL。
论文的事件形成与检索也不能直接等同于我们的固定K在线淘汰规则。
[原文 §3.2](https://arxiv.org/html/2606.03787v3#S3.SS2)。

对应 **CV 的早期 cue 启动问题**：四帧共同前缀可救回62个 cue，其中21个预测误差
反而上升。阅读时分开问：一个事件是否被触发、是否一直保留、reader 是否能利用它？
在 **PM 卡片2**，总分下降可以伴随 signal 变差，进一步提醒我们“异常度”与“任务
价值”不同。现在还没有验证“低惊讶关键线索被高惊讶动态干扰挤掉”，不要把启动
问题写成原始假说已经证实。

**对照图册逐项回答：**

1. WR trace 中 cue 到来时是否已有 Gaussian/MAD 状态？第一次失败是未接纳，还是接纳后被淘汰？
2. 共同四帧前缀为何能隔离启动问题？896次固定子集时间平移检查排除了什么，又没有排除什么？
3. 62个 cue 救回后41个改善、21个变差；每条轨迹额外执行4步；数据集最大逻辑状态增加32字节、本展示例峰值增加40字节后，能否声称同总成本的策略胜利？

**读完能回答：** 事件形成与下游任务价值是两道检查；这里证实的是适配实现的短序列启动问题。
原论文并未给出我们的固定 K 淘汰协议，这些例子也未验证高惊讶动态干扰挤掉低惊讶重要信息。

## 2. PaFu-KV：最接近当前 proposal 的监督对照

[Past- and Future-Informed KV Cache Policy with Salience Estimation in Autoregressive Video Diffusion，v3](https://arxiv.org/html/2601.21896v3)
—— Hanmo Chen、Chenghao Xu、Xu Yang、Xuan Chen、Cheng Deng。
必读 **图2、§3.2 式4–5、§3.3、§3.4 式6**；附录C选读。

论文从双向 teacher 的 attention 构造过去／当前／未来分区的显著性，再以
Q/K/V 表征训练轻量 head，推理时用预测的显著性管理缓存。它的标签不是删除后的
任务损失。先弄清 teacher 标签怎样聚合、student 推理究竟能看什么，再看效率结果。
[原文方法部分](https://arxiv.org/html/2601.21896v3#S3)。

对应 **PM 卡片1和成功卡片3**：同一个 attention 信号有时读到有用值，有时指向
有害旧值。带着这个具体问题阅读：如果两个条目表达同一信息，attention 高是否仍
等于边际贡献大？如果 teacher 本身偏好有害条目，提高 student imitation 有何上限？
这是假说与诊断问题，不是声称原论文已被这些退化 bank 案例推翻。

在我们的适配中，必须保留未来 attention 标签与过去可用学生的区分；不能把删除
损失换进 PaFu 的标签，再声称在比较两种监督。原论文改变视频扩散模型训练和 KV，
我们的共享冻结 reader／帧级历史项实验只能称机制适配。

**对照内容失败＋成功组以及 PM 卡片1、3，逐项回答：**

1. 在同时间、同动作下替换内容，失败和成功为何方向相反？attention 命中证明了哪一步，没有证明哪一步？
2. 原文式4–5的标签怎样由 attention 得到？式6的学生能否看到未来？为什么不能把删除损失替代标签后仍称 PaFu 对照？
3. 时间戳辅助消融与内容存在交互时，或者 PM 删除后只是回退零残差时，oracle gap 中有多少是保留收益、多少是 reader／目标问题？

**读完能回答：** 将来会被读取与相对剩余记忆的任务增量贡献不是同一标签。
CV 的输入替换诊断直接针对我们共享冻结 reader，不能据此断言原 PaFu 视频扩散系统失败。
CV 独立新场景尚未确认 cost scorer 优于 shuffled-cost，且仍落后强 attention/diversity；
案例解释不等于新监督已获胜。

## 3. Gated DeltaNet：给 PM 的改写／保护问题一个正确参照

[Gated Delta Networks: Improving Mamba2 with Delta Rule，v1](https://arxiv.org/html/2412.06464v1)
—— Songlin Yang、Jan Kautz、Ali Hatamizadeh；另见
[ICLR 2025 正式页面](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4904fad153f6434a7bcf04465d4be2cc-Abstract-Conference.html)。
结合 PM 时必读 **§2.3、§3.1、式8和表1**；只做 CV 时选读。§3.2 的并行算法暂时选读。

这里的门控衰减控制旧状态的保留，delta rule 根据当前 key 的旧读出误差进行定向
修正。两者结合是既有方法，不应把 A2 的共享 α/β 更新重新当成自己的创新。
[原文 §3.1](https://arxiv.org/html/2412.06464v1#S3.SS1)。

对应 **PM 卡片4**：我们的旧结论是降噪收益与局部 signal 代价的交换，不是已经证明
共享 β 的表达能力不够；账面0.0125%的修复空间也不适合作主线。读完后应能明确：
什么证据允许覆盖一个 key？怎样改写变化对象，同时保护未变化对象？这对应状态更新
proposal 的配对实验，并不自动给 CV 的帧保留策略一个公平的新 baseline。

**对照 PM 卡片1–4逐项回答：**

1. 式8中 α 的整体衰减与 β 的 key 定向修正各会影响哪些已有信息？原有 A2 与它有何共同机制？
2. 旧值有害时，应该删除、写入新值还是保护其他 key？用什么过去证据把真实变化与噪声区分？
3. 成功对照和 signal/noise 分解是否支持大而稳定的任务收益？旧 A2 的0.0125%账面修复量为何不应重新变成主线？

**读完能回答：** “什么时候改写、什么时候保护”要用任务证据检验，不能由一次总分降低推断。
PM 继续使用 oracle L0，不借阅读打开新的感知诊断；也不把矩阵状态和历史 bank 的分数混为同一淘汰比较。

## 读完后与 Saining 讨论的三个问题

1. 我们要学的是“会被注意／会被读取”，还是“在当前剩余记忆下对任务的增量贡献”？
   用卡片1和3同时回答，避免只展示失败。
2. Oracle gap 有多少来自已丢证据、读取器偏差、未来噪声或未来挑时间？这些部分中，
   哪些能由策略允许的过去输入预测？用卡片2和 CV 同状态冲突回答。
3. 若需要改写，收益是大到足够研究的稳定现象吗？能否在同输入、同总内存预算、同
   reader／任务目标下赢过简单控制？先保留卡片4的负结论，再谈方法。

三篇论文提供的是候选机制；当前开发结果提供的是诊断事实。它们还没有证明我们的
删除价值监督有优势，也不能证明它不可学。下一次方法选择应由同一协议下的成功／
失败比例及可学习收益决定，而不是由最醒目的单例决定。
