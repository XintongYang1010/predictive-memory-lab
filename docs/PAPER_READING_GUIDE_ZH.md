# 从已有 bad case 出发的论文阅读顺序

2026-10-22，已核对下面原论文的指定版本。先看[四张 PM 案例卡](PM_CASE_GALLERY_ZH.md)，
再读这三篇。目的不是增加 baseline 数量，而是判断 surprise、attention 和状态更新
分别解决哪一段问题。建议先读 PaFu-KV，再读 Worth Remembering，最后读 Gated DeltaNet。

## 1. PaFu-KV：最接近当前 proposal 的监督对照

[Past- and Future-Informed KV Cache Policy with Salience Estimation in Autoregressive Video Diffusion，v3](https://arxiv.org/html/2601.21896v3)
—— Hanmo Chen 等。优先读 **§3.2、§3.3、§3.4，图2，附录C**。

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

## 2. Worth Remembering：把“接纳事件”与“有限容量保留”分开

[Worth Remembering: Surprise-Gated Robot Episodic Memory，v3](https://arxiv.org/html/2606.03787v3)
—— Nicolas Gorlo、Derek K. Wise、Alberto Speranzon、Luca Carlone。
优先读 **§3.2 的式1–3、§3.3、§5；附录A只核查近似关系**。

实际触发分数是局部 Gaussian 标准化的 L1 偏差，结合 median/MAD 阈值、局部峰值和
NMS，存储事件片段供后续检索。不要把式1直接当成普通预测 MSE 或精确计算的KL。
论文的事件形成与检索也不能直接等同于我们的固定K在线淘汰规则。
[原文 §3.2](https://arxiv.org/html/2606.03787v3#S3.SS2)。

对应 **CV 的早期 cue 启动问题**：四帧共同前缀可救回62个 cue，其中21个预测误差
反而上升。阅读时分开问：一个事件是否被触发、是否一直保留、reader 是否能利用它？
在 **PM 卡片2**，总分下降可以伴随 signal 变差，进一步提醒我们“异常度”与“任务
价值”不同。现在还没有验证“低惊讶关键线索被高惊讶动态干扰挤掉”，不要把启动
问题写成原始假说已经证实。

## 3. Gated DeltaNet：给 PM 的改写／保护问题一个正确参照

[Gated Delta Networks: Improving Mamba2 with Delta Rule，v1](https://arxiv.org/html/2412.06464v1)
—— Songlin Yang、Jan Kautz、Ali Hatamizadeh；另见
[ICLR 2025 正式页面](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4904fad153f6434a7bcf04465d4be2cc-Abstract-Conference.html)。
优先读 **§2.3、§3.1、式8和表1**；暂时不必精读 §3.2 的并行算法。

这里的门控衰减控制旧状态的保留，delta rule 根据当前 key 的旧读出误差进行定向
修正。两者结合是既有方法，不应把 A2 的共享 α/β 更新重新当成自己的创新。
[原文 §3.1](https://arxiv.org/html/2412.06464v1#S3.SS1)。

对应 **PM 卡片4**：我们的旧结论是降噪收益与局部 signal 代价的交换，不是已经证明
共享 β 的表达能力不够；账面0.0125%的修复空间也不适合作主线。读完后应能明确：
什么证据允许覆盖一个 key？怎样改写变化对象，同时保护未变化对象？这对应状态更新
proposal 的配对实验，并不自动给 CV 的帧保留策略一个公平的新 baseline。

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
