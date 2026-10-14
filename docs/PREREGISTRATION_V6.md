# V6 预注册（状态：PROPOSED，待用户确认后冻结）

机器可读版本：`pmr/predictive_memory_v6/prereg.py`（两者必须一致）。

## 1. 冻结流程

1. 用户确认（或修改）第 3 节阈值。
2. 执行者把本文件首行改为 `状态：FROZEN（<日期>，用户确认）`，把 `prereg.py` 的 `STATUS` 改为 `"frozen"`，单独提交。
3. 记录该提交的 hash；S2 及之后的每次运行都带 `--prereg-commit <hash>`（运行器拒绝没有它的 S2+）。
4. 冻结后：不降低任何阈值，不为了通过而换种子；门槛失败先按实现错误排查，确认不是 bug 后作为发现报告。
   任何协议变更都要用户批准，并另起 `PREREGISTRATION_V6.1.md`，旧结果保留。

## 2. 判定规则

- 每个假设的通过条件对 **每个训练种子** 分别成立才算通过；区间为种子内按 episode 配对的 bootstrap（1000 次，95% percentile）。
- 判定数据：dev split（固定种子命名空间，与训练池、dev-select 分离）。test split 只在 FINAL 打开一次，用于确认，不用于选择。
- 缺数据记为 missing，不算通过也不算失败；`report.py` 自动给出 pass / fail / missing。

## 3. 假设与阈值

| 编号 | 主要条件（S2 的 `l0_ctrl` 训练模型在 `sched_T*` 上评测，除非注明） | 通过条件 | 放弃/收窄 |
|---|---|---|---|
| H1 协议有效性 | ours 演化，只在写入时改状态的记忆暂停 | ours：ER∈[0.85,1.15] 且 NHE≤0.30（T=64）；static_delta、gdn、gdn_neg、fifo、lact：ER≤0.20 | gdn_neg 或 lact 在 9T 上 ER≥0.8：撤回"key 侧不能推进"，论文收窄为协议 + surprise |
| H2 结构必要性/等价 | 异速时固定旋转失败；坐标系形式一致；通用读出训练长度内追平、外推失败 | vo_rotation ER≤0.30；ours_frame−ours 的 ER 差 CI 在 T=64、192 上含 0；tpa_readout 的 NHE ≤1.2×ours（T=64），≥2×ours（T=576） | vo 成功：生成元叙事作废并作为发现报告；tpa 在 9T ≤1.2×ours：结构化生成元对外推不必要 |
| H3 纠正与非干扰 | `ctrl_T64` 意外变化组；`change_T64`；S3 ρ 扫描 | ours +3 步误差 ≤2×analytic_kf；never-recovered ≤5%；引理 4 斜率 ∈[0.8,1.2]；ρ≤0.5 时干扰 ≤ 被纠正误差的 10% | 干扰高于 gru：收窄到低 ρ；速度纠正失败：加 age 门控增益，仍失败则只声称位置纠正 |
| H4 外推与数值 | `sched_T64` vs `sched_T576` | ours NHE(9T)/NHE(T)≤1.5；gru、slot ≥3；S1 fp32/fp64 回放最大偏差 ≤1e-3 世界单位 | ours≥3：查谱；结构化 L 仍漂移则撤回外推主张；精度超限：按 2609.32447 式问题报告、改用补偿求和 |
| H5 记忆感知 surprise | S3 计数（k 份拷贝）与 S2 `ctrl_T*` 揭示事件 | lfp_analog 的 MRA 在 k=3 比 k=1 相对下降 ≥50%；记忆型（ours、static_delta）下降 ≤10 个百分点；AUROC：ours≥0.90，static_delta 与 nn_novelty ≤0.75，lfp ≤0.65；T 上选的阈值迁移到 3T/9T 后 F1 下降 ≤0.05 | lfp 不复现失败：先修类比；ours 与 max(static, nn_novelty) 的 AUROC 差 <0.05：收窄为"记忆感知 novelty"，撤回 predictive 主张 |
| H6（次要） | S4 漂移世界；TTT 学习率在 `drift_T64` 上选 | 隐藏误差比 ours 降低 ≥30%（`drift_T192`）；同测试时 FLOPs 下不劣于 amortized_dL | amortized ≥ TTT：报告 TTT 不必要 |
| L1 复现 | S5 `l1_dev`（500 步） | H1、H3、H5 方向在 3 个种子上一致，配对 CI 不含 0 | 只在 L0 成立：结论限定为 torus 恒速世界 |
| H7 真实 token | S7 nuScenes 自然遮挡（严格隐藏定义） | ours ER∈[0.8,1.2]；隐藏 ADE ≤1.1×cv_kf；gap closure ≥0.6；static_delta 与 gdn_neg ER≤0.2；MOT 方向一致 | 合成 ER≥0.85 而真实 ER<0.5：主张限定为合成世界；自然事件 <200：主指标改为 R-b 伪遮挡，自然遮挡只作描述 |

软件门槛（S0/S1）：KF oracle 隐藏 RMSE <1e-3；MRA 四例逐位一致；GDN 与 FLA naive 对拍 <1e-5；TTT 等价 <1e-6；全部变体 300 步内过拟合到首损失的 50% 以下；
L1 可见性与深度一致率 ≥95%；生成元谱横坐标 ≤1e-6。

## 4. 训练与评测固定设置

- 训练：AdamW 3e-4（500 步 warmup，余弦到 10%）、weight decay 0.1、梯度裁剪 1.0、batch 256、T=64、20k 步、fp32 递推状态；种子 0/1/2；数据种子与训练种子分离。
- checkpoint：每 1000 步在固定 dev-select 集（512 episode）上按 prequential 损失选择。
- 评测：T∈{64,192,576}；episode 数见 `experiments.py` 中的 SUITE_*；FloWM 协议模式 50 步上下文 + 150 步盲推。
- 课程退路（只在训练长度内学不出隐藏演化时，且对全部学习型变体统一启用）：`curriculum=True`，隐藏时长 0.25→1.0 success-gated。

## 5. 预算

S1 2、S2 30、S3 8、S4 18、S5 35、S6 20、S7 6、预留 25 GPUh；首个决策点（S0–S3）≤40；硬上限 160。
S1 实测若与估算偏差 >2×，先更新本表并告知用户，再继续。

## 6. 冻结前的残余新意检查（执行者完成后记录在此）

- [ ] LiveWorld（2603.07145）方法节：monitor 是否是线性实体动力学模型
- [ ] Koopman Dreamer（2607.19719）算子细节
- [ ] SPACE（2609.32592）细节
- [ ] MbD v1→v4 变更
- [ ] FERNN（2507.14793）
- [ ] OpenReview ICLR 2027 检索（关键词：value-side transition、predictive memory、object permanence linear attention、memory-referenced surprise）
