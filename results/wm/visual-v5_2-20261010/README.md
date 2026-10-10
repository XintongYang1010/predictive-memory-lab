# V5.2：黄色干扰物运动是当前 bad case 的主要可干预来源

根据 V5.1 bad cases 与网页端讨论，先登记协议，再在原有八张 H800 上完成四条件干预。保持目标、标签、时间顺序、encoder 与 reader 不变，只改变黄色干扰物的跨帧位置。256 对已知 bridge 场景中，固定黄色物体使水平位移 MSE 平均降低 **85.63%**；245/256 对场景改善。交换黄色运动使约90%的 pair-common 水平残差反号。

| 黄色干扰条件 | 水平位移 MSE | 四维 motion skill |
|---|---:|---:|
| 原始运动 | 0.0007296754 | −1.538523 |
| 固定在 t6 位置 | 0.0001040312 | +0.389818 |
| 固定在 t7 位置 | 0.0001056721 | +0.381036 |
| 交换黄色运动 | 0.0007111271 | −1.492813 |

预先登记的主效应为原始损失减去两个静止条件的平均损失：**0.000624824**，完整 world 配对 bootstrap 95% CI **[0.000573715, 0.000676973]**。匹配每个时间槽黄色位置边际分布的辅助效应为0.000615550，CI [0.000567901, 0.000663719]。两种静止位置全部报告，未挑选最好者，未对预测作集成。

![四条件结果](analysis/condition_comparison.png)

这支持指定干预对冻结 encoder→pooling→reader 整体的因果影响，限于已检查的场景。85.63% 是该干预减少的当前误差比例，不是 encoder 的责任比例或未见场景上的保证。训练中黄色位移恒为零，更准确的解释是缺少区分目标运动与干扰运动的训练约束，不能说已证明学到了黄色速度与标签的相关性。

固定黄色后仍有明显残差：水平运动方向错误约18–20%，预测幅度只有真实值的0.57–0.63倍。最差静止残余场景为300、209、142、355、246。位置外推、反射及目标运动可辨识度仍需区分；不能把 confirm 与 bridge 的 skill 差直接解释为某因素贡献，两者分母不同。

**原移动干扰任务尚未修复。** 旧 V5 gate 和 V5.1 bridge 的失败记录保持不变；本轮没有重训，没有打开 Test，没有启动 P0 或记忆方法比较。下一轮建议检验固定 encoder 下的静止控制 reader 与 nuisance-balanced reader，并使用审计未访问的新 FIT/Dev；该建议尚未执行。

## 实际执行与核验

- 科学源码/协议在执行前发布：[`9792b93`](https://github.com/XintongYang1010/predictive-memory-lab/commit/9792b93f4e0203b52a2e26ab7700eb1975150231)。
- 共享 parent allocation **415453**，节点 **dgx-097**，账户 **hansirui**，partition/QOS **llm**，实际GPU为 **8×H800 80GB**。
- GPU 提取 **415453.26**：COMPLETED，214秒，8 workers，共2048张图像槽。CPU评分 **415453.27**：COMPLETED，3秒。
- 原始RGB、目标像素和标签检查通过；新 original 特征及三个 reader 的预测与旧结果 **bitwise一致**。
- 123项CPU测试通过；独立实现重算三个 reader 的10,000次配对 bootstrap、234个数值项及2048行逐例CSV，最大数值差6.94e-18。
- 本轮科学计算0.476619 GPUh，另保守计入0.002222 GPUh管理查询；累计 **2.208254 / 60 GPUh**。新leases全部闭合。
- 最终对账时既有 IndexMem **415453.5** 和 parent allocation 均为RUNNING，未停止任何已有作业。
- **清理：远端科学失败调查现场保留，未删除。** 本轮技术诊断成功未改变原任务失败状态，按用户要求保留失败现场及已有内容。

## 结果和复现材料

- [完整分析、逐轴结果和三张图](analysis/README.md)
- [原 bad case 探索及可视化](exploration/README.md)
- [独立统计审计](independent_audit/audit_summary.json)
- [归档完整性审计](archive_completeness_review.md)
- [网页端结果讨论](web_review_after_execution.md)
- [复现说明](REPRODUCE.md)、[原始报告](run/report/report.json)、[逐例数据](run/report/per_case.csv)
- [完整远端场景 Release（含冻结reader检查点与特征缓存）](https://github.com/XintongYang1010/predictive-memory-lab/releases/tag/visual-v5_2-20261010)

Release文件为 `visual-v5_2-20261010-remote-scene.tar.gz`，196,878,854 bytes，SHA256 `ecbe23a3af9cb819059387b81b38ad24b539f8be1f4ebf07c9469a0ba9e30932`。内含675个封存文件及清单，逐文件SHA覆盖232,934,447 bytes。官方10GB I-JEPA权重按来源与SHA引用，不重复上传。GitHub回读验证收据在本目录之外的 `../visual-v5_2-20261010-verification.json`，避免自引用清单。
