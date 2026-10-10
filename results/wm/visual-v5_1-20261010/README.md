# V5.1：时间信息确认通过，原任务迁移失败

本轮已在原八张 H800 上完成真实实验。结论是：**冻结 I-JEPA 的 4×4 池化特征能支持受控场景中的双帧运动读取，但本次冻结读出器不能迁移到原 V5 分布；尚无启动 P0 或比较记忆架构的依据。**

| 独立评估 | 主读出器运动 skill | 预设单侧下界 | 位置 skill | 结果 |
|---|---:|---:|---:|---|
| 256 组未见四视图确认 | 0.674117 | 0.648395（九项校正） | 0.988371 | 全部预设检查通过 |
| 原生成器 256 对世界 / 512 episodes | −1.538523 | −1.683154（95%） | 0.942354 | 迁移失败 |

确认的八个坐标对比下界全部为正，最小为 0.581202。两个单帧对照均不能通过时间确认。反对称读出器下两类 anchor 对比代数上重复，不能算作八项独立证据。

桥接中两个物体的水平位移 skill 为 −2.768、−3.358。独立标量计算核对了全部 512 条标签；按已保存预测重算的全部指标和判定完全一致。反射世界（87 对）和非反射世界（169 对）的运动 skill 分别为 −1.499、−1.558，所以仅靠反射不能解释失败。移动干扰、空间分布和读出器外推仍需单独的受控实验才能区分，当前结果不建立其中任何一个因素的因果结论。

## 实际执行与停止边界

- 既有 allocation：`415453`，节点 `dgx-097`，账户 `hansirui`，分区/QOS `llm`。实测硬件为八张 H800；本报告不声称 A100 性能。
- 八卡提取：FIT `415453.18`（202 秒）、确认 `415453.20`（192 秒）、bridge `415453.22`（200 秒），全部 Slurm `COMPLETED / 0:0`。
- CPU 选择/评估：`415453.19`、`415453.21`、`415453.23`；没有计作 GPU 计算。
- 科学控制器按预先登记的 bridge 失败停止。P0、A0/A1/B0/B1/C0/C1、full/recent/FIFO 和 reader-only **未启动**。原 V5 的失败门槛没有改写。Test 未开启。
- IndexMem step `415453.5` 在末次查询中仍为 RUNNING。未取消父 allocation，未停止或修改 IndexMem。
- 累计保守选卡用量 **1.729414 / 60 GPUh**，包含旧轮次、启动/退出开销和旧 Slurm elapsed 补计。此值不是新增 Slurm 账单，也不把整个已有 allocation 的占用归给 V5。全部预算 lease 和控制器预留均已闭合。

科学执行源码冻结于 [`0e50f6a`](https://github.com/XintongYang1010/predictive-memory-lab/commit/0e50f6aadbdcec9c9b787208362d9dde9b6360dd)。FIT 内部选择二次核、mean-loss alpha `7.8125e-7`；确认前已将冻结收据和全部候选分数发布于 [`03c1b8b`](https://github.com/XintongYang1010/predictive-memory-lab/commit/03c1b8bc650c7347d0e7b532db22bf4a720d8b66)，并回读核验。随后没有改模型、阈值、种子或样本数量。

## 结果与复现材料

- [完整指标、对照、图表和作业表](report/README.md)
- [事后标签核验、反射分组、残差与失败样例](diagnostics/README.md)
- [冻结检查点独立重算](independent_recompute/README.md)：跨平台预测最大差异 2.93×10⁻¹⁵，所有判定一致。
- [科学运行收据和配置](run/)
- [确认前的 FIT 冻结证据](freeze/)
- [完整远端文件清单](remote_scene_manifest.json)与[归档资产 SHA256](release_asset.json)
- [完整 GitHub Release](https://github.com/XintongYang1010/predictive-memory-lab/releases/tag/visual-v5_1-20261010)：包含全部 1,739 个清单文件（274,246,065 字节）、特征缓存/分片、冻结读出器检查点 `reader/frozen_model.npz`、源码、日志和复现配置。既有官方 I-JEPA 权重以固定下载来源和 SHA256 引用，没有声称重新训练或重复上传该第三方权重。

解压完整归档后，可用冻结目录的 reader 对已有缓存重新评估，无需重训或重新生成数据；使用新的输出目录：

```bash
python3 code/scripts/v5_confirmation_reader.py confirm --model reader --cache confirm_cache/cache.npz --receipt confirm_cache/receipt.json --out new_confirm_report
python3 code/scripts/v5_confirmation_reader.py bridge --model reader --cache bridge_cache/cache.npz --receipt bridge_cache/receipt.json --out new_bridge_report
```

本轮远端目录 `/aifs4su/guhao/KVCache/WM/continuation-20261010` **保留**，原因是科学门槛失败，遵循用户“运行或归档失败时保留现场”的要求。既有 WM 内容、原失败实验、共享环境和 IndexMem 全部保留。GitHub 回读验证收据独立存放于本目录外，避免自引用哈希。
