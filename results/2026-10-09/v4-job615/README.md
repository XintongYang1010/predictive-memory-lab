# V4 单 A100 实验交接：Slurm 615

代码：`e66bdd368e1895f62bde788f3075edcf32180415`；2026-10-09 北京时间 14:17:57–14:32:39。

## 执行与校验

- 作业 `COMPLETED`、退出码 0；正式结果 `status=complete`、`smoke_only=false`。调度总用时 882 秒，正式 runner 用时 850.78 秒，实验预算 3600 秒。
- 58 项 V4 测试和 GPU smoke 通过。5 个模型 × 3 个种子全部完成；12 个学习模型均训练 1500 步，保存 12 份 checkpoint。
- Train/Dev/Eval=4096/512/512，训练长度 24，评价长度 24/48/96，种子 42/43/44。270 个评价组、138240 条 episode 记录完整。
- 原始结果包 359646477 字节（约 343 MiB），252 个文件的大小与 SHA256 全部核验；37 个源码指纹与固定提交一致。三个主比较的 episode 配对区间已独立复算，与原报告一致。
- Python 3.11.14、PyTorch 2.9.1+cu128、CUDA 12.8、A100-SXM4-80GB。申请 1 GPU / 4 CPU，容器内存限制 40 GiB。PyTorch 峰值分配 103454720 字节，不含全部 CUDA 上下文及容器开销。

## 主比较：clean / T48

数值为 episode 等权平均的 hidden_position_rmse；收益 = GRU − slot。CI 为每个模型种子内的 512 个配对 episode、1000 次 bootstrap；三个种子共享数据，不合并成独立 episode 样本。

| 种子 | GRU | slot | 收益 | 95% CI |
|---:|---:|---:|---:|---|
| 42 | 0.767277 | 0.246115 | 0.521162 | [0.498093, 0.542348] |
| 43 | 0.726502 | 0.251029 | 0.475473 | [0.450471, 0.498985] |
| 44 | 0.719027 | 0.297719 | 0.421307 | [0.397860, 0.444411] |

三组主比较均偏向 slot；同时须保留以下限制：

- T96 仍有明显隐藏状态漂移。seed44 的 slot preserved_post_position_rmse=3.922922，GRU=3.411755；不能将 T48 的改善推广为所有长度和所有对象均改善。
- T96 的 analytic hidden_position_rmse=0.564435；slot 为 1.481270 / 1.506818 / 2.080200。恒速解析对照仍比学习模型稳健。
- slot 的 unsolicited_update_rmse=0 来自对象独立结构；这本身不能证明学会局部修正。已知对象 ID、直接测得速度、合成恒速动力学构成强先验，模型也未按参数或 FLOP 完全匹配。
- 本轮只支持这个合成任务中的机制观察，不支持视觉 JEPA、真实数据效果或论文新颖性结论。

## 资源映射异常

外层作业日志中的 GPU UUID 为 `GPU-4fd11e10-6ef5-dd51-fa9c-4bb55aae271a`，容器内日志为 `GPU-174ae482-12b7-8b2c-0582-4921ad4803af`，两者不同。启动器沿用 Slurm 的 remapped ordinal `0` 传给容器；按平台 UUID 对照，外层记录对应物理卡 6，容器记录对应物理卡 0。**本次计算完整，但“使用调度器分配的物理卡”这一项未通过核验。**

保留两层原日志和实际执行的启动器（`launcher_used/`）。资源后检查与作业结束状态见 `resource_after.json`、`execution.json`；未自动重跑或操作其他用户任务。

`launcher_next/` 是修正后的本地模板：传入分配可见卡的 UUID，并在容器内核对相同 UUID 后才启动测试。语法及正确/错误/多卡 UUID 的防误用检查通过；尚未提交或做新的 A100 实测。未来使用前须选择新目录，不能覆盖本轮结果。

## 完整结果包

原始 .tar.gz 与 .tar.gz.sha256 发到 [Slack V4 交接线程](https://dn-lab-hq.slack.com/archives/C0C7S3PFJSE/p1791512716621849)，三名项目成员均有频道权限。343 MiB 的完整附件保留在 Slack 与服务器，本 PR 保存报告、汇总、训练曲线、执行日志与校验记录。

- [下载完整 .tar.gz](https://dn-lab-hq.slack.com/files/U0C4D2NER62/F0C7TNW6ZCK/state_memory_v4_20261009t061757z_615.tar.gz)
- [下载 .tar.gz.sha256](https://dn-lab-hq.slack.com/files/U0C4D2NER62/F0C81UNMVS5/state_memory_v4_20261009t061757z_615.tar.gz.sha256)

```text
state_memory_v4_20261009T061757Z_615.tar.gz
SHA256 21541e4d6a09750edc2af3b5c8ae8bcc6a902d3e39adbfd6a8b2d7c800bc1beb
```

包内包括全部 checkpoint、逐 episode 记录、完整轨迹、配对 CI、失败案例和逐文件清单。原始包未修改。

服务器原目录：`/share/home/s2026316084/yxt_pmr_v4_20261009/source_e66bdd3/runs/state_memory_v4_20261009T061757Z_615`。

复现使用固定提交与原环境，在新目录执行：

```bash
bash scripts/retest_v4.sh /share/home/s2026316084/yxt_pmr_20261002
```

提交前须修正并实测上述平台 GPU UUID 绑定；实验算法、数据及训练配置未改动。

详细原报告见 [REPORT.md](REPORT.md)，完整汇总见 [summary.json](summary.json)，校验记录见 [verification.json](verification.json)。
