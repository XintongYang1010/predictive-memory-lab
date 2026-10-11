# V5.7：已冻结代价面的事后诊断

输入为既有 256 个 world、512 个 Dev episode 的完整 625 点代价面。没有重新编码或拟合。

GT-y/GT-x 使用真实标签，是特权定位控制；错配 GT-y 固定循环平移 37 个 world。以下 GT-y 比较只解释未提供的 dx；提供 oracle 轴后的 4D 指标不能当可部署性能。

| Arm | 原 2D dx RMSE (px) | GT-y dx RMSE | y=0 dx RMSE | 错配 GT-y dx RMSE | 原−GT-y dx MSE, 95% CI (px²) |
|---|---:|---:|---:|---:|---|
| raw_native | 1.5343 | 1.5617 | 1.5082 | 1.5169 | -0.0848 [-0.2732, 0.0461] |
| pool_native | 3.8780 | 4.0803 | 3.9299 | 4.0058 | -1.6098 [-2.2815, -1.0076] |
| uniform_budget | 3.7531 | 4.3093 | 3.9733 | 4.1437 | -4.4842 [-5.5104, -3.5140] |
| roi_adaptive_budget | 1.7384 | 1.8503 | 1.7287 | 1.7996 | -0.4014 [-0.7265, -0.1468] |
| random_budget | 4.7724 | 5.0043 | 4.7003 | 4.8511 | -2.2668 [-3.4016, -1.1343] |
| change_priority_budget | 6.6630 | 6.7697 | 6.6673 | 6.7149 | -1.4333 [-2.6523, -0.2336] |

方向计数按 |真实 dx|≥1 px，wrong、zero 和合计 non-correct 分开；完整 per-axis 分母、GT 网格化误差、out-of-support、全局/行内 rank、方向修复及伤害见 report.json 和逐例 CSV。

错配 donor 与真实 y 落在同一网格行：95/1024；这些预测应精确一致。报告另列不同网格行子集的 dx MSE 差，bootstrap 同时重采样每个 world 的误差和子集数量。

MSE 使用原 normalized displacement；px²=MSE×224²。四轴 skill 是逐轴 1−MSE/mean((GT−冻结 FIT 均值)²) 后求均值，不使用 Dev 方差。

95% CI 使用固定 seed 7817 的 10,000 次 whole-world bootstrap，同时保留两个 variant 和两个对象。区间条件于当前代价面和固定规则，多臂描述性比较未做多重校正。

几何 oracle 下界仅适用于当前有限网格。GT-nearest-grid 的 cost/rank 仅描述这张固定代价面，不能证明 descriptor 无信息；oracle 收益也不是某模块造成的错误比例。没有新确认、Test 或 P0 授权。

三张图固定选择既往 761/1、844/1、891/1，未按本轮效果选例。
