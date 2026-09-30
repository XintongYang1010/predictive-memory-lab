# 验证记录

这些记录描述实际执行过的软件检查，不代表模型效果或 A100 性能。

|检查|实际结果|
|---|---|
|目标 Python 安装路径|Python 3.11.16 全新虚拟环境，运行 `setup.sh --cpu --av2` 成功|
|核心框架|PyTorch 2.9.1+cpu，numpy 1.26.4，AV2 0.3.6，pandas 2.2.3，pyarrow 19.0.1|
|单元/协议测试|23 项通过；无跳过|
|完整 smoke|两轮修复监督刷新、每轮 30 步训练、独立合成场景评估、报告生成成功|
|官方 checkpoint|公开 remotion.pth 已实际下载；256 个权重键严格匹配，2,908,233 参数|
|真实 AV2 格式|AV2 API 仓库的公开单场景测试 fixture 已实际完成 parquet/地图预处理|
|官方权重 + 真实 fixture|八种干预均前向成功，输出有限；no-op、分支隔离、未来盲性、场景清空检查通过|
|CUDA 12.8 / A100|尚未实测；需在目标机器运行 GPU doctor、smoke 和 profile|
|完整 AV2 基准分数|尚未跑；单 fixture 的数值不用于研究结论|

官方权重 SHA256：

`5f3c9f9cffecd094ee0fccce42a54d1797c77b2b3491393855f1846a71630f1c`

AV2 fixture 来源：

https://github.com/argoverse/av2-api/tree/main/tests/unit/test_data/forecasting_scenarios/0a1e6f0a-1817-4a98-b02e-db8c9327d151

Parquet SHA256：

`b7790ba7092dbb60d268e8e43d8f920236fb4cb5e6b8864ca7706a879e84e455`

测试覆盖 shifted neighborhood attention 的前向/梯度、坐标刚体变换、绝对残差、新增证据边界与重复调用、缺失观察、随机删除数量匹配、Reset 路径、Rebuild 首步 no-op、预处理前的未来标签修改、公共绝对预测区间、ES 手算例、空目标拒绝、场景分割、权重严格加载及已知 wrapper 前缀。

未将第三方 fixture、模型权重或合成实验结果附入代码包。目标机器运行时会生成自己的完整验证与实验日志。
