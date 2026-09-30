# 运行与交接

## 环境

目标：Linux、Python 3.11、torch 2.9.1+cu128、可用 NVIDIA 驱动。
`setup.sh` 在项目 `.venv` 中复用镜像 PyTorch；版本不匹配才安装官方 cu128 wheel。
Python/CUDA 运行时或 GPU 不匹配时会明确失败，不偷偷改成 CPU 训练。
PyTorch 官方安装参考：https://pytorch.org/get-started/previous-versions/#v291

核心环境不依赖 Lightning、timm 或 NATTEN 编译。AV2 仅用于从原始 parquet/地图生成官方格式缓存。
`requirements-av2.txt` 固定 pandas/numpy/scipy/opencv/numba 等关键兼容版本。
数据只读挂载时，用 `PMR_PROCESSED_ROOT=/writable/path` 指定缓存位置。

## 输入

原始目录：`train/<scenario>/scenario_<id>.parquet` 及同目录地图 JSON；val 同理。
处理后目录：`train/scenario_<id>.pt` 和 `val/scenario_<id>.pt`。
未标注 test split 不用于训练和评分。

官方权重来源：RealMotion README 中 remotion.pth 的公开链接。
Google Drive 配额或网络故障时可手动下载，设置 `CHECKPOINT=/path/remotion.pth`。
严格检查全部 model keys，不在加载失败后偷偷换成随机权重。
`--trust-inputs` 只应用于作者权重与自己生成的缓存，启用其 Python pickle 反序列化。

## 推荐执行顺序

1. 安装，运行 CPU 单元测试和 GPU smoke。
2. 小规模运行 profile/pilot，记录 forward 时间、显存和实际场景数。
3. 确认 Keep 与已知官方基线接近。本文默认样本子集不等于完整官方验证集；复核用 `--eval-scenes 0`（全部）。
4. 跑默认首周设置，传回整个输出目录（数据和 checkpoint 不需传回）。
5. 有可信信号再增加种子/轮数和数据。

batch=1 用于严格独立的分支状态，不使用 DDP，也不占满两张卡。
纯 PyTorch 局部 attention 可能比 NATTEN 慢；实际预算以 profile 为准。
数据文件内容哈希在读入前验证；长任务故障保存 `failure.json`，不会跳过坏场景后给出不完整总分。
中断恢复只复用完成阶段。不要对同一输出目录并发运行两个进程。

## GitHub

计划仓库：`XintongYang1010/predictive-memory-lab`，默认 private。
提交作者邮箱：`xintongyang1010@outlook.com`。
身份由 GitHub 登录决定，git author email 本身不能登录账号。

如从 ZIP 接手，已登录该账号的 GitHub CLI 可创建并推送：

```bash
bash scripts/create_github_repo.sh
```

脚本先确认登录账号，拒绝覆盖已有 origin。数据、权重、上游源码副本、日志不会上传。
私有仓库的合作者需要仓库权限，或由仓库所有者提供代码 ZIP。
