# 网页学习交接：发布与回读收据

本次仅整理项目材料，没有新增实验、GPU 作业或远端目录。

- [完整讲解](../WEB_PROJECT_FULL_CONTEXT.md)
- [分步教学提示词](../WEB_PROJECT_CHAT_PROMPT.md)
- [下载阅读包与单文件讲解](https://github.com/XintongYang1010/predictive-memory-lab/releases/tag/web-project-handoff-20261010)

冻结阅读包对应提交 `cd478951d710237d7736cdb9c3a981d1dfc4bf3d`。先前科学结果快照为 `cb1d31315d1c936ca93afe1cb18460e9d3d57778`；确认包含用户指定的 `df9e780fbe8372269ab4066c11231ea04b19e4e3`。

## 已完成的核验

1. 从 GitHub 新下载原科学快照，对全部 1,094 个文件验证 Git blob 身份，见 [scientific_git_readback.json](scientific_git_readback.json)。
2. 发布新交接后，从 GitHub API 读取完整 1,105 文件树，全部 blob ID 与本地提交一致；新增或修改的16个文件另逐字节下载核对，见 [handoff_git_readback.json](handoff_git_readback.json)。
3. Release 上传结束后，在新目录完整下载4个附件，全部字节与上传原件相同；ZIP 中1,105个成员逐一校验字节数、SHA256和Git blob ID，见 [github_release_verification.json](github_release_verification.json)。
4. 讲解和入口的227个本地链接存在，`git diff --check` 通过。三例30行位移与原 JSON/CSV 对齐；项目状态另经独立审读。本次没有代码行为变化，因此未重复跑训练或软件测试。

## 资产范围

`web-project-git-snapshot.zip` 包含冻结提交的全部 Git 文件，约33.2MB；`WEB_PROJECT_FULL_CONTEXT.md` 为约90KB的合并讲解；另含提示词和逐文件清单。

旧 V5 检查点/缓存继续放在原 Releases，本次只索引、不重复上传或重新下载全部历史大资产。旧 V1–V4 的独立结果分支以固定提交索引。**V4 约343MiB完整包当时存于 Slack/服务器，本次未镜像到 GitHub，也未验证当前外部可达性**；小型结果报告可从 GitHub 读取，详见 [资产索引](../WEB_PROJECT_ARTIFACT_INDEX.md)。外部官方权重和数据仍依照既有来源与哈希恢复。

本目录的发布/回读收据是资产创建后保存的 Git 文件，不递归塞入它们所验证的 ZIP。后续包含本收据的分支提交不改变冻结阅读包的源码身份。

仓库可见性仍为私有；所有已有本地原件、远端现场与作业均保留。本次没有适用的远端清理步骤。
