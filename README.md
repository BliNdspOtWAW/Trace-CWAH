# TRACE

Teammate Representation and Adaptive Coordination through Experience.
本仓库用于私人科研备份、协作开发和服务器部署，不是公开发布包。

- `CWAH/`：WAH 任务、CollabBench 队友画像、Continuous C-WAH 和 Training-Free TRACE。
- `EntCollabBench/`：独立的企业协作 Benchmark 与 Docker 服务。
- `commit-records/`：版本变更记录；`docs/`：环境、数据与部署说明。

两个 Benchmark 共用一个 Git 仓库；当前实验开发分支为 `CWAH-TF`。
不要为了部署移动目录，也不要将 VirtualHome 的上游 `wah` 分支当作研究分支。

## 服务器部署

参见 [服务器部署指南](docs/SERVER_DEPLOYMENT.md) 和
[CWAH 实验入口](CWAH/experiment_runner/README.md)。
Git 仅同步代码、空密钥模板和文档；API 密钥、任务二进制、Unity、
VirtualHome、实验结果和 Docker 持久化数据需要分别配置或传输。

当前使用的 WAH Episode 44 不代表严格复现 CollabBench 论文 Task 26。
保留各上游项目的 README、版权和许可文件；私人仓库不改变第三方许可要求。
