# 私人仓库与服务器部署准备

Parent: `4cc2d6d`；分支：`CWAH-TF`。本记录随本次变更提交，提交 SHA 以 Git 日志为准。

- 区分 Alice/Bob 的 Active 和 Reactive Prompt；Bob 保留画像，Alice 不加载画像。
- 对齐两类 Planner 的 TRACE 上下文与决策引导，保存实际 Prompt 路径及摘要。
- 支持 JD Responses 协议、独立模型名、推理预算和请求重试；保留 DS Chat Completions。
- API 配置统一到 runner 的 `.env.ds` / `.env.jd`，仅提交空密钥模板和加载脚本。
- 整理 Peer23/24 Continuous 模板及配置库，补充 Responses 与角色通信测试。
- 增加根 README、服务器部署指南和 Ent Agent 配置模板，补齐直接下载/HTTP 依赖。

密钥、原始二进制任务、Unity、VirtualHome 和运行结果不纳入本提交。
未在全新服务器重新安装；没有在本次整理中启动完整付费实验。
本次验证：63 项单元测试通过，pip check、Shell 语法检查和 git diff --check 通过。
TRACE 仍为探索版本，不代表模型推断、动作回退与 Reliability 归因已完全解决。
