# CWAH Peer Agent 通信机制

对应提交：`e063f0edc38dae523d9d0fed248d6e4ae617cc3f` — `Implement CWAH peer communication (one-step smoke tested)`，位于 `CWAH-TF` 分支。

## 本次改动

- Alice 与 Bob 改用统一的 `CWAH/CWAH-MultiPlayer/LLM/prompt_Peer.txt`。双方有平等的规划、沟通、协商和拒绝权限；发现目标物品、完成子任务或获得重要信息时，发送消息是有条件的推荐，而非绝对要求。Bob 仍加载指定画像，Alice 默认不加载。
- 新增 `LLM/prompt_Reply.txt` 与 `LLM/communication.py`：一次主动 `[send_message]` 仍占一个环境步；环境步结束后，接收方可调用独立模型立即回复或选择不回复。每条主动消息最多触发一次即时回复，回复不触发连锁回复，也不清空接收方当前 Skill。
- 双方所有实际消息按顺序进入完整 Dialogue History。高层规划 Prompt 不再只截取最近三条；回复 Prompt 仅使用接收方可见信息、当前 Skill 和对话历史。
- 轨迹增加通信事件、即时回复模型调用次数及 API 返回的 token usage；缺失 usage 时单独标记，不将零值当作免费调用。规划调用也保存 API 返回的 token usage。
- 保留此前的任务画像来源映射：可将实际 WAH 任务的队友画像查找映射到 CollabBench `26_prepare_food` 的同一 cluster，不改任务数据或 Alice 的画像。

## 验证

- `CWAH` conda 环境下 15 项单元测试通过；Python 编译、Shell 语法和 Git diff 检查通过。
- WAH 候选数据 Episode 44、Unity v2.3.0 上的一步真实集成测试通过：Alice 主动发信后，Bob 完成一次真实模型回复；环境步数 `0 → 1`，双方历史各有两条消息，Bob 原 Skill 未被改变，API 返回了 token usage。
- 测试用临时脚本已删除；此前 Episode 44 的正式 Baseline 结果未改动。

## 边界

这只验证了单步通信链路，**尚未在新机制下重跑完整 Episode 或多任务评测**，不能据此判断任务成功率或通信策略收益。Episode 44 使用原始 WAH 候选数据及迁移的队友画像，并非 CollabBench 论文 Task 26 的严格复现。先前的成功实验属于改动前 Baseline，不应与本提交后的实验直接混记。
