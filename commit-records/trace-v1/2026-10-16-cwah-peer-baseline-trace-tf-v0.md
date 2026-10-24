# CWAH Peer Baseline and TRACE-TF v0

对应提交：`e7ba288e256d6ed78daa02865baccb8a1b370460` —
`Stabilize CWAH peer baseline and introduce TRACE-TF v0`，位于 `CWAH-TF`
分支。

## 版本定位

本版本在 `e063f0e` 的初始 Peer Communication 基础上，完成了可复现的
WAH/CWAH Peer Baseline 实验流程，并建立可运行的 training-free TRACE v0。
Baseline 的场景、双 Agent、规划、通信、日志和结果保存均完成真实 Episode
验证；TRACE Updater 仍是需要继续优化的研究原型。

## 主要改动

- 正式实验改用原始 WAH 候选数据中的显式 `dataset path + episode index`，
  Episode 与 Bob Profile 独立配置，不再依赖 CollabBench Task ID 映射。数据
  audit 和 100-Episode manifest 被纳入仓库，下载的 pickle 保持忽略。
- Alice 与 Bob 的正式配置统一使用 Peer Prompt。Active Planner 会参考任务
 进度、Action History、完整 Dialogue History 和 TRACE Context，并保持可根据
  新证据修订的 `reasonable consistency`。
- 被动回复升级为 Reactive Planner。它读取最新合法局部状态、持有物、当前
  Skill、任务进度、Action History、完整对话及可用 TRACE Belief，强制输出
  `<intention>` 与 `<reply>`；删除模型主动选择 `<no_reply/>` 的机制。
  Intention 只写入日志，不发送给队友、不替换 Runtime Skill；Reply 不占环境
  Action，也不触发免费回复链。
- 修复 Reply 缺少自身持有物状态导致消息与真实状态矛盾的问题。队友消息只
  作为声明，不被当成环境执行结果；TRACE 和 Planner 不读取隐藏 Profile、
  私有 Prompt、私有 Plan 或全局场景图。
- TRACE-TF v0 提供六维 Belief Schema、Unknown/Supported 状态、逐维
  uncertainty、证据引用和 JSON 快照。Information 仅保存 Alice 的合法 Evidence
  History，不推断 Bob 的二阶知识状态。
- Hybrid Updater 使用一次联合 LLM 推断 Intent、Preference、Commitment
  candidate 和 uncertainty proposals；Capability 与 Reliability 使用可验证
  outcome 和 commitment fulfillment 规则计算。Policy 模块分别渲染 Active
  与 Reactive Context。
- TRACE-on 的 Alice 在 Active Planning 前更新 Belief；收到 Bob 消息时，先把
  post-step partial Observation 和消息写入 Ledger，再更新 TRACE 并调用
  Reactive Planner。Bob 当前不运行 TRACE，TRACE-off 不创建 Updater。
- 轨迹增加 Evidence、Belief before/after、最终 Belief、Planner Context、
  Reactive intention/reply、调用耗时及可取得的 Reply/Updater Token usage，
  支持分析 `Evidence → Belief Delta → Decision → Action → Outcome`。
- 新增 JSON 实验运行器，支持独立选择 WAH Episode、Bob Profile、两端模型、
  TRACE mode、采样参数和 Unity executable；支持自动端口锁和有界并行运行。
  每个实验保存时间戳结果目录、配置与 Prompt hash、代码 revision 和汇总，
  不保存 API Key。批量入口直接向当前终端输出进度，不再生成 `batch_logs/`
  或 `batch_summary.json`。
- `expand_scene` 失败改为带 Task/Environment 信息的显式异常，移除交互式
  `ipdb` 路径；旧的 CollabBench `task26/` 实验脚手架随无效映射一并移除。

## 验证

- CWAH conda 环境下 31 项单元/集成测试通过。
- Python 编译检查和 `git diff --check` 通过。
- `run_many.py --config-dir CWAH/experiment_runner/configs --check` 验证两份正式
  Peer17、Episode 44、TRACE off/on 配置通过。
- WAH Episode 44、Environment 3、Unity v2.3.0、DeepSeek-V3.1 的最终并行
  验证中，TRACE-off 与 TRACE-on 均成功完成：
  - off：`1/1` 成功，63 environment steps，6 次 Reactive calls；每次均有
    非空 intention/reply，无 Reply error。
  - on：`1/1` 成功，41 environment steps，4 次 TRACE Updater calls。
- 上述结果目录和完整轨迹属于本地生成数据，由 Git ignore 规则排除。

## 已知限制

- Episode 44 来自 WAH 候选数据，不是 CollabBench 论文 Task 26 的严格复现。
- Environment 3 的 WAH Episode 可运行；Environment 6 在 Unity v2.3.0 的
  `expand_scene` 兼容问题尚未解决。
- 只有各一次 off/on 运行，63 与 41 步不能用于证明 TRACE 的稳定性能收益。
- 最终 TRACE-on 轨迹没有自然发生 Bob→Alice 的非终止消息，因此 Alice 的
  TRACE-before-Reactive 路径主要由测试覆盖。
- Updater Prompt 历史和 Token 成本仍偏高；Intent/Preference 可能长期
  Unknown；Capability/Reliability uncertainty 和 pending commitment 表达
  仍需继续优化。
- 当前版本是 TRACE-TF v0 检查点，不代表最终 Representation、Updater 或
  Coordination Policy。
