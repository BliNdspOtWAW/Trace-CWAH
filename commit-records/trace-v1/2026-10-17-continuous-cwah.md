# Continuous C-WAH

对应提交：`cb40191d0da578361f5d9f6b9b72bb03785cf7f9` —
`Implement Continuous C-WAH`，位于 `CWAH-TF` 分支。

## 版本定位

本版本在可运行的 WAH Episode 44 / Environment 3 上增加 Continuous C-WAH
实验模式，用持续补充的公共 Subtask 延长协作轨迹。它不改变 TRACE 的表示、
更新触发或 Agent 间委派机制；Unity 场景、物体状态、Agent 状态、对话历史和
TRACE Belief 在整个 Step Budget 内持续保留。

## 主要改动

- 新增独立 `ContinuousSubtaskManager`。初始目标沿用 Episode 44 的 Cupcake、
  Pudding 和 Apple 到 Kitchen Table，并具体化到物体实例 ID。
- 每当一个 Subtask 完成，立即从当前环境状态随机选择可搬运物体和配置的合法
  目标位置，补充到指定的活跃数量；当前配置保持 3 个活跃 Subtask。
- 每个候选依次检查：尚未满足、物体未被 Agent 持有、未被其他活跃 Subtask
  占用、目标容量未满、目标与关系合法、未违反同一物体的反向搬运冷却。
- 随机生成失败时重新执行完整采样流程，单轮最多尝试 64 次。单轮失败不会
  结束实验，而是保留剩余活跃 Subtask，并在下一环境步继续补充。
- 支持多个目标位置和关系：Kitchen Table、Coffee Table、Desk 使用 `on`，
  Fridge 使用 `inside`。目标白名单、容量、随机种子和冷却步数由实验配置控制。
- Agent 和 LLM Planner 支持精确的“物体实例 → 目标实例”高层动作，例如
  `[goput] <apple> (375) <fridge> (103)`，并根据关系转换为 Unity 的
  `[putback]` 或 `[putin]`。
- 公共 Goal 在正常决策入口刷新，不清空当前 Skill、Action History、Dialogue
  History 或 TRACE。生成新 Subtask 本身不调用 TRACE Updater，也不作为队友
  Evidence。
- 修复旧 `get_goal()` 使用全局字符串替换造成的精确 ID 损坏：例如对象 ID
  `372` 和目标 ID `72` 同时出现时，现按解析字段重建 Predicate。
- Experiment Runner 增加 Continuous 配置校验、CLI 传递、终端标识和结果
  元数据；提供 TRACE off/on 示例配置和使用文档。本地运行配置与生成结果仍由
  Git ignore 排除。

## 验证

- CWAH conda 环境下 40 项单元/集成测试全部通过。
- `git diff --check` 和两份 Continuous off/on 配置预检通过。
- Unity 无 LLM Smoke Test 成功完成 Episode 44 初始化、精确目标同步和环境步进。
- 使用 DeepSeek-V3.1 完成一次真实 TRACE-on 验证：Episode 44、Peer17、
  Continuous 模式、100 Environment Steps，Runner 退出码为 0。
- 该轨迹累计完成 10 个 Subtask，结束时仍保持 3 个合法活跃 Subtask；实际覆盖
  Kitchen Table、Fridge、Desk 和 Coffee Table 目标，包含 `putin` 与 `putback`
  两种执行路径。没有候选生成失败、Unity 动作错误或异常退出。
- 成功结果保存在本地
  `CWAH/experiments/trace/prepare_food_ep44_profilepeer17_trace-on_continuous_statebased100_20261017-115756-173238/`，未纳入 Git。

## 已知限制

- Continuous 模式中的 `success=1` 表示 Runner 正常运行到 Step Budget，并不表示
  截止时所有活跃 Subtask 已清空；主要任务指标应使用完成数和单位步数吞吐量。
- 当前目标容量是实验层的保守逻辑容量，不是 Unity 几何碰撞或真实容积计算。
- 当前随机生成仅在 Episode 44 原任务物体类别中采样，不扩展到场景中的所有
  `GRABBABLE` 类别。
- 100-step TRACE-on 验证调用 Updater 29 次，约消耗 30 万 Token；完整历史带来
  明显成本，后续需要优化 Evidence Selection 或上下文压缩。
- 本次单轨迹验证只能证明机制可运行，不能证明 TRACE 的性能收益。
