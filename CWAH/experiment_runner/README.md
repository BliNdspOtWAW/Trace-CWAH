# 手动运行 WAH + 独立队友画像实验

当前默认角色模板为 Alice 的 `prompt_Alice.txt` 和 Bob 的 `prompt_Bob.txt`，
而非旧的共用 Peer 模板；各自的 Reply 模板也已拆分。Continuous 目录当前提供
Peer23/24、JD 模型的模板，其他画像见 `configs/config_lib/`。
任务生成不会额外触发 TRACE；任务完成会作为合法 Evidence 记录。
下文旧示例若提及 Peer17 或共用 Peer，应以实际 JSON 和运行时打印的 Prompt 为准。

## API 配置入口（DS / JD）

API 配置统一保存在本目录：`.env.ds` 和 `.env.jd`，均被 Git 忽略。
可提交的空密钥模板是 `.env.ds.example` 和 `.env.jd.example`。
DS 已从之前的本地设置迁移，之后不再读取 EntCollabBench 的 `.env`。

```bash
source CWAH/experiment_runner/env.sh ds
# 或
source CWAH/experiment_runner/env.sh jd
```

DS 使用 Chat Completions；JD 使用 Responses，密钥填写在 `.env.jd` 的
`PROXY_OPENAI_API_KEY`。实验 JSON 中的 `alice_model` / `bob_model` 优先于
env 的默认模型，切换供应商时也要选择匹配的实验 JSON。
旧 `scripts/env.local.sh`（DS）和 `scripts/env.jd.sh`（JD）仍可使用，内部
转到此统一入口；不要在 shell 开启 `set -x` 时加载密钥文件。

这里的任务来自原始 WAH 候选数据，**不是 CollabBench 论文 Task 26 的严格复现**。任务与 Bob 画像分开选择：`episode_index` 选择 WAH Episode；`bob_profile.text_file` 直接读取画像文本，不根据 Task ID 做任何映射。Alice 不加载画像。`profiles/peer{8,17,23,24}.txt` 是从旧 CollabBench `26_prepare_food` 画像表复制的文本，供迁移实验使用；它们不是 WAH 原生画像。也可以换成自己的 UTF-8 文本文件，或把 `bob_profile` 设为 `null`（无画像）。

## 第一次配置

在 TRACE 仓库根目录执行：

```bash
conda activate CWAH
cp CWAH/experiment_runner/config.example.json CWAH/experiment_runner/config.local.json
```

编辑 `config.local.json`：

- `dataset.path`：WAH 数据文件路径，相对于 TRACE 根目录，也可用绝对路径。
- `dataset.episode_index`：**列表索引**，例如 `44`，不是 CollabBench 论文 Task ID。
- `bob_profile`：`id` 仅用于结果目录标签；`text_file` 才是 Bob 实际收到的画像内容。两者不参与任务选择。无画像时填 `null`。
- `trace_mode`：`off` 完全不创建 TRACE Updater、不向 Alice Planner 注入 Belief；`on` 才启用。Bob 始终不读 Alice 的 TRACE Belief。
- `continuous_subtasks`：可选的动态 Subtask 配置。`enabled=true` 时，环境保持指定数量的物体级 Subtask；完成一个后，根据当前场景随机选择可搬运物体和白名单目标，经“未满足、未被持有或占用、不与活跃任务冲突、容量和反向搬运冷却”校验后补充。失败会重新采样，并在后续环境步继续尝试。`seed` 使采样序列可复现。该机制不创建委派、不写入 TRACE Evidence，也不额外触发 TRACE Updater。
- `alice_prompt_file`、`bob_prompt_file`：可选的 Planner Prompt 文件路径；默认两人都用 `LLM/prompt_Peer.txt`。路径相对于 TRACE 根目录，结果 `config.json` 会记录实际路径和 SHA-256。TRACE `on` 时 Alice 的 Prompt 必须包含 `$TRACE_BELIEF`。
- `experiment_label`：可选的结果目录标签，仅接受字母、数字、`_`、`-`。
- `alice_model`、`bob_model`：两人的实际 API 模型 ID，可以分别设置。
- `temperature`、`max_tokens`、`max_episode_length`、`num_runs`：单次实验参数。`num_runs` 建议先设为 1。`port` 可省略或设为 `"auto"`（默认），也可指定整数端口。
- `executable_file`：Unity 可执行文件路径；默认示例为本地 v2.3.0。

模型的 API Base/Key 从当前 shell 的 `CWAH_AGENT_API_BASE`、`CWAH_AGENT_API_KEY`、`CWAH_HUMAN_API_BASE`、`CWAH_HUMAN_API_KEY` 获取；配置文件不存密钥。可以先加载已有的本地配置（请勿提交）：

```bash
source CWAH/CWAH-MultiPlayer/scripts/env.local.sh
```

Runner 会用 JSON 中的模型 ID 覆盖环境中的模型 ID，不覆盖 API Base/Key。若你不使用本地脚本，也可以在 shell 中单独 `export` 这些变量。请确保当前 shell 已激活 `CWAH` conda 环境。

## 检查与运行

以下命令都从 TRACE 根目录执行。`--check` 仅读取本地配置和可信数据，不启动 Unity、不调用 LLM、不创建结果目录：

```bash
python CWAH/experiment_runner/run_experiment.py --config CWAH/experiment_runner/config.local.json --check
```

确认后运行：

```bash
python CWAH/experiment_runner/run_experiment.py --config CWAH/experiment_runner/config.local.json
```

Runner 自动从 6390–6490 中选择空闲端口，并在整个实验期间持有跨进程锁；不会结束已有进程。若你手动指定整数端口且该端口被占用，会报错。实际端口打印为 `assigned_port=...`，也保存在结果 `config.json` 中。单次运行会创建唯一目录：

```text
CWAH/experiments/baseline/prepare_food_ep44_profilepeer17_trace-off_<时间>/
CWAH/experiments/trace/prepare_food_ep44_profilepeer17_trace-on_<时间>/
```

每个目录含 `config.json`（生效参数、数据及画像 SHA-256、代码 Commit、API 端点主机名，不含 Key）、`console.log`、`results.pik`、`logs_agent_*.pik`、`summary.json` 和 `status.md`。`summary.json` 汇总成功次数、步数、两人的 Planner 调用数、即时回复调用数以及可取得的 TRACE Updater Token 用量；其他模型的 Token 统计未必可得，不应把 0 解释为零成本。原始轨迹包含完整对话和 Prompt，结果目录被 Git 忽略。

## 多配置并行实验

在 `CWAH/experiment_runner/configs/` 下放多个 `*.json`。该目录中的本地 JSON 被 Git 忽略；仓库提供了两个 `.json.example` 样例。复制后按需修改 Episode、画像、模型和 `trace_mode`；**不需要手动分配端口**。同一 WAH 数据文件和 Unity 可执行文件可供多个实验读取和启动，不需要复制。示例：

```bash
cp CWAH/experiment_runner/configs/config_lib/prepare_food_ep44_peer17_continuous_off.json.example CWAH/experiment_runner/configs/peer17_off.json
cp CWAH/experiment_runner/configs/config_lib/prepare_food_ep44_peer17_continuous_on.json.example CWAH/experiment_runner/configs/peer17_on.json
```

先在同一个 shell 中激活 conda 环境并加载 API 配置。批量预检与并行启动：

```bash
conda activate CWAH
source CWAH/CWAH-MultiPlayer/scripts/env.local.sh
python CWAH/experiment_runner/run_many.py --check
python CWAH/experiment_runner/run_many.py --max-parallel 2
```

`run_many.py` 默认只选 `configs/` 下以 `.json` 结尾的文件，先校验全部配置（仅在手动指定整数端口时检查重复），随后最多同时运行 `--max-parallel` 个。每个子进程会自动领取并锁定一个空闲端口。如果配置放在别处，可指定 `--config-dir CWAH/exp_config`；注意该自建目录默认不被 Git 忽略。

批量运行不会创建额外的 `batch_logs/` 或 `batch_summary.json`。各子实验的启动信息、实际端口、结果目录、Step 进度和最终退出码直接显示在当前终端。完整 Prompt、动作、错误、配置和汇总仍分别保存在每个实验自己的结果目录中，包括 `console.log`、`config.json`、`summary.json` 和轨迹文件。当 30 秒仍有子进程运行时，批量入口会显示当前活动的配置名；模型调用较慢并不表示实验失败。

也可以在不同 Bash 中分别执行单实验命令，例如：

```bash
python CWAH/experiment_runner/run_experiment.py --config CWAH/experiment_runner/configs/peer17_off.json
python CWAH/experiment_runner/run_experiment.py --config CWAH/experiment_runner/configs/peer17_on.json
```

不同 Bash 各自需要激活 `CWAH` 环境并加载 API 变量。单实验入口同样自动分配端口，并使用跨进程锁；即使两个 Bash 同时启动，也不会领取同一个端口。顺序运行可复用端口。多个 Unity/LLM 请求会占用更多内存与 API 配额，建议先用两个实验试运行。

对照实验应在相同 Episode、Bob 画像、两个模型、采样参数与 Unity 版本下分别运行 `trace_mode=off` 和 `on`。这里的 off 是**当前统一 Peer/Reply Prompt 下的对照**，不是旧版 CoELA Prompt。当前工作树如果有未提交修改，单靠 Commit SHA 不能完全恢复实验代码；正式实验前请保存补丁或提交版本。

## Continuous C-WAH

`configs/continuous/` 提供 Episode 44、Peer17、TRACE off/on 的成对配置。初始任务仍是 Episode 44 的三个 `prepare_food` 目标；之后从当前场景状态随机生成具体物体实例到合法目标位置的新 Subtask。示例目标包括 Kitchen Table、Coffee Table、Desk 和 Fridge，且同时活跃的 Subtask 不会争用同一物体。Unity、物体位置、Agent 执行状态、Dialogue History 和 TRACE Belief 均不重置。

Subtask 更新是公共任务状态变化，不是队友证据。Agent 在下一次正常高层规划时读取最新 Goal；不会仅因补充 Subtask 立即重规划或调用 TRACE。随机生成在单次补充达到 `max_generation_attempts` 后暂时保留已有活跃任务，并在下一环境步重试；不会因候选暂时生成失败提前结束。Continuous 运行仅以 Step Budget 结束。结果轨迹中的 `continuous_subtasks` 和 `continuous_subtask_events` 记录具体物体、目标、累计完成数、采样次数及拒绝原因，`summary.json` 会复制最终汇总。

运行命令：

```bash
python CWAH/experiment_runner/run_many.py \
  --config-dir CWAH/experiment_runner/configs/continuous \
  --check

python CWAH/experiment_runner/run_many.py \
  --config-dir CWAH/experiment_runner/configs/continuous \
  --max-parallel 2
```

## Active 与 Reactive Planner

正式实验使用角色专属的 `prompt_Alice.txt` / `prompt_Bob.txt`，回复使用
`prompt_Reply_Alice.txt` / `prompt_Reply_Bob.txt`。Active Planner 在需要新高层动作时运行，
输出合法 Action；Reactive Planner 在收到主动消息后运行，强制输出内部
`intention` 和实际 `reply`。`intention` 只保存在轨迹中，不发送给队友，也不
替换当前 Runtime Skill。`reply` 进入双方完整 Dialogue History，供后续 Active
Planner 保持合理的跨轮次一致性。

TRACE-on 的 Alice 在回复 Bob 前先将新消息写入 Evidence Ledger 并更新 Belief，
然后把精简的 TRACE Context 交给 Reactive Planner。Bob 当前不运行 TRACE；
TRACE-off 时两名 Agent 都只使用 Goal、Personality、合法观察、任务进度、
Action History 和 Dialogue History。每条成功处理的主动消息必须产生一次回复，
模型格式错误或 API 异常会记录为 `reply_error`，不再作为主动沉默处理。
