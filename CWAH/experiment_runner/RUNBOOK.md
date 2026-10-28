# CWAH 实验运行与轨迹阅读

以下命令均从 TRACE 仓库根目录执行。先准备 `CWAH` conda 环境、WAH 数据集和可执行的 Unity 文件，并在 JSON 中确认路径。`.pik` 只读取可信本地来源，因为 pickle 反序列化会执行代码。

## 1. 配置环境和实验

以 Qwen API 为例：

```bash
conda activate CWAH
cp -n CWAH/experiment_runner/.env.qwen.example CWAH/experiment_runner/.env.qwen
mkdir -p CWAH/experiment_runner/configs/tracev2/local_qwen
cp -n CWAH/experiment_runner/configs/tracev2/config_lib/prepare_food_ep44_peer1_Cooperative_qwen_api_thinking_on.json.example CWAH/experiment_runner/configs/tracev2/local_qwen/peer1_qwen_on.json
```

编辑私有 `.env.qwen`：填写实际 `QWEN_API_BASE`、`QWEN_API_KEY`、`QWEN_MODEL`；按需设置 Alice/Bob 各模块的 `QWEN_*_THINKING`。示例地址 `127.0.0.1:18038` 指实验运行机器；若服务在远程，事先建立本地端口转发。再编辑复制出的 JSON：确认 `dataset.path`、`episode_index`、`executable_file`、`alice_model`、`bob_model`、`bob_profile.text_file`、`trace_mode`、`max_episode_length`、`num_runs` 和 `results_root`。

使用 self API 时，改用 `.env.self.example` → `.env.self`、`source .../env.sh self`，并从 `configs/tracev2/config_lib/` 复制匹配的 `continuous_on/off.json.example`。`env.sh` 还支持 `ds`、`jd`，使用前须自行准备对应的私有环境文件和匹配的配置。JSON 模型 ID 会覆盖环境模型 ID；换供应商时必须同时匹配 API 地址、协议、模型 ID 与 `backend`。OpenAI 兼容 Qwen 使用 `backend: "openai"`。`thinking_on` 文件名不会开启 Thinking，开关由环境变量控制。

## 2. 单次预检与运行

```bash
source CWAH/experiment_runner/env.sh qwen
python CWAH/experiment_runner/run_experiment.py --config CWAH/experiment_runner/configs/tracev2/local_qwen/peer1_qwen_on.json --check
python -u CWAH/experiment_runner/run_experiment.py --config CWAH/experiment_runner/configs/tracev2/local_qwen/peer1_qwen_on.json
```

`--check` 只检查本地配置、数据与文件，不启动 Unity 或调用模型。正式运行前确认 API 可达。单实验自动在 6390–6490 分配并锁定端口，终端显示 `assigned_port`。在配置的 `results_root/trace/` 或 `results_root/baseline/` 下查找新结果目录。

## 3. 批量对照

把需要的 `.json.example` 复制到同一个本地目录，去掉 `.example` 后缀，再逐一检查模型与 `trace_mode`。目前 `configs/tracev2/continuous/` 默认为空，先建立本地 JSON。例如 self API 的一对配置：

```bash
cp -n CWAH/experiment_runner/configs/tracev2/config_lib/prepare_food_ep44_peer1_Cooperative_continuous_off.json.example CWAH/experiment_runner/configs/tracev2/continuous/peer1_self_off.json
cp -n CWAH/experiment_runner/configs/tracev2/config_lib/prepare_food_ep44_peer1_Cooperative_continuous_on.json.example CWAH/experiment_runner/configs/tracev2/continuous/peer1_self_on.json
```

`run_many.py` 会运行目录直属的**所有** `.json`；单项试跑使用单实验命令。

```bash
source CWAH/experiment_runner/env.sh self
python CWAH/experiment_runner/run_many.py --config-dir CWAH/experiment_runner/configs/tracev2/continuous --check
python -u CWAH/experiment_runner/run_many.py --config-dir CWAH/experiment_runner/configs/tracev2/continuous --max-parallel 2
```

此批量示例要求该目录内全是与 self API 匹配的配置。上面的 Qwen 示例单独放在 `local_qwen/`；一个 shell 的加载器不会按每份配置自动切换服务。`--max-parallel` 是并发实验数，不是画像数。

## 4. 阅读结果和轨迹

先看 `summary.json` 的 `status`、错误、步数、任务进展和模型指标；再用 `config.json` 核对模型、Prompt 哈希、Thinking 开关与数据来源。`calls.jsonl` 只记录调用耗时、Token 和状态，不含完整回答。`console.log` 用于定位运行时错误。要逐步看 Prompt、回答、动作和 TRACE 更新，导出轨迹：

```bash
python CWAH/experiment_runner/inspect_trajectory.py CWAH/experiments/tracev2/trace/<结果目录>/trajectory.pik --output /tmp/cwah-trajectory.txt
python CWAH/experiment_runner/inspect_trajectory.py CWAH/experiments/tracev2/trace/<结果目录>/trajectory.pik --run-index 0 --include-prompts --full-trace --output /tmp/cwah-trajectory-full.txt
```

第一条便于顺读双方可见输出、选定及实际执行动作、对话和 TRACE 更新；第二条展开 Prompt、证据与完整信念。导出不修改原始归档。新轨迹可能包含服务返回的原始 `<think>`，但不能保证服务暴露全部内部推理；旧轨迹保存前丢失的部分无法恢复。

审阅当前静态 Prompt 组合可运行 `python CWAH/experiment_runner/render_prompt_flow.py`，输出到仓库根目录 `docs/CWAH_PROMPT_FLOW_REVIEW.txt`；某一步的实际 Prompt 以轨迹为准。离线意图后验比较见 `docs/OFFLINE_INTENT_AGREEMENT.md`，不会随实验自动运行。
