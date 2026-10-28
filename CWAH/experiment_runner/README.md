# CWAH experiment runner：代码说明

本目录用 WAH 数据集启动 Unity 双人实验，选择 Bob 画像与 TRACE 条件，并归档结果。`episode_index` 是 WAH 数据集列表索引，不是 CollabBench Task ID。操作命令见 [RUNBOOK.md](RUNBOOK.md)。

## 运行流程

1. `run_experiment.py` 读取 JSON，检查数据、Unity 可执行文件、画像和 Prompt；`--check` 到此结束。正式运行时分配并锁定端口，为每次实验创建独立结果目录，再启动 `CWAH-MultiPlayer`。
2. `run_many.py` 非递归扫描指定目录直属的 `*.json`，验证所有配置，按 `--max-parallel` 启动单实验进程。它不读取 `.json.example`。
3. `env.sh` 从私有 `.env.self`、`.env.qwen` 等文件向当前 shell 加载 API 地址、密钥、协议和 Thinking 参数。JSON 的 `alice_model`、`bob_model` 会覆盖加载器提供的模型 ID；只改模型名不会自动切换服务。JSON 的 `backend` 选择 OpenAI 兼容 API 或本地 Hugging Face 模型。
4. `CWAH-MultiPlayer/LLM/` 组合角色专属 Active、Reactive Prompt。`trace_mode=on` 时 Alice 的 Updater 在规划前及处理 Bob 消息前更新 TRACE 信念，信念随后进入 Alice 的 Prompt；Bob 不读取 Alice 的信念。`trace_mode=off` 不创建该 Updater。
5. `continuous_subtasks.enabled=true` 时，已完成的物体级子任务根据当前场景补充，Unity、对话与 TRACE 状态继续运行直到步数上限。补充子任务本身不触发 Updater。

## 目录和脚本

| 路径 | 作用 |
| --- | --- |
| `config.example.json` | 单实验配置起点，复制为被 Git 忽略的本地 JSON。 |
| `configs/tracev2/config_lib/*.json.example` | 五种 Bob 画像 × TRACE off/on 的模板，以及 Qwen API 示例。 |
| `configs/tracev2/continuous/*.json` | 本地批量配置；被 Git 忽略，可从模板恢复。 |
| `configs/tracev1/`、`profiles/tracev1/` | 历史实验模板与画像。 |
| `profiles/tracev2/` | 当前 Bob 行为画像，由 `bob_profile.text_file` 显式选择。 |
| `metric_reports.py`、`result_storage.py` | 模型指标汇总、轨迹和结果归档。 |
| `inspect_trajectory.py` | 从可信的 `trajectory.pik` 导出逐步可读文本。 |
| `render_prompt_flow.py` | 展开当前静态 Prompt 组合供审阅。 |
| `compare_intent_posteriors.py` | 独立的离线意图后验比较，不参与在线实验。 |
| `download_qwen3.py` | 本地 Hugging Face 模型下载辅助工具，与 Qwen API 无关。 |

tracev2 标准模板默认使用 Episode 44、200 步、一次重复、三个持续子任务及 seed 1。对照实验需核对 on/off 两份 JSON 的任务、画像、模型、采样参数和 Unity 版本一致。文件名中的 `thinking_on` 只是标签；真正的 Qwen Thinking 开关由 `.env.qwen` 控制。

## 结果

每次运行写入 `CWAH/experiments/` 下所选 `results_root` 的 `trace/` 或 `baseline/` 子目录。正常结果包括 `config.json`（生效配置与来源哈希）、`summary.json`（结果和指标）、`calls.jsonl`（逐调用指标）、`trajectory.pik`（逐步轨迹）及 `console.log`。异常中断可能保留检查点与 `.runtime/`。`calls.jsonl` 的调用或解析成功不等于任务完成，且不含模型正文。新轨迹可能包含服务返回的 `raw_chat_output` 和 `<think>`；旧轨迹已丢失的内容无法恢复。

私有 `.env.*`、本地 JSON、运行结果和锁文件不进 Git。`CWAH/virtualhome/`、`CWAH/executable/` 是外部本地依赖。算法与指标细节见仓库根目录的 `docs/TRACE_V2.md`、`docs/LLM_METRICS.md` 和 `docs/OFFLINE_INTENT_AGREEMENT.md`。
