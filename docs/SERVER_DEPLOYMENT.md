# 私人仓库与服务器部署

## 1. 同步范围

推送当前开发分支 `CWAH-TF`，不要默认仅上传 `main`（旧基础版本）。
收到私人 GitHub 仓库地址后，在本地确认 remote，再推送；服务器示例：

```bash
git clone --branch CWAH-TF <PRIVATE_REPOSITORY_URL> TRACE
cd TRACE
git rev-parse HEAD
```

密钥通过服务器本地文件或 Secret 管理器配置，不放入 Git、命令参数或实验日志。
完整实验结果另行备份，不能认为 `git push` 已备份被忽略的数据。

## 2. CWAH Python 环境

建议 Linux x86-64，先从两个并行实验开始，按实际 Unity 内存占用调整并发。

```bash
conda create -n CWAH python=3.8 -y
conda activate CWAH
pip install -r CWAH/CWAH-MultiPlayer/requirements.txt
pip check
```

新增直接依赖采用已安装环境的版本；原 requirements 中未固定的依赖仍可能随平台变化。
这不是完整锁文件，尚未在全新服务器重装验证。PyTorch 如无需 CUDA，可先按其官方
安装说明安装适合服务器的 CPU wheel，再安装其余依赖。

## 3. 外部 API、模拟器与任务数据

VirtualHome API 保持以下目录及 revision：

```bash
git clone --branch wah https://github.com/xavierpuigf/virtualhome.git CWAH/virtualhome
git -C CWAH/virtualhome checkout 6773be207680d47985091b90568e2a12fd3d321f
```

Unity v2.3.0 来自 CWAH 上游 README 指定的 Google Drive 文件：
`https://drive.google.com/uc?id=1L79SxE07Jt-8-_uCvNnkwz5Kf6AjtaGp`。
下载到独立临时目录解压、检查后，将完整 `executable/` 放到 `CWAH/executable/`。
也可从工作站安全传输现有完整目录，不能只复制 `.x86_64` 启动文件；需要
`UnityPlayer.so` 和对应 Data 目录。保留已有模拟器，不覆盖其他版本。

```bash
chmod +x CWAH/executable/linux_exec.v2.3.0.x86_64
mkdir -p CWAH/dataset-candidates/wah
curl -fL https://raw.githubusercontent.com/xavierpuigf/watch_and_help/21f5ab7416111d22a2ebc8d0d2427c799d01398a/dataset/test_env_set_help.pik \
  -o CWAH/dataset-candidates/wah/test_env_set_help.pik
sha256sum CWAH/dataset-candidates/wah/test_env_set_help.pik
```

仅在目标文件不存在时下载，不覆盖已有任务文件。期望 SHA-256：
`11992fddda2ef5c9e98d9d19fc7bc425c68f6cfac16dfdbb6832e909a93de1d7`。
Pickle 能执行代码，只加载可信且已核对来源的文件。
Episode 44 / Environment 3 已有运行记录；Environment 6 的候选场景存在初始化问题，
不能假设全部 100 个 Episode 都兼容此模拟器。

## 4. 配置 API 和实验

```bash
cp CWAH/experiment_runner/.env.jd.example CWAH/experiment_runner/.env.jd
cp CWAH/experiment_runner/.env.ds.example CWAH/experiment_runner/.env.ds
```

只复制尚不存在的文件，然后编辑所选供应商的密钥；不要覆盖已有凭据。
JD 使用 Responses，DS 使用 Chat Completions。JSON 的 `alice_model` 和
`bob_model` 优先于 env 默认值：选择 JD 时必须使用该服务支持的模型，不能沿用 DS 模型名。

```bash
source CWAH/experiment_runner/env.sh jd
cp CWAH/experiment_runner/configs/continuous/prepare_food_ep44_peer23_continuous_off.json.example CWAH/experiment_runner/configs/continuous/prepare_food_ep44_peer23_continuous_off.json
cp CWAH/experiment_runner/configs/continuous/prepare_food_ep44_peer23_continuous_on.json.example CWAH/experiment_runner/configs/continuous/prepare_food_ep44_peer23_continuous_on.json
python CWAH/experiment_runner/run_many.py --config-dir CWAH/experiment_runner/configs/continuous --check
python CWAH/experiment_runner/run_many.py --config-dir CWAH/experiment_runner/configs/continuous --max-parallel 2
```

先确认目录中的全部实际 `.json`、模型和步数；扫描不递归，`.json.example` 不运行。
端口自动分配；密钥加载必须与运行命令在同一 shell。`--check` 不调用 LLM，
也不启动 Unity，因此成功不等于完整运行验证。服务器缺少图形运行库或显示环境时，
应先解决 Unity 日志中的依赖错误，再进行付费实验。

## 5. 检查、结果与限制

```bash
PYTHONPATH=CWAH/CWAH-MultiPlayer python -m unittest discover -s CWAH/CWAH-MultiPlayer/tests
```

结果位于 `CWAH/experiments/baseline/` 和 `CWAH/experiments/trace/`，
各次运行独立保存配置、日志、汇总和轨迹。正常处理中断可保存已执行部分，
但断电或 SIGKILL 不保证最终汇总完成，建议同时备份增量日志。
原始 Prompt 与交互属于实验数据，应只向可信人员共享。

TRACE 仍是探索版本：非法动作回退、模型默认化输出和承诺语义归因等不能仅凭
单元测试判定完全解决。迁移后先小步数验证，再扩大预算；模型任务评分不等于环境健康状态。

## 6. EntCollabBench

这是独立工作流，不需要为了 CWAH 启动其 Docker 服务。按
[上游说明](../EntCollabBench/README.md) 安装 Docker Compose、准备外部评测任务，
并在 `EntCollabBench/agent/.env` 填入本地配置（可参考 `.env.example`）。
服务启动入口是 Arena 的 `docker-compose-mcp.yml` 和 agent 的 `docker-compose.yml`。
迁移前检查 MCP 端点、容器宿主地址、挂载目录及防火墙；不要将 Redis/MCP 端口直接暴露公网。
镜像、volume、外部任务集以及本地运行数据不通过 Git 自动迁移。
