# GraphSkillEvo 复现基础设施

面向成员 A 的运行工程：原生 Windows + 32GB 显存，当前使用 Transformers 加载用户提供的 `E:\Qwen3.5-9B`，BF16 推理。
已准备官方源码、真实数据、独立环境、项目共享配置、统一入口、评分、日志与成本计量。**双数据集两轮资格验收已通过；v1 保存为历史记录，不再强制校验其模型路径和环境，正式研究对照尚未运行。**
原有团队分工和日程完整保存在 [docs/team-plan.md](docs/team-plan.md)。
正式复现的步骤、命令和成员 A 的交付物见 [docs/reproduction-plan.md](docs/reproduction-plan.md)。
本次 pilot 的设置、输出、截断及重试问题见 [docs/pilot-results.md](docs/pilot-results.md)。
历史 v1 参数、截断/重试规则和验收证据见 [docs/frozen-settings-v1.md](docs/frozen-settings-v1.md)。

## 当前环境

| 项目 | 已验证配置 |
|---|---|
| Python | 3.13.11，项目内 `.venv` |
| GPU 显存 | 32GB |
| 当前推理引擎 | PyTorch 2.14.1+cu130 / Transformers 5.21.1，BF16 / SDPA |
| 当前模型 | `E:\Qwen3.5-9B`，4 个原格式权重分片 |
| GPU 验证 | 双数据集两轮验收使用 16K 上下文，整卡显存采样峰值约 21.8 GiB |
| 备用后端 | llama.cpp b10901 / CUDA 13.3，保留已有 Qwen3.8 GGUF 路线 |
| 官方源码 | GraphSkillEvo `722a8a913b05dc9d1b4078f0bec604cde4dc7529` |
| 项目共享配置 | 根 `config.yaml`；各数据集入口选择其中的方案 |

无需安装 WSL2。Linux/vLLM 脚本仅作为可选路线，尚未在你的 Linux 系统安装验证。
PDF 中的 A6000 与 40 GPU 小时作为资源参考；真正的调用和生成 token 预算通过配置和日志控制。

## 安装

准备 Python 3.13（已验证 3.13.11）、Git，以及支持 CUDA 和 BF16 的 32GB 显存显卡与兼容驱动。
在项目根目录用 PowerShell 创建独立环境，拉取锁定版本的上游源码，再通过 [requirements.txt](requirements.txt) 安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe scripts/fetch_sources.py
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

`requirements.txt` 复用 `requirements/lock.windows.txt` 的固定版本，并安装本项目与上游 GraphSkillEvo。
其中包含 PyTorch CUDA、Transformers、数据准备、模型服务和开发检查所需依赖；上游源码须在安装前准备好。
模型权重由使用者单独提供，当前模型路径由 `config.yaml` 中的 `model.local_path` 指定；数据准备和服务启动方式见下文。

### 组员环境配置

根 [config.yaml](config.yaml) 集中管理模型路径和服务、解码参数、数据准备、数据集位置、实验规模和预算。`presets` 中的 `course`、`acceptance`、`offline` 与数据集方案组合出不同实验；`configs/` 中的小文件只负责选择方案，旧命令仍可使用。项目相对路径以仓库根目录为准。
Git 源码版本锁、依赖锁和数据划分清单保留为独立文件：它们由各自的工具或数据校验流程使用。凭据仍放在不提交的 `.env`，不放入共享 YAML。

也可以只使用根配置并指定方案，例如查看 SearchQA 正式设置：

```powershell
.\.venv\Scripts\python.exe -m repro show-config --config config.yaml --preset course --preset searchqa.course
```

组员可以修改根 `config.yaml` 中的本机路径，或复制 [组员配置模板](configs/team.local.example.yaml) 并修改其中的模型、数据和输出路径，以进行 SearchQA 试跑：

```powershell
Copy-Item configs/team.local.example.yaml configs/team.local.yaml
.\.venv\Scripts\python.exe -m repro show-config --config configs/team.local.yaml
.\.venv\Scripts\python.exe -m repro doctor --config configs/team.local.yaml
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/serve_model.ps1 -Config configs/team.local.yaml
```

本机 `configs/team.local.yaml` 不会提交。另一个终端可使用相同配置运行 `python -m repro run --config configs/team.local.yaml --condition original`；数据须先按下方步骤准备。也可将 [.env.example](.env.example) 复制为 `.env` 并设置 `REPRO_MODEL_PATH`；非空环境变量和 `.env` 中的模型覆盖项优先于 YAML，请用 `show-config` 检查实际生效值。
正式 course 配置不再强制校验 v1 的模型路径或安装包清单；每次运行仍记录实际配置、模型文件指纹、依赖和 GPU 采样，便于比较不同机器的结果。不同显卡需要满足所选模型、BF16 和上下文长度的运行要求。

## 现在可以运行

在本项目根目录执行；使用显式 Python 路径，无需激活环境：

```powershell
.\.venv\Scripts\python.exe scripts/run_example.py
```

该入口要求服务端口空闲，加载本地权重，用初始技能执行固定训练集第一题并自动释放临时服务。
本次样例答案 `Bluebeard` 与标准答案一致，请求约 3.94 秒，输入 771 / 输出 105 token；没有访问测试样例。
记录位于 `outputs/qwen35-example/20261030T013343516812Z-8e44c3/`，包含原始输出、配置、权重与 tokenizer SHA-256、调用和显存采样。
这是单题工程验证，不能用来推断完整数据集成绩、长上下文或进化算子的速度。

```powershell
.\.venv\Scripts\python.exe -m repro doctor --config configs/searchqa.pilot.yaml
.\.venv\Scripts\python.exe -m repro validate-data --config configs/searchqa.course.yaml
.\.venv\Scripts\python.exe -m repro validate-data --config configs/hotpotqa.course.yaml
.\.venv\Scripts\python.exe -m repro smoke --dataset searchqa
.\.venv\Scripts\python.exe -m repro smoke --dataset hotpotqa
```

`doctor` 将基础设施、模型文件、服务就绪状态分别报告；未启动本地服务时 `experiment_ready` 为 False。
`smoke` 使用明确标记 synthetic 的小数据与固定协议响应，走完整的官方初始化、进化、选择和保存路径；其 EM/F1、token 估算与延迟不能用于研究比较。

无权重的真实 GPU 架构检查：

```powershell
.\.venv\Scripts\python.exe scripts/check_native_qwen.py
```

该检查仅在内存构造 218,664 参数随机小模型，覆盖 gated delta、attention 和 cache；不下载或保存权重，也不代表真实 9B 模型的速度、显存或效果。

## 真实数据与固定划分

| 配置 | 位置 | train / val / test |
|---|---|---|
| 两个 `*.course.yaml` | `data/processed/searchqa`、`data/processed/hotpotqa` | 各 100 / 30 / 50 |
| 两个 `*.pilot.yaml` | `data/pilot/searchqa`、`data/pilot/hotpotqa` | 各 10 / 5 / 0 |
| 两个 `offline.*.yaml` | `tests/fixtures/` | synthetic，各 6 / 3 / 0 |

SearchQA 严格使用 GraphSkillEvo 官方 ID 清单。
HotpotQA 使用你给出的 [hotpotqa/hotpot](https://github.com/hotpotqa/hotpot) 对应的官方 Hugging Face 数据镜像，配置为 distractor；官方 train 中抽取互不重叠的训练与验证子集，官方 dev 中抽取本地留出测试。
**这是自行固定的公开数据子集，清单明确记录 `course_split_verified=false`，不能声称与尚未提供的课程特定划分一致。**
原始 Parquet 已缓存到 `data/raw`，约 684 MB；来源 revision、原文件 SHA-256、样本 ID 和标准化文件 SHA-256 都已记录。
每个数据目录中的 `split_manifest.json` 是该目录的清单。试跑数据只从既定训练/验证 ID 中派生，不包含测试样本。

需要重建数据时：

```powershell
.\.venv\Scripts\python.exe scripts/prepare_searchqa.py
.\.venv\Scripts\python.exe scripts/prepare_hotpotqa.py
.\.venv\Scripts\python.exe scripts/make_pilot_data.py
```

准备脚本从根配置的 `preparation` 和数据集 course 方案读取缓存目录、固定 revision、数量与输出路径，默认拒绝覆盖已固定的数据；命令行参数可以显式覆盖。网络异常会重试，缓存可断点续传；不会因为一次下载失败重新抽 ID。
数据格式和完整准备方式见 [data/README.md](data/README.md)。
`data/processed/` 不随仓库提交；正式配置将重建后的数据与 `manifests/data-v1/` 中的历史数据清单比较，核对 ID、分区、来源 revision 和已记录的哈希，同时允许上游源码的本机目录不同。

## 共享配置与真实运行

根 [config.yaml](config.yaml) 管理模型、16K 总上下文、4096 输出上限、温度 0.6、采样与重试种子规则。
两个 `*.course.yaml` 引用对应的历史数据清单，正式规模为三轮、每轮两个新候选，预算为 600 次请求 / 200 万输出 token。
正式运行前用 `repro validate-data --config configs/searchqa.course.yaml` 和 HotpotQA 配置分别核验数据；v1 的环境和模型路径不再作为运行门槛。
例如 `configs/searchqa.pilot.yaml` 仅继承根配置并覆盖数据集路径。
优先级为内置默认值 → `_base_` → 实验 YAML → 环境变量 → `--set`。
`.env` 可以为试跑或正式运行提供本机模型路径；完整优先级见 [docs/configuration.md](docs/configuration.md)。

只检查实际配置，不运行模型：

```powershell
.\.venv\Scripts\python.exe -m repro show-config --config configs/searchqa.pilot.yaml
```

启动当前 Qwen3.5-9B 模型：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/serve_model.ps1 -Config config.yaml
```

服务只读本地模型、监听本机，BF16、单卡串行，输入加输出总窗口为 16384 token，显式关闭 thinking。
启动器读取共享配置；执行和技能优化共用一份模型，每次请求有独立生成上限。
所有基线、改进和消融固定权重、tokenizer、推理库版本及解码设置；每次真实运行保存模型文件 SHA-256。
当前文件的上游下载 revision 未独立核验，在正式报告中应补记来源 revision 或明确披露这一点。

另一个终端运行：

```powershell
.\.venv\Scripts\python.exe -m repro doctor --config configs/searchqa.pilot.yaml --endpoint
.\.venv\Scripts\python.exe -m repro run --config configs/searchqa.pilot.yaml --condition no_skill
.\.venv\Scripts\python.exe -m repro run --config configs/searchqa.pilot.yaml --condition initial_skill
.\.venv\Scripts\python.exe -m repro run --config configs/searchqa.pilot.yaml --condition original
```

### 模型文件与备用后端

当前完整模型在 `E:\Qwen3.5-9B`。仓库内另一个 `models/Qwen3.5-9B` 目录仅含早期准备的非权重资源，不能作为当前完整模型使用。
这两个目录独立保留；推理只从共享配置指定的完整目录读取，不混用 tokenizer 或配置。
早期非权重资源仍可离线重验：

```powershell
.\.venv\Scripts\python.exe scripts/prepare_model_assets.py --verify-local
```

若需重看已有 Qwen3.8 GGUF 的工程验证，可使用独立配置启动备用后端：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/serve_gguf.ps1 -Config configs/local.gguf.yaml
```

正式 Qwen3.5-9B 对照统一使用当前 Transformers 配置。

HotpotQA 替换配置名即可。先根据试跑消耗调整预算，再使用 `*.course.yaml`。
独立评估已保存技能：

```powershell
.\.venv\Scripts\python.exe -m repro evaluate --config configs/searchqa.course.yaml --skill "outputs/运行ID/upstream/best_skill.md" --split val
```

正式测试默认关闭。方法、提示词、模型与解码设置冻结后，使用 `--split test --set experiment.allow_test=true` 执行留出评估；进化循环始终无法读取测试分区。
`improved` / `ablation` 选择器预留给 B，未注册实现时明确拒绝，不能回退为原方法。

## 产物与方法适配

每次实验创建独立 `outputs/<run_id>/`：配置、源码版本与文件哈希、环境版本、数据清单、模型服务信息、逐题预测、原始提示词和响应、调用与 token、GPU 采样、技能文件和 SHA、进化历史、Mermaid 内部流程图与候选谱系图。
失败和重试也计量；后端缺失 usage 时明确标记，不以零 token 冒充完整测量。

```powershell
.\.venv\Scripts\python.exe -m repro summarize --run "outputs/运行ID"
.\.venv\Scripts\python.exe -m repro render-skill --skill "outputs/运行ID/upstream/best_skill.md" --output "outputs/selected-skill.mmd"
```

官方源码保持原样。项目适配层将每轮新候选数与保留种群大小分开，按配置跨轮轮转变异/交叉，并约束无效图生成的重试；原方法与 B 的改进共享这些设置。
SearchQA 保留原提示词及 6000 字符上下文截断，HotpotQA 保留所有提供的段落；具体差异保存于 `integration_adaptations.json`。
HotpotQA 使用项目中性初始技能 `skills/hotpotqa/initial.md`，不是官方发布的进化结果。

## 结构与重建

`src/repro/` 按职责拆分为 `datasets`、`inference`、`serving`、`integration`、`experiments`、`selectors`、`evaluation`、`artifacts` 和 `visualization`。
`cli.py` 只做命令解析和分发。边界及 A/B/C 所有权见 [docs/architecture.md](docs/architecture.md)。

新 Windows 环境按上方“安装”小节使用 `requirements.txt` 准备依赖。
若使用可选的 GGUF 后端，再安装固定版本的 llama.cpp 运行库：

```powershell
.\.venv\Scripts\python.exe scripts/install_llama_cpp.py
```

源码固定于 `third_party/*.lock.json`，Windows 依赖固定于 `requirements/lock.windows.txt`。
环境与本地服务细节见 [docs/environment.md](docs/environment.md)。测试命令：

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ruff check src/repro tests scripts
.\.venv\Scripts\python.exe -m pytest -q
```
