# 项目共享配置

根目录的 `config.yaml` 保存项目共同默认设置。它属于本仓库，不会修改系统、其他项目或用户主目录的配置。
模型后端、模型路径、服务地址、上下文、解码、数据准备、数据集路径、实验方案和预算都集中在这里。

每个实验入口显式继承根配置，并选择根配置内的命名方案：

```yaml
# configs/searchqa.pilot.yaml
_base_: ../config.yaml
_presets_: [searchqa.pilot]
```

`_base_` 是一个非空文件路径。相对路径以声明它的 YAML 文件所在目录为基准；可以递归继承，循环、空文件和未知字段都会报错。
`_presets_` 按列表顺序应用根配置 `presets` 中的方案，例如正式 SearchQA 入口选择 `[course, searchqa.course]`。方案字段覆盖共同默认值；未知方案或字段会报错，运行快照保存合并后的生效值，而不保存整份方案目录。
也可以直接使用根配置并按顺序传入 `--preset course --preset searchqa.course`；`show-config`、`doctor`、`validate-data`、`run` 和 `evaluate` 支持此方式。
普通 `local_path`、`split_dir`、`output_root` 等项目路径仍以仓库根目录为基准，完整绝对路径也可使用。
不带 `_base_` 的独立配置只合并 Python 中的内置 schema 默认值，不会暗中读取项目或机器上的其他配置。

配置优先级从低到高是：

1. `src/repro/config.py` 中的内置默认值和 schema。
2. 由 `_base_` 显式指定的共享 YAML，按继承顺序合并。
3. 各层选中的 `_presets_` 和该层自己的字段；命令行 `--preset` 在文件层之后按顺序应用。
4. `REPRO_MODEL_NAME`、`REPRO_BASE_URL`、`REPRO_MODEL_PATH` 环境变量的非空值。
5. 命令行 `--set section.option=value`。

CLI 与服务启动器使用同一配置加载函数；仓库 `.env` 的模型设置也参与加载，进程环境变量优先于 `.env`。
`.env` 适合凭据和本机路径覆盖；修改后应检查实际生效配置。组员也可以调整根 `config.yaml`，或参考 `configs/team.local.example.yaml` 创建不提交的本机配置。
每个实验的 `outputs/<run_id>/config.json` 保存最后实际生效的完整配置。

不启动模型即可查看实际生效值：

```bash
python -m repro show-config --config configs/searchqa.pilot.yaml
```

## 模型设置

新环境与正式实验使用 `model.backend=transformers`，`local_path` 指向完整本地 Hugging Face 模型目录。`llama_cpp` / GGUF 已弃用，仅保留历史兼容，不是新实验的安装路线。

```yaml
model:
  backend: transformers
  name: Qwen/Qwen3.5-9B
  local_path: /home/admin1/Qwen3.5-9B
  base_url: http://127.0.0.1:8000/v1
  max_context_tokens: 16384
  max_completion_tokens: 4096
  temperature: 0.6
  top_p: 1.0
  top_k: 50
  seed_policy: optimizer_retry_v1
  enable_thinking: false
```

共享 YAML 默认指向当前 Linux 的完整模型目录；其他机器修改 `model.local_path`，Windows 可用 `E:/Qwen3.5-9B`。若本机 `.env` / 进程环境中的 `REPRO_MODEL_PATH` 非空，它会优先于 YAML；修改 YAML 后仍显示旧路径时应清空或同步该覆盖项。实际值以 `show-config` 和运行的 `config.json` 为准。

模型使用 BF16 / SDPA，已在 RTX 5090 32GB 上完成双数据集正式开发与测试。当前 Transformers 服务的 BF16、SDPA 和 `cuda:0` 属于服务实现固定设置，不是 YAML 可切换选项。
`python -m repro.serving.transformers_launcher --config config.yaml` 读取同一份模型、路径、上下文和 thinking 设置；PowerShell 包装脚本 `scripts/serve_model.ps1` 也调用这一入口。
客户端在执行前核对服务报告的模型名、thinking、上下文和本地路径；更改配置不会自动下载权重。

同一轮比较中所有条件必须固定模型文件 SHA-256、tokenizer、精度、推理运行库版本、上下文、thinking 和解码设置。
真实 Transformers 运行记录所有权重分片及相关配置、tokenizer 文件的 SHA-256，并保存整体 fingerprint。
当前模型为用户提供的本地文件，上游 revision 尚未独立核验，不把另一目录的非权重 revision 当作该模型 revision。
已弃用的 `configs/local.gguf.yaml` 与 `scripts/gguf.py` 仅用于历史兼容，见 [GGUF 归档说明](gguf.md)。它不参与当前 Qwen3.5-9B 正式对照。

`max_context_tokens` 是输入加输出的总窗口。候选技能生成另有 `evolution.operator_max_completion_tokens`；两种角色共享同一模型上下文，窗口必须能同时容纳提示词和请求的输出。
加载配置会检查生成上限小于总窗口；实际输入长度仍由服务检查，超长时不静默滑窗或截断。

## 实验配置

- `preparation`：两个数据集的固定 revision、下载方式、可选本地来源、缓存目录、数据数量和种子；pilot 始终不含测试样例。
- `presets` 中的 `*.pilot`：各数据集 pilot 路径；默认使用一轮进化。
- `course` 与 `*.course`：正式实验规模、预算、数据路径及基准清单；默认禁止测试。
- `hotpotqa.provisional`：按当前 [hotpot_split.md](../hotpot_split.md) 的 300/150/1000 方案，使用独立 v2 路径、固定清单、`custom` profile 和 3000 次请求 / 1000 万输出 token 上限；三轮及模型规则与原方案一致。名称和历史状态字段保留，不表示当前说明仍有数量冲突。
- `offline` 与 `*.offline`：使用明确标记的合成数据；通过 `repro smoke` 运行，不产生研究结果。
- `local.gguf`：**已弃用**，仅保留历史 GGUF 的模型位置与解码设置。

数据准备脚本读取 `preparation` 和对应 `*.course` 路径；`scripts/make_pilot_data.py` 读取 `preparation` 的 pilot 数量以及 `*.pilot` 路径。三者支持 `--config` 指向另一份包含相同方案的配置，准备脚本的显式参数优先于 YAML。
`prepare_hotpotqa.py --stratified` 改用 `hotpotqa.provisional`，读取 `hotpotqa_train_per_level`、`hotpotqa_val_per_level`、`hotpotqa_test_count`、`hotpotqa_protocol_file` 和 `hotpotqa_ids_output`；只读本地固定 revision 缓存并验证下载收据。数据种子仍为 `preparation.seed`，不随实验种子变化。
当前说明文件统一为根目录 `hotpot_split.md`，历史清单中的旧文件名保留。准备与开发预检按内容 SHA-256 验证身份，允许单纯重命名；当前文件还修订了数量表述，因此触发下述版本检查。
正式数据清单 `manifests/data-v1/` 保存已经固定的样本和来源校验值，不能仅通过修改 YAML 换一批数据。
上游 Git 锁、Python 依赖锁、数据清单和 `.env` 凭据仍分别存放：这些文件供专门工具验证或只属于本机。

路径对应表见 [README](../README.md#模型与数据设置)，复制数据、生成 pilot 和路径核验见 [环境说明](environment.md#数据准备与路径检查)。`data.split_dir` 必须指向含三个分区 JSONL 及 `split_manifest.json` 的目录，不能指向 `preparation.raw_cache_dir`。移动同一份冻结数据时只修改对应 preset 的 `split_dir`，保留配套 `manifest`。根 `data.manifest=null` 供不同数据集 preset 绑定各自的固定清单，不表示可以关闭正式数据校验。

40 GPU 小时是课程规划上限；调用和生成 token 的硬预算由 `budget.max_calls`、`budget.max_completion_tokens` 控制。
失败选择器、训练/验证/测试边界和所有运行条件的生效值应随结果保存。

## HotpotQA 说明版本

用户指定当前 [hotpot_split.md](../hotpot_split.md) 为 HotpotQA 划分依据：训练 300、验证 150、测试 1000。当前文件已将两处旧的 200 题表述统一为 1000，实际冻结的 v2 数据原本已经是 300/150/1000。

| 说明版本 | SHA-256 |
|---|---|
| 当前 `hotpot_split.md` | `da13133aa6ff4ca8f9511516ad887ad459d1fcd7de6cebdbefa269037a8de923` |
| 历史 v2 清单记录的说明 | `47d0623b7c263ce8011a06ac0da5e53e6a1be03d9298b84ec8f53b171d16fc85` |

`datasets/protocol.py` 要求说明字节与记录完全一致。因此当前 `python scripts/prepare_hotpotqa.py --stratified` 和 `run_formal.py --provisional-hotpotqa` 的新开发预检会报 `HotpotQA teacher instruction differs from the recorded protocol`，不能直接照旧命令开始新正式实验。

本次只更新文档，不改动固定 ID、数据、来源 revision、原始校验值或已完成运行。后续需为新说明显式建立协议版本迁移，保留旧说明身份及历史结果；不能直接把冻结清单中的旧哈希改成新哈希。代码中的 `hotpotqa.provisional`、历史 `protocol_status=provisional` 和 `course_split_verified=false` 继续作为当时的记录保留。

`validate-data` 校验已有数据，B/C 的小型 pilot 使用独立的 10/5/0 工程验收数据；它们不经过这项新分层说明检查。已完成 Linux 结果的逐题审计仍有效，报告应同时交代当前划分要求与当时记录的说明版本。

## 历史协议与正式数据

v1 是两轮验收后保存的历史协议，保留在 `manifests/experiment-v1/`；发布及强制校验的旧命令已移除。组员可配置自己的模型路径和环境，运行时记录实际值。历史记录见 [归档说明](archive/README.md)，其中的旧命令不再作为操作步骤。

正式数据仍由各 course 配置的 `data.manifest` 指向 `manifests/data-v1/` 归档清单。运行前校验数据文件、ID 顺序、分区、来源 revision 和原始数据哈希；上游源码目录可以因机器而异。每次运行记录生效配置、数据清单、模型指纹和环境信息，供团队核对可比性。
