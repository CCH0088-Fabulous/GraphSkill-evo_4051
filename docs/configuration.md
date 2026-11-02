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

```powershell
.\.venv\Scripts\python.exe -m repro show-config --config configs/searchqa.pilot.yaml
```

## 模型设置

`model.backend` 支持 `transformers` 和 `llama_cpp`。前者的 `local_path` 是完整本地 Hugging Face 模型目录；后者是完整本地 `.gguf` 文件路径。
GGUF 服务与 Transformers 服务都通过同一 HTTP client 和成本账本接入研究代码，模型后端不进入技能优化算法。

```yaml
model:
  backend: transformers
  name: Qwen/Qwen3.5-9B
  local_path: E:/Qwen3.5-9B
  base_url: http://127.0.0.1:8000/v1
  max_context_tokens: 16384
  max_completion_tokens: 4096
  temperature: 0.6
  top_p: 1.0
  top_k: 50
  seed_policy: optimizer_retry_v1
  enable_thinking: false
```

根配置选择用户提供的完整 Qwen3.5-9B 目录，使用 BF16 / SDPA，已通过 32GB 显存环境的双数据集两轮资格验收。当前 Transformers 服务的 BF16、SDPA 和 `cuda:0` 属于服务实现固定设置，不是 YAML 可切换选项。
`python -m repro.serving.transformers_launcher --config config.yaml` 读取同一份模型、路径、上下文和 thinking 设置；PowerShell 包装脚本 `scripts/serve_model.ps1` 也调用这一入口。
客户端在执行前核对服务报告的模型名、thinking、上下文和本地路径；更改配置不会自动下载权重。

同一轮比较中所有条件必须固定模型文件 SHA-256、tokenizer、精度、推理运行库版本、上下文、thinking 和解码设置。
真实 Transformers 运行记录所有权重分片及相关配置、tokenizer 文件的 SHA-256，并保存整体 fingerprint。
当前模型为用户提供的本地文件，上游 revision 尚未独立核验，不把另一目录的非权重 revision 当作该模型 revision。
GGUF 备用路线使用 `configs/local.gguf.yaml`，入口统一为 `scripts/gguf.py`，安装与启动见 [GGUF 说明](gguf.md)。它不参与当前 Qwen3.5-9B 正式对照。

`max_context_tokens` 是输入加输出的总窗口。候选技能生成另有 `evolution.operator_max_completion_tokens`；两种角色共享同一模型上下文，窗口必须能同时容纳提示词和请求的输出。
加载配置会检查生成上限小于总窗口；实际输入长度仍由服务检查，超长时不静默滑窗或截断。

## 实验配置

- `preparation`：两个数据集的固定 revision、下载方式、可选本地来源、缓存目录、数据数量和种子；pilot 始终不含测试样例。
- `presets` 中的 `*.pilot`：各数据集 pilot 路径；默认使用一轮进化。
- `course` 与 `*.course`：正式实验规模、预算、数据路径及基准清单；默认禁止测试。
- `hotpotqa.provisional`：按用户指定的 300/150/1000 方案，使用独立 v2 路径及参考清单、`custom` profile 和 3000 次请求 / 1000 万输出 token 上限；三轮及模型规则与原方案一致。用户已授权恢复开发集实验，整体规模与 PDF 限额的关系仍待确认。
- `offline` 与 `*.offline`：使用明确标记的合成数据；通过 `repro smoke` 运行，不产生研究结果。
- `local.gguf`：历史备用后端的模型位置与解码设置。

数据准备脚本读取 `preparation` 和对应 `*.course` 路径；`scripts/make_pilot_data.py` 读取 `preparation` 的 pilot 数量以及 `*.pilot` 路径。三者支持 `--config` 指向另一份包含相同方案的配置，准备脚本的显式参数优先于 YAML。
`prepare_hotpotqa.py --stratified` 改用 `hotpotqa.provisional`，读取 `hotpotqa_train_per_level`、`hotpotqa_val_per_level`、`hotpotqa_test_count`、`hotpotqa_protocol_file` 和 `hotpotqa_ids_output`；只读本地固定 revision 缓存并验证下载收据。数据种子仍为 `preparation.seed`，不随实验种子变化。
正式数据清单 `manifests/data-v1/` 保存已经固定的样本和来源校验值，不能仅通过修改 YAML 换一批数据。
上游 Git 锁、Python 依赖锁、数据清单和 `.env` 凭据仍分别存放：这些文件供专门工具验证或只属于本机。

40 GPU 小时是规划参考；调用和生成 token 的硬预算由 `budget.max_calls`、`budget.max_completion_tokens` 控制。
失败选择器、训练/验证/测试边界和所有运行条件的生效值应随结果保存。

## 历史协议与正式数据

v1 是两轮验收后保存的历史协议，保留在 `manifests/experiment-v1/`；发布及强制校验的旧命令已移除。组员可配置自己的模型路径和环境，运行时记录实际值。历史记录见 [归档说明](archive/README.md)，其中的旧命令不再作为操作步骤。

正式数据仍由各 course 配置的 `data.manifest` 指向 `manifests/data-v1/` 归档清单。运行前校验数据文件、ID 顺序、分区、来源 revision 和原始数据哈希；上游源码目录可以因机器而异。每次运行记录生效配置、数据清单、模型指纹和环境信息，供团队核对可比性。
