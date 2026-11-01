# 原生 Windows 环境与模型服务

本项目使用原生 Windows，不安装或要求 WSL2。项目级共享配置 `config.yaml` 选择模型后端：
Hugging Face 原格式使用 CUDA PyTorch + Transformers，已有 GGUF 文件使用原生 CUDA llama.cpp。
所有 Python 包安装在项目内的 `.venv`，不会修改现有 Miniconda 环境。
基础设施安装只下载 Python 包和官方源码，不下载任何模型权重。
PDF 中的 A6000 和 40 GPU 小时仅供资源规划参考；实际预算按 RTX 5090 试跑记录确定。

本机已验证 `torch 2.14.1+cu130`、CUDA 13.0、RTX 5090（计算能力 12.0），并通过 BF16 GPU 矩阵乘。
Windows 原生推理所需的 CUDA 环境可用。用户现已提供完整的 `E:\Qwen3.5-9B`，当前共享配置选择它，
BF16 真实单题验证通过。另保留 Qwen3.8-27B GGUF 的独立 llama.cpp 路线，不复制或修改两个外部模型目录。

另已运行 `scripts/check_native_qwen.py`：在内存中随机初始化一个 218,664 参数的极小 Qwen3.5 模型，
保留 3 层 gated delta 与 1 层 full attention，以 BF16、SDPA 和 KV cache 在 RTX 5090 上实际生成 3 个 token。
该检查约用 1.64 秒，峰值 PyTorch allocated 显存约 32.86 MiB；未下载或保存任何权重。
当前使用 PyTorch reference fallback，未安装 `causal_conv1d` 和 `flash-linear-attention` 优化内核。
这验证了原生架构路径能够执行，不能据此推算真实 9B 模型的显存、吞吐或任务成绩。

## 重建开发与推理环境

在项目目录运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/bootstrap.ps1
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest
```

脚本安装项目的开发、数据准备、HTTP 服务、官方 GraphSkillEvo 依赖和固定的 `torch==2.14.1+cu130`，
生成实际环境的 `requirements/lock.windows.txt`。
重建时会先重放该锁文件，再从固定源码安装两个本地包。已有兼容 torch 时可传入 `-SkipTorch` 跳过显式 CUDA 包安装。
若需明确重算依赖版本，使用 `-RefreshLock`；正式实验开始后保存锁文件，避免中途升级。

首次使用可把 `.env.example` 复制为 `.env`，填入本地服务地址；凭据不要写入 YAML 或提交到版本库。

## 已准备的非权重模型资源

`models/Qwen3.5-9B` 已准备官方 tokenizer、聊天模板、配置、词表、预处理配置和权重索引，共 8 个文件，约 22 MB。
固定 Hugging Face revision 为 `c202236235762e1c871ad0ccb60c8ee5ba337b9a`，
`asset_manifest.json` 保存每个文件的 SHA256、字节数和本地 tokenizer 验证结果。
`model.safetensors.index.json` 只是文件索引，Qwen3.5-9B 的实际权重分片没有下载；选择该目录时 `model_ready=False`。
当前共享配置选择另一个完整目录 `E:\Qwen3.5-9B`，不从这个非权重缓存目录混取文件。

重建这些非权重资源，或完全离线重验现有文件：

```powershell
.\.venv\Scripts\python.exe scripts/prepare_model_assets.py
.\.venv\Scripts\python.exe scripts/prepare_model_assets.py --verify-local
```

下载脚本使用精确文件白名单和固定 commit，禁止任何 `.safetensors`、`.bin`、`.pt` 等权重本体。
真实本地 `AutoTokenizer` 已验证字符串形式的聊天消息和 `enable_thinking=False` 模板。
使用官方提示构造器对一个简短 synthetic 样例计数：无技能 QA 147 token、初始技能 QA 392 token、
带一个 synthetic 失败反馈的 graph-mutation 输入 912 token。这些数字用于模板接线验证，不能作为实际数据的最大提示长度。

## 后续接入本地模型

### 已有的 Qwen3.8-27B GGUF

本机文件为 `E:\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_K_M.gguf`，约 16.25 GiB，Q4_K_M 量化。
`llama.cpp` 固定为官方 Windows CUDA 13.3 的 b10901（commit `28ff09582`），
本地可执行文件、SHA256 与来源归档记录在 `tools/llama.cpp/runtime.lock.json`。
该后端直接在 GPU 执行 GGUF；不把 27B 权重还原成 BF16，也不要求通过 PyTorch 加载这个文件。

备用配置 `configs/local.gguf.yaml` 的 `model.backend` 使用 `llama_cpp`，`model.local_path` 指向该文件，
`model.name` 为服务别名 `Qwen3.8-27B-Q4_K_M`，`model.max_context_tokens` 起步为 8192。
启动方式：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/serve_gguf.ps1 -Config configs/local.gguf.yaml
```

脚本通过 Python 参数列表启动进程，避免 Windows PowerShell 5.1 改写 JSON 引号。
默认单个 slot、全部模型层交给 GPU、Flash Attention 开启、关闭上下文自动滑窗，只监听 `127.0.0.1`。
启动命令设置 `--predict`，但不同 API 路由报告的默认值可能不同；实验客户端始终显式发送并记录 `max_tokens`，
分别控制执行器和优化器的生成上限，不依赖服务的隐式默认值。
`enable_thinking=False` 同时固定聊天模板和 reasoning 设置，覆盖模型模板原本的 thinking 默认值。
启动脚本可以显式传 `-ModelPath`、`-ServerPath`、`-Port`，不会下载权重。
显存需求还包含 KV cache 和运行缓冲，应以真实加载后的采样结果决定是否增加上下文和并发。

GGUF 的基线、进化、改进和消融必须使用相同模型文件、量化类型、引擎版本和生成设置。
结果应称为 Qwen3.8-27B-Q4_K_M 条件下的方法复现，不能把量化模型与原论文其他模型的数字直接当作同设置对比。

### Hugging Face 原格式的 Qwen3.5-9B

完整本地目录为 `E:\Qwen3.5-9B`，已检查四个分片及索引中全部 775 个张量。在 PowerShell 运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/serve_model.ps1 -Config config.yaml
```

对应的 Python 启动命令是：

```powershell
.\.venv\Scripts\python.exe -m repro.serving.transformers_launcher --config config.yaml
```

启动时先检查模型配置、tokenizer、权重文件以及索引引用的每个分片。
加载明确使用 `local_files_only=True`，并开启 `HF_HUB_OFFLINE`；不存在的目录和远程仓库 ID 都会被拒绝。
没有模型文件时不会启动假服务，也不会自动下载。

服务启动器读取共享配置，当前为 BF16、CUDA 单卡、16384 个输入加输出总 token、单个串行生成请求、固定关闭 thinking。
它对外提供 `GET /health`、`GET /v1/models`、`POST /v1/chat/completions`，仅监听本机 `127.0.0.1:8000`。
任务执行与技能优化共用一个模型服务，无需加载两份权重。

可通过 PowerShell 参数 `-MaxModelLen`、`-Port`、`-ModelName` 改变服务设置。
若决定使用 thinking，启动时传入 `-EnableThinking`，并在所有比较条件中保持一致。
服务不支持流式输出、工具调用或每次请求改变 reasoning effort；不支持的参数会明确报错。
实验客户端的 `chat_template_kwargs.enable_thinking` 必须与服务启动时的固定值一致，避免实际运行模式偏离配置。
上下文不足时请求失败，原始输入不会被无声截断。

usage 按 tokenizer 的实际输入 ID 和模型新生成 ID 计数，包含生成的 EOS 等特殊 token。
2026-10-30 的初始技能单题运行使用 771 输入 / 105 输出 token，请求耗时约 3.94 秒，
整卡显存每秒采样峰值 19765 MiB（含其他应用）；临时服务已退出。
权重、配置和 tokenizer 的 SHA-256 在该次运行的 `model_identity.json`，上游 revision 未独立核验。
两数据集两轮资格验收已完成，涵盖变异与交叉；当时的公共设置以 v1 历史记录保留，见 `docs/frozen-settings-v1.md`。正式课程运行不再强制校验 v1 的本机路径或依赖清单，数据划分仍须核验。

## 就绪状态

doctor 分别报告开发包、固定源码版本、GPU、所选原生推理后端、数据、模型文件及可选服务检查。
GGUF 后端检查固定 llama-server 的文件校验值、版本和 CUDA 设备，不把 PyTorch 状态当作 GGUF 引擎的就绪条件。
模型文件可读只说明静态资源就绪；`--endpoint` 会核对实际服务的别名，以及服务报告的上下文和 thinking 设置。
`reasoning_format=none` 和 `reasoning_in_content=False` 只描述解析/展示方式，不能证明模型没有进行 thinking。
服务未显式报告 `enable_thinking` 时，doctor 将 `thinking_mode_verified` 留为 False；当前请求仍显式固定模板设置。
WSL 仅作为可选状态，不影响 Windows 基础设施是否就绪。
`infrastructure_ready` 不要求模型权重存在；`model_ready` 单独报告本地权重情况。
只有基础设施、权重及服务检查都成功时，`experiment_ready` 才为真。
synthetic fixtures 仅用于接线验证，不能被当作课程 SearchQA / HotpotQA 正式数据。

## 可选 Linux 环境

已有 Linux 机器可使用 `scripts/bootstrap_linux.sh` 创建 `.venv-linux`，再用 `scripts/serve_model.sh` 启动 vLLM。
这条路线不会安装 WSL2，也不是 Windows 路线的前置条件。Linux 锁文件只在实际安装验证后生成。
两个启动脚本都要求完整本地权重，绝不自动拉取模型。

参考：[Qwen3.5-9B 模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)、
[vLLM 安装文档](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)。
