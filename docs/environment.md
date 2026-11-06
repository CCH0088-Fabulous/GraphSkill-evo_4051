# Linux 环境配置

按顺序完成系统工具、Python 依赖、模型路径和固定数据配置。以下命令从仓库根目录执行；`graphskill` 是示例 Conda 环境名，可自行更换。

## 系统准备

准备 Linux x86_64、Conda、Git，以及支持 CUDA/BF16 的 NVIDIA GPU。建议为默认 16,384 token 上下文准备 32GB 显存。

安装支持 CUDA 12.8 的 NVIDIA 驱动、CUDA Toolkit 12.8 和兼容的 GCC/G++。Ubuntu 可先安装基础工具：

```bash
sudo apt-get update
sudo apt-get install -y git build-essential
nvidia-smi
nvcc --version
gcc --version
```

驱动和 Toolkit 按 [NVIDIA CUDA 12.8 Linux 安装指南](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-linux/index.html)安装。`nvidia-smi` 检查驱动，`nvcc --version` 检查实际编译工具；pip 的 CUDA 运行库不能代替用于编译扩展的 `nvcc`。

若 Toolkit 安装在 `/usr/local/cuda-12.8`，设置：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
```

其他位置使用实际安装目录；确认 `nvcc --version` 显示 12.8，再进行下一步。

## 创建环境并安装依赖

```bash
conda create -n graphskill python=3.12 pip -y
conda activate graphskill
python -m pip install --upgrade pip
python -m pip install -c requirements/lock.linux.txt setuptools wheel packaging ninja
python -m pip install -c requirements/lock.linux.txt torch --index-url https://download.pytorch.org/whl/cu128
python scripts/fetch_sources.py
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
```

已有独立 Python 3.12 环境可直接激活并从 pip 步骤开始。`requirements.txt` 读取 [Linux 依赖锁](../requirements/lock.linux.txt)，安装项目、固定上游源码、数据工具、Transformers 服务和开发依赖；`fetch_sources.py` 只恢复锁定源码，不下载模型权重。

先安装锁定的 PyTorch，再安装 CUDA 扩展；`--no-build-isolation` 使扩展使用该环境的 PyTorch 和编译工具。CUDA wheel 来源参照 [PyTorch 安装命令](https://pytorch.org/get-started/previous-versions/#v2-11-0)，扩展构建在配置阶段需要导入 PyTorch，见 [causal-conv1d 构建脚本](https://github.com/Dao-AILab/causal-conv1d/blob/main/setup.py)。

安装后检查 GPU 和两个加速包：

```bash
python - <<'CHECK'
import torch
import fla
import causal_conv1d
assert torch.cuda.is_available(), "CUDA unavailable: check driver and PyTorch wheel"
assert torch.cuda.is_bf16_supported(), "This GPU must support BF16"
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
CHECK
```

`fla` 由 `flash-linear-attention` / `fla-core` 提供，不要安装同名的其他包；包和后端说明见 [FLA 安装指南](https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md)。

## 配置本地模型

自行准备完整 Qwen3.5-9B 模型目录，在根 `config.yaml` 修改这两项，保留文件中其他设置：

```yaml
model:
  backend: transformers
  local_path: /path/to/Qwen3.5-9B  # 替换为你机器上的完整绝对路径
```

目录须含配置、tokenizer、权重索引及索引引用的全部分片。仓库 `models/` 中的非权重缓存不能代替完整模型，安装与服务脚本不自动下载权重。GGUF / `local.gguf` 已弃用，新环境无需准备其模型或运行库。

路径统一由根配置管理，`configs/*.yaml` 选择 preset。普通相对路径以仓库根目录为基准；只有 `_base_` 引用相对其 YAML 所在目录。

非空的 `.env` / 进程环境变量 `REPRO_MODEL_PATH` 会覆盖 YAML；要使用 YAML，清空这两处旧覆盖项。用以下命令确认实际值，不要提交 `.env`：

```bash
python -m repro show-config --config configs/searchqa.pilot.yaml
```

## 数据准备与路径检查

按 [README 路径表](../README.md#模型与数据设置)，从 A 取得完整 `data/processed/` 和 `data/pilot/`，复制到配置指定位置。每个数据目录须含 `train.jsonl`、`val.jsonl`、`test.jsonl`、`split_manifest.json`；pilot 的 test 文件为空。正式清单必须与仓库 `manifests/` 配套，不重新抽样或手改校验值。

正式 HotpotQA 使用 `hotpotqa.provisional` 的 `data/processed/hotpotqa-stratified-v2`（300/150/1000）。`hotpotqa.course` 的 `data/processed/hotpotqa` 是历史 100/30/50，仅保留为工程 pilot 来源。迁移到其他位置时，只改对应 preset 的 `data.split_dir`，保持固定数据和 `data.manifest` 不变。

如果缺少 pilot，先取得两个 `*.course` 目录及清单，再运行：

```bash
python scripts/make_pilot_data.py
```

它从 `searchqa.course`、`hotpotqa.course` 的 train/val 生成 10/5/0，输出到 `*.pilot` 指定目录；已有内容会校验，不覆盖不匹配数据。它不从新的分层 HotpotQA 自动替换旧 pilot 来源。

检查模型和两份 pilot：

```bash
python -m repro doctor --config configs/searchqa.pilot.yaml
python -m repro validate-data --config configs/searchqa.pilot.yaml
python -m repro validate-data --config configs/hotpotqa.pilot.yaml
```

正式数据可额外执行：

```bash
python -m repro validate-data --config configs/searchqa.course.yaml
python -m repro validate-data --config config.yaml --preset course --preset hotpotqa.provisional
```

`preparation.raw_cache_dir` 是原始下载缓存，`searchqa_manifest_dir` 是上游固定 ID 目录，二者不能用作 `split_dir`。从源重建 SearchQA 可使用 `scripts/prepare_searchqa.py`；HotpotQA 新分层准备仍受 [说明版本检查](configuration.md#hotpotqa-说明版本) 限制，优先复制冻结数据，不能以旧划分替代新协议。

## 运行与服务管理

配置通过后，按 [README 的独立验收](../README.md#b-c-独立复现验收) 运行双数据集两轮 pilot。`run_example.py`、`run_pilots.py`、`run_formal.py` 自动启动和关闭自己的临时服务，无需另开模型服务；同一 GPU 上依次运行。

只有逐条件使用 `repro run` / `repro evaluate` 时，才在一个终端手动启动，并在另一个终端检查：

```bash
python -m repro.serving.transformers_launcher --config configs/searchqa.pilot.yaml
```

```bash
python -m repro doctor --config configs/searchqa.pilot.yaml --endpoint
```

服务采用 BF16、SDPA、逻辑 `cuda:0`，模型及解码设置来自配置。未启动服务时，`doctor` 的 `experiment_ready=false` 不代表依赖安装失败；先看具体依赖、GPU、模型和数据检查结果。强制终止后需检查残留服务与未完成账本。

## 常见问题

| 现象 | 处理方法 |
|---|---|
| `No module named torch` 或构建扩展失败 | 激活正确环境，先安装锁定 PyTorch，再用 `--no-build-isolation` 安装 requirements |
| `nvcc` 缺失或 CUDA 版本不匹配 | 安装 Toolkit 12.8，检查 `CUDA_HOME`、`PATH` 和 `nvcc --version` |
| `torch.cuda.is_available()` 为 false | 检查 GPU 是否可见、NVIDIA 驱动及是否安装了 cu128 wheel |
| 模型仍指向旧目录 | 查看 `show-config`，清空或同步 `.env` / 进程环境中的 `REPRO_MODEL_PATH` |
| 数据缺失或清单不符 | 核对 preset 的 `split_dir`，复制完整冻结目录，不关闭校验 |
| 服务端口被占用 | 结束自己遗留的服务，再使用自动管理服务的批次入口 |
