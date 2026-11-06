# 环境与模型服务

当前已验证环境是原生 Linux conda `jev`：Python 3.12.0、RTX 5090 32GB、PyTorch 2.11.0+cu128、Transformers 5.17.0，已安装 `fla` 与 `causal_conv1d`。本轮完整实验见 [成员 A 报告](member-a-report.md)。保留原生 Windows 备用路线，不需要 WSL。

## 当前 Linux 环境

从仓库根目录执行：

```bash
conda activate jev
python -c "import sys; print(sys.executable); print(sys.version)"
python -m pip check
python -m repro show-config --config configs/searchqa.pilot.yaml
python -m repro doctor --config configs/searchqa.pilot.yaml
python -m repro validate-data --config configs/searchqa.pilot.yaml
python -m repro validate-data --config configs/hotpotqa.pilot.yaml
```

当前解释器为 `/home/admin1/miniconda3/envs/jev/bin/python`，`config.yaml` 的 `model.local_path` 默认指向完整模型 `/home/admin1/Qwen3.5-9B`。换机器时修改该项，Windows 可用 `E:/Qwen3.5-9B`。也可在本机 `.env` 设置 `REPRO_MODEL_PATH`；非空 `.env` / 进程环境变量会覆盖 YAML，若要使用 YAML 则需清空旧覆盖项。修改后检查 `show-config`，不要提交 `.env`。

[requirements/lock.linux.txt](../requirements/lock.linux.txt) 是已完成实验的依赖版本快照。当前环境无需重装；`requirements.txt` 引用 Windows 锁，不用于 Linux。新 Linux 机器需先建立独立 Python 3.12 环境，准备与 GPU、驱动匹配的 CUDA PyTorch，以及兼容的 `fla`、`causal_conv1d`；这些包可能需要 CUDA Toolkit 和编译工具，不能将版本快照视作跨硬件的一键安装保证。依赖准备后安装项目与固定源码：

```bash
python scripts/fetch_sources.py
python -m pip install -e ".[data,dev,server]" -e ./third_party/GraphSkillEvo
python -m pip check
```

记录新机器的 Python、依赖、驱动和 GPU，并按 [独立复现验收](reproduction-acceptance.md) 运行两个真实 pilot。硬件或依赖不同应如实记录，不冒称与现有结果环境完全一致。

## 数据准备与路径检查

模型和数据路径统一由根 `config.yaml` 管理，`configs/*.yaml` 选择对应 preset。模型、数据、缓存、输出的相对路径均以仓库根目录为基准；`_base_` 引用配置文件时才相对该 YAML 所在目录。目录对应表见 [README](../README.md#模型与数据设置)。

新机器从 A 取得完整 `data/processed/`、`data/pilot/`，复制到仓库相应位置。每个数据目录须包含 `train.jsonl`、`val.jsonl`、`test.jsonl`、`split_manifest.json`；pilot 的 test 文件为空。正式目录中的清单须与仓库 `manifests/` 的对应固定清单一致，不能只复制 JSONL、重新抽样或手改校验值。

当前正式 HotpotQA 是 `data/processed/hotpotqa-stratified-v2`，对应 `hotpotqa.provisional`；`hotpotqa.course` 的 `data/processed/hotpotqa` 是历史 100/30/50，保留为已有工程 pilot 的来源，不是当前 300/150/1000 的正式实验。要放在仓库外，可将相应 preset 的 `data.split_dir` 改为完整绝对路径，文件身份及 `data.manifest` 保持不变。

如果尚无 pilot，但已取得两个原始 `*.course` 目录及清单，可运行：

```bash
python scripts/make_pilot_data.py
```

该脚本从 `presets.searchqa.course` 与 `presets.hotpotqa.course` 的 train/val 生成两个 10/5/0 pilot，输出到 `presets.*.pilot` 的目录。已有 pilot 会验证其父清单和数量，拒绝覆盖不匹配内容；它不从新分层 HotpotQA 自动替换旧 pilot 来源。

准备完后，从仓库根目录核对路径并校验数据，全程不调用模型：

```bash
python -m repro show-config --config configs/searchqa.pilot.yaml
python -m repro show-config --config configs/hotpotqa.pilot.yaml
python -m repro validate-data --config configs/searchqa.pilot.yaml
python -m repro validate-data --config configs/hotpotqa.pilot.yaml
python -m repro validate-data --config configs/searchqa.course.yaml
python -m repro validate-data --config config.yaml --preset course --preset hotpotqa.provisional
```

报文件缺失时，检查生效的 `split_dir` 和复制内容；报清单不符时，重新取得对应冻结文件，不关闭清单校验。`preparation.raw_cache_dir` 是原始下载缓存，`searchqa_manifest_dir` 是固定上游 ID 目录，均不是标准化的 `split_dir`。原始缓存不能代替已准备好的四个分区文件。

数据准备脚本读取同一配置中的固定 revision、种子和路径；新机器优先复制现有冻结数据。若需从源重建，SearchQA 使用 `scripts/prepare_searchqa.py`，但当前 HotpotQA 分层准备仍受 [说明版本检查](configuration.md#hotpotqa-说明版本) 限制，不能用普通 `prepare_hotpotqa.py` 的旧划分冒充当前协议。

## 原生 Windows 备用安装

激活独立的 Python 环境后执行：

```powershell
python scripts/fetch_sources.py
python -m pip install -r requirements.txt
python -m pip check
python -m repro doctor --config configs/searchqa.pilot.yaml
```

Windows 历史环境使用 Python 3.13.11。安装读取固定依赖与上游源码锁，不重写锁文件；模型仍由使用者提供。

## 模型与服务

项目脚本不下载模型权重。完整模型目录必须包含配置、tokenizer、权重及索引引用的全部分片；仓库 `models/` 中的早期非权重缓存不能替代它。模型来源 revision 尚未独立核验，运行会记录所有本地模型文件的指纹。

`run_example.py`、`run_pilots.py` 和 `run_formal.py` 自动管理一个临时模型服务；使用这些入口时无需预先启动服务，端口已被占用会拒绝运行。正常退出关闭本批次拥有的服务；强制杀进程或关机后应检查遗留进程与未完成账本，不能假定清理已经完成。

只有逐条件使用 `repro run` / `repro evaluate` 时，才需另开终端手动启动：

```bash
python -m repro.serving.transformers_launcher --config configs/searchqa.pilot.yaml
python -m repro doctor --config configs/searchqa.pilot.yaml --endpoint
```

第二条命令在服务启动后从另一个终端执行。Windows 也可用 `scripts/serve_model.ps1 -Config configs/searchqa.pilot.yaml`，默认采用当前环境的 `python`。

服务只读本地文件并禁止 Hub 下载，固定 BF16、SDPA、逻辑 `cuda:0` 和串行生成；配置控制模型、地址、总上下文和解码。提供 `/health`、`/v1/models`、`/v1/chat/completions`，客户端核对模型名、路径、上下文和 thinking；输入超限报错，不静默截断。

`doctor` 不加 `--endpoint` 时用于检查解释器、依赖、源码、GPU、数据和模型文件。只有基础设施、模型与服务均通过时 `experiment_ready` 才为真；未启动服务时不能只凭该字段认定环境安装失败。

GGUF / `local.gguf` 已弃用，仅保留历史兼容，见 [GGUF 归档说明](gguf.md)；新环境无需准备 GGUF 模型或运行库。HotpotQA 新正式运行还需处理 [说明版本校验](configuration.md#hotpotqa-说明版本)，与 Linux 依赖是否可用是两个问题。
