# GraphSkillEvo 课程复现

固定 Qwen3.5-9B，比较 SearchQA、HotpotQA 的无技能、初始技能和原始 GraphSkillEvo。两个数据集均已完成 seed 0、42、127 的开发与冻结测试，审计通过；B 的改进与消融尚未包含。

SearchQA 固定 100/30/50；HotpotQA 按 [hotpot_split.md](hotpot_split.md) 使用 300/150/1000，测试取自官方 distractor dev。

## Linux 安装

先准备 Conda、Git、NVIDIA 驱动及 CUDA 12.8 编译工具，详见 [系统准备](docs/environment.md#系统准备)。在仓库根目录依次执行：

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

`requirements.txt` 使用 Linux 依赖锁。先安装 PyTorch 再安装 CUDA 扩展；安装不会下载模型权重。已有独立环境可直接激活，名称不必为 `graphskill`。

## 模型与数据设置

所有入口继承根 [config.yaml](config.yaml)。先按下表准备文件；相对路径均以**仓库根目录**为基准，换机器时只在该文件调整对应路径。

| 用途 | `config.yaml` 配置项 | 设置为 |
|---|---|---|
| 完整 Qwen3.5-9B 模型 | `model.local_path` | 你本机完整模型目录的绝对路径 |
| SearchQA 正式数据 | `presets.searchqa.course.data.split_dir` | `data/processed/searchqa` |
| HotpotQA 当前正式数据 | `presets.hotpotqa.provisional.data.split_dir` | `data/processed/hotpotqa-stratified-v2` |
| 两个小实验数据 | `presets.<数据集>.pilot.data.split_dir` | `data/pilot/searchqa`、`data/pilot/hotpotqa` |
| HotpotQA 历史 pilot 来源 | `presets.hotpotqa.course.data.split_dir` | `data/processed/hotpotqa` |

修改 `model.local_path` 指向含配置、tokenizer、权重索引及全部分片的完整目录；仓库 `models/` 的早期非权重缓存不完整。**非空的 `.env` / 环境变量 `REPRO_MODEL_PATH` 会覆盖 YAML**；想使用 YAML 时将其清空。**GGUF / `local.gguf` 已弃用（deprecated）**，新实验使用 Transformers。

数据不随 Git 提供：向 A 复制完整的 `data/processed/` 和 `data/pilot/`，保留各目录的 `train.jsonl`、`val.jsonl`、`test.jsonl`、`split_manifest.json`，并与仓库 `manifests/` 配套。`preparation.raw_cache_dir=data/raw` 仅是原始数据缓存，不能用作 `split_dir`。移动目录只改相应 `split_dir`，不改固定 ID、来源或清单。

缺少 pilot 时，先取得两个 `*.course` 目录（HotpotQA 为历史 `data/processed/hotpotqa`），再运行 `python scripts/make_pilot_data.py`；它只从既有 train/val 生成 10/5/0，不重抽正式数据。详细准备步骤见 [环境说明](docs/environment.md#数据准备与路径检查)。

## 测试结果

EM/F1 为三个种子的百分比均值 ± 样本标准差。各数据集重复使用同一份固定测试题，不增加独立题数。

| 数据集 | 条件 | EM | F1 |
|---|---|---:|---:|
| SearchQA | 无技能 | 64.67 ± 1.15 | 71.88 ± 1.53 |
| SearchQA | 初始技能 | 62.67 ± 1.15 | 70.63 ± 0.99 |
| SearchQA | 原方法选定技能 | 63.33 ± 1.15 | 69.73 ± 0.70 |
| HotpotQA | 无技能 | 63.40 ± 0.82 | 78.24 ± 0.66 |
| HotpotQA | 初始技能 | 60.93 ± 0.15 | 74.93 ± 0.44 |
| HotpotQA | 原方法选定技能 | 63.43 ± 0.25 | 77.53 ± 0.23 |

本轮没有显示原方法稳定优于无技能。延迟、成本、技能演化及原始证据见 [成员 A 报告与演示稿](docs/member-a-report.md)。

## B C 独立复现验收

B、C 完成上述安装和路径设置后，各自从仓库根目录执行一次；单 GPU 上依次运行。

```bash
conda activate graphskill
python -m pip check
python -m repro show-config --config configs/searchqa.pilot.yaml
python -m repro doctor --config configs/searchqa.pilot.yaml
python -m repro validate-data --config configs/searchqa.pilot.yaml
python -m repro validate-data --config configs/hotpotqa.pilot.yaml
python scripts/run_pilots.py --set evolution.generations=2
```

pilot 使用两个真实数据集各 10 train / 5 val / 0 test，自动管理模型服务，完成八个子运行；它是流程验收，不是研究成绩。无需预先手动启动服务。

保留打印的 `outputs/pilot-check/<batch>/`。分别对两个 `runs/<dataset>/selected_val/<run>/` 执行：

```bash
python -m repro summarize --run "本次 selected_val 子运行的实际目录"
```

通过条件：批次及八个子运行完成；两次进化各两轮；两个独立验证各有 5 条唯一 val 预测；调用账本、技能与原始输出可追溯；没有测试预测、pending 或未知用量。成员填写 [验收记录](docs/reproduction-acceptance.md)，A 复核。目前 B、C 均待本人执行，不以 A 的旧运行代替。

无模型时可先运行 `python -m repro smoke --dataset searchqa`；它仅检查 synthetic 离线接线，不替代真实 pilot 验收。

## 文档入口

- [成员 A 初稿与演示](docs/member-a-report.md)：设置、实现、结果、限制、技能图及演示讲稿。
- [Linux 配置指南](docs/environment.md)：系统依赖、环境安装、模型与数据准备、服务管理及排错。
- [实验与交接](docs/reproduction-plan.md)：正式入口、当前批次、B/C 分工及固定证据。
- [配置](docs/configuration.md)、[实验协议](docs/experiment-protocol.md)、[模块职责](docs/architecture.md)。
- [课程要求](nlp-project5.pdf)、[小组分工](group_div.md)、[HotpotQA 划分说明](hotpot_split.md)。

HotpotQA 说明已修订为统一的 1000 题，历史清单仍锁定旧说明哈希；新的分层准备／正式开发预检会拒绝该哈希差异，详见 [协议版本说明](docs/configuration.md#hotpotqa-说明版本)。pilot 验收及已完成结果不受影响。数据、模型和 `outputs/` 不随 Git 提交，交接需另外复制完整文件。
