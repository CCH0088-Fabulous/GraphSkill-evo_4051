# GraphSkillEvo 课程复现

使用固定 Qwen3.5-9B，在 SearchQA 和 HotpotQA 上运行无技能、初始技能和原始 GraphSkillEvo。当前已完成成员 A 的正式验证集对照；留出测试、改进选择器和消融尚未完成。

| 数据集 | 无技能 EM / F1 | 初始技能 EM / F1 | 进化后 EM / F1 |
|---|---:|---:|---:|
| SearchQA | 50.00 / 64.33 | 66.67 / 72.52 | 66.67 / 72.52 |
| HotpotQA | 83.33 / 88.33 | 86.67 / 91.56 | 90.00 / 92.22 |

数值为百分比，各 30 条验证样例，种子 42；验证集参与技能选择，以上不是留出测试成绩。设置、成本与限制见 [正式结果报告](docs/formal-results.md)，课程要求见 [PDF](nlp-project5.pdf)，团队分工见 [group_div.md](group_div.md)。

## 安装与配置

默认使用原生 Windows、已提供的完整本地模型和 CUDA / BF16。已有 32GB 显存环境完成验证，BF16 / SDPA / `cuda:0` 为当前 Transformers 服务的实现设置。

先激活自己的 Python 环境（已验证 Python 3.13.11），在仓库根目录执行：

```powershell
python scripts/fetch_sources.py
python -m pip install -r requirements.txt
python -m pip check
```

安装使用固定依赖和上游源码，不下载模型权重。将完整模型目录写入 [config.yaml](config.yaml) 的 `model.local_path`；当前机器使用 `E:/Qwen3.5-9B`。

根配置集中管理模型、服务地址、解码、数据准备和实验方案。`configs/` 中的文件只选择方案；也可以直接使用根配置：

```powershell
python -m repro show-config --config config.yaml --preset course --preset searchqa.course
```

本机覆盖可参考 `configs/team.local.example.yaml`；凭据和临时覆盖放入不提交的 `.env`。完整优先级见 [配置说明](docs/configuration.md)，环境及服务说明见 [environment.md](docs/environment.md)。

## 准备与核验数据

```powershell
python scripts/prepare_searchqa.py
python scripts/prepare_hotpotqa.py
python scripts/make_pilot_data.py
python -m repro validate-data --config configs/searchqa.course.yaml
python -m repro validate-data --config configs/hotpotqa.course.yaml
```

| 方案 | train / val / test | 数据目录 |
|---|---|---|
| course | 100 / 30 / 50 | `data/processed/<数据集>` |
| pilot | 10 / 5 / 0 | `data/pilot/<数据集>` |
| offline | 6 / 3 / 0，synthetic | `tests/fixtures/<数据集>` |

SearchQA 使用上游固定 ID。HotpotQA 当前是官方公开 distractor 数据自行固定的子集，`course_split_verified=false`，仍需确认课程划分一致性。来源 revision、原始文件哈希和固定分区保存在数据清单中；准备脚本默认拒绝覆盖。详见 [data/README.md](data/README.md)。

## 基本检查与实验入口

无需真实模型的离线流程检查：

```powershell
python -m repro smoke --dataset searchqa
python -m repro smoke --dataset hotpotqa
```

这些结果明确标记为 synthetic，只验证接线。环境检查和单题真实模型示例：

```powershell
python -m repro doctor --config configs/searchqa.pilot.yaml
python scripts/run_example.py
```

示例使用一个训练样例和临时模型服务，不使用测试集，不代表研究成绩。小规模全流程及正式验证对照分别执行：

```powershell
python scripts/run_pilots.py
python scripts/run_formal.py
```

pilot 默认一轮；需覆盖交叉算子时可加 `--set evolution.generations=2`。正式入口在启动时保存配置快照，串行完成两数据集的三组条件及选定技能重复验证，共用一份临时模型服务，结束后释放服务。重复验证仍使用同一种子，不能当作额外种子。

后续实验种子固定为 `0、42、127`，通过 `--set experiment.seed=数字` 分批执行；数据选择种子和固定分区保持不变。目前仅 seed 42 已完成，补跑及旧结果复用条件见 [复现计划](docs/reproduction-plan.md)。

需要逐组运行时，先在一个终端启动服务：

```powershell
python -m repro.serving.transformers_launcher --config config.yaml
```

然后在另一个已激活相同环境的终端执行，例如：

```powershell
python -m repro run --config configs/searchqa.course.yaml --condition original
python -m repro evaluate --config configs/searchqa.course.yaml --condition original --skill "outputs/运行目录/upstream/best_skill.md" --split val
```

`no_skill`、`initial_skill` 和 `original` 共用模型与运行规则。进化只接收训练和验证数据；`improved` / `ablation` 预留给 B，未实现时明确报错。测试默认关闭，应在所有方法和技能选择定稿后独立评估。

## GGUF 备用路线

GGUF 统一使用一个入口，不参与当前 Qwen3.5-9B 正式成绩：

```powershell
python scripts/gguf.py --help
python scripts/gguf.py inspect
```

安装固定 llama.cpp 运行库、启动已有 GGUF 的命令及限制见 [GGUF 使用说明](docs/gguf.md)。它复用相同数据、评分、HTTP 客户端和成本账本，不保留专门的历史探针流程。

## 代码阅读与检查

核心过程为：训练执行产生失败反馈 → 选择失败案例 → 变异或交叉技能 → 验证集排名与保留 → 保存选定技能 → 单独评估。

| 阅读位置 | 职责 |
|---|---|
| `src/repro/cli.py`、`config.py` | 命令分发、配置合并与约束 |
| `experiments/` | 基线、pilot、正式批次编排 |
| `integration/` | 固定上游的适配、训练批次、失败反馈和算子桥接 |
| `inference/`、`serving/` | 模型请求、成本计量和本地模型服务 |
| `datasets/`、`evaluation/` | 数据身份与分区检查、答案 EM/F1 |
| `artifacts/`、`visualization/` | 可追溯产物、技能图与候选谱系 |

各职责和 A/B/C 边界见 [架构说明](docs/architecture.md)，实验约束见 [实验协议](docs/experiment-protocol.md)。正式运行与个人交付清单见 [复现计划](docs/reproduction-plan.md)。

```powershell
python -m pytest -q
python -m ruff check src/repro tests scripts
```

正式批次位于 `outputs/formal/20261101T135059023401Z-163af6/`，其中 `handoff.md` 索引逐题预测、调用、技能和图文件。`outputs/` 不随 Git 提交，交接需复制整个批次目录。固定数据身份保留在 `manifests/data-v1/`；旧 pilot/v1 说明归档到 `docs/archive/`，原始清单保持不变，旧发布和环境锁定命令已移除。
