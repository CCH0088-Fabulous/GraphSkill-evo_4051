# GraphSkillEvo 课程复现

使用固定 Qwen3.5-9B，在 SearchQA 和 HotpotQA 上运行无技能、初始技能和原始 GraphSkillEvo。SearchQA 已在 Linux conda `jev` 完成三种子开发与留出测试：12/12开发子运行、9/9测试子运行成功。B的改进选择器和消融尚未包含。

SearchQA 保持PDF规定的 **100 train / 30 val / 50 test**，实验种子为0、42、127。下表为固定50题测试集上的三种子均值 ± 样本标准差，EM/F1单位为百分比，延迟为平均每题秒数：

| 条件 | 测试EM | 测试F1 | 每题延迟(s) |
|---|---:|---:|---:|
| 无技能 | 64.67 ± 1.15 | 71.88 ± 1.53 | 1.82 |
| 初始技能 | 62.67 ± 1.15 | 70.63 ± 0.99 | 4.18 |
| 原方法选定技能 | 63.33 ± 1.15 | 69.73 ± 0.70 | 3.55 |

原方法选定技能相对无技能平均EM -1.33、F1 -2.15个百分点；相对初始技能EM +0.67、F1 -0.91。
只有seed0选中第1轮变异技能，seed42和127均保留初始技能；本轮未显示出优于无技能的留出测试表现。
同一50题在三个种子间重复使用，不是150道独立测试题。完整结果见[审计与汇总](outputs/reports/searchqa-linux/20261106T074952Z/audit-results.json)和[逐种子数据](outputs/reports/searchqa-linux/20261106T074952Z/per-seed.csv)。
开发及测试合计1716次调用，约70分钟墙钟；逐题指标、预算、固定身份及原始文件校验均通过，测试3条截断输出按原规则计零分。

HotpotQA 固定为 **300 train / 150 val / 1000 test**，训练每难度100、验证每难度50。以下表格保留历史Windows开发批次的结果：相同150条验证题上 seed 0、42、127 的百分比均值 ± 样本标准差；验证集参与技能选择，不能当作留出测试成绩。后续Linux开发及已完成测试的状态见 `MEMORY.md`，不与此表混合。

| 条件 | EM | F1 |
|---|---:|---:|
| 无技能 | 69.78 ± 1.02 | 80.77 ± 0.94 |
| 初始技能 | 68.00 ± 1.33 | 78.88 ± 1.73 |
| 进化选定技能 | 70.89 ± 1.02 | 82.36 ± 1.39 |

该历史开发批次的进化相对初始技能平均 EM +2.89、F1 +3.47 个百分点；相对无技能仅 +1.11、+1.59。各种子开发成绩、成本、核验和限制见 [HotpotQA v2 结果](docs/hotpotqa-v2-results.md)。

SearchQA 的旧 seed42 开发集结果为无技能50.00/64.33、初始及进化后66.67/72.52（EM/F1）；旧 HotpotQA 未分层结果另行保留，不混入新比较，见 [旧批次报告](docs/formal-results.md)。课程要求见 [PDF](nlp-project5.pdf)，团队分工见 [group_div.md](group_div.md)。

[教授说明](hotpot_split.md)已明确允许从官方 distractor 数据自建分层划分。测试数量1000按用户明确指定执行；整体规模与原 PDF 限额的关系仍待确认。用户已授权对成员 A 的三种固定条件开展留出测试；它使用独立批次，测试成绩不混入上面的验证集表。

## 安装与配置

当前运行环境为原生 Linux、conda `jev`（Python 3.12、RTX 5090），使用已有完整本地模型和 CUDA / BF16。BF16 / SDPA / `cuda:0` 为当前 Transformers 服务的实现设置。

已有环境直接激活并核验；Linux依赖快照见 `requirements/lock.linux.txt`：

```bash
conda activate jev
python -m pip check
python -m repro doctor --config configs/searchqa.course.yaml
```

原生Windows仍保留支持。Windows环境安装入口为 `requirements.txt`（Python 3.13.11已验证），在仓库根目录执行：

```powershell
python scripts/fetch_sources.py
python -m pip install -r requirements.txt
python -m pip check
```

安装使用固定依赖和上游源码，不下载模型权重。完整模型目录可由 [config.yaml](config.yaml) 的 `model.local_path` 或不提交的 `.env` 中 `REPRO_MODEL_PATH` 指定；当前Linux使用 `/home/admin1/Qwen3.5-9B`，共享YAML保留Windows路径供其他机器覆盖。

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

HotpotQA 新分层方案使用以下显式入口，读取已下载并校验过的完整 Parquet 缓存：

```powershell
python scripts/prepare_hotpotqa.py --stratified
python -m repro validate-data --config config.yaml --preset course --preset hotpotqa.provisional
```

当前数据位于 `data/processed/hotpotqa-stratified-v2/`；三份原始 ID 列表及数据参考清单位于 `manifests/hotpotqa-stratified-v2/`。数据抽样 seed 固定为 42；实验 seed 为 `0、42、127`。v2 保持 v1 的训练／验证 ID，测试集从 200 扩为 1000，包含原来的 200 条。原有目录不覆盖。

| 方案 | train / val / test | 数据目录 |
|---|---|---|
| course | 100 / 30 / 50 | `data/processed/<数据集>` |
| pilot | 10 / 5 / 0 | `data/pilot/<数据集>` |
| offline | 6 / 3 / 0，synthetic | `tests/fixtures/<数据集>` |

SearchQA 使用上游固定 ID。旧 HotpotQA 100/30/50 未按难度分层，保留为历史划分；新方案的数据来源和分层方法已有教授说明，数量口径仍待确认。来源 revision、原始文件哈希和固定分区保存在数据清单中；准备脚本默认拒绝覆盖。详见 [data/README.md](data/README.md)。

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

后续实验种子固定为 `0、42、127`；数据选择种子和固定分区保持不变。旧方案仅 seed 42 已完成，新的 HotpotQA 划分需要分别运行全部三个种子。SearchQA 的旧结果复用条件见 [复现计划](docs/reproduction-plan.md)。

新 HotpotQA 暂定开发集实验使用同一入口的显式模式：

```powershell
python scripts/run_formal.py --provisional-hotpotqa --seeds 0 42 127
```

此模式只比较新 HotpotQA 的无技能、初始技能、原方法和选定技能重复验证；每个种子独立进化，共用本批次的一份临时模型服务。采用三轮、每轮两个新候选及原解码、重试规则；因验证集扩大五倍，每条件调用/token 预算上限同步扩大五倍至 3000 次 / 1000 万输出 token。规模暂定状态写入批次，不宣称已获教授确认。默认不带该选项的命令仍运行旧的双数据集 100/30/50 方案。

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

成员 A 的三种子技能已按验证集自动选定。留出测试入口读取完成批次的配置和技能快照，不读取正在编辑的根配置，也不执行进化：

```powershell
python scripts/run_formal.py --test-from-batch outputs/provisional-hotpotqa/20261102T050827834273Z-ae48d3
```

SearchQA 可以在当前环境独立运行三个种子的课程规模开发实验，不会顺带重跑 HotpotQA：

```bash
conda activate jev
python scripts/run_formal.py --dataset searchqa --seeds 0 42 127 --set experiment.output_root=outputs/linux-transformers
```

每个种子运行无技能、初始技能、三轮原方法，以及选定技能的重复验证，共12个开发子运行，始终关闭测试推理。
SearchQA 保持固定100/30/50划分、每轮最多两个新候选及每条件600次调用/200万输出token预算。
全部开发子运行完成后，用打印的本次批次目录执行：

```bash
python scripts/run_formal.py --test-from-batch outputs/linux-transformers/formal/<本次SearchQA开发批次>
```

入口从完成的单数据集三种子计划自动识别 SearchQA，冻结各自选定技能与固定数据，在新的 `heldout-searchqa/` 中运行九组、每组50题。
测试模式禁止配置、数据集、种子或参数覆盖，不能根据测试表现重新选择技能。相同入口继续支持原有HotpotQA分层批次。
当前Linux执行环境是conda `jev`；使用已有本地权重，启动和运行不下载模型。

两种数据集的入口均冻结初始技能、三个选定技能、模型指纹、数据文件及全部固定测试ID（SearchQA 50题、分层HotpotQA 1000题），再串行评估三个种子 × 三条件，共九组。测试模式禁止额外配置、种子或参数覆盖；只在局部派生配置中开启测试，根配置仍默认关闭。测试结果、逐题失败及样例不得反馈给进化、选择器设计、提示词或调参。B 的改进和消融不包含在本次 A 测试中。

测试批次被中断后可续跑：

```bash
python scripts/run_formal.py --resume-heldout outputs/.../heldout-hotpotqa/<批次目录>
```

续跑先核对该批次冻结的技能、数据、全部固定测试 ID、开发证据和模型指纹，保留已完成的子运行，只重跑其余子运行；SearchQA 使用对应的 `heldout-searchqa/` 批次路径。被中断的子运行保留为部分结果，重跑写入新目录，不覆盖，其未知用量保持未知。旧的 `summary.json` 会改名为 `summary.attempt-N.json`。该入口不接受其他参数。留出测试不再校验项目 Python 源码的逐字节哈希，改以冻结的技能、数据和模型身份为准；续跑前请确认没有仍在运行的同批次进程。

## GGUF 备用路线

GGUF 统一使用一个入口，不参与当前 Qwen3.5-9B 正式成绩：

```powershell
python scripts/gguf.py --help
python scripts/gguf.py inspect
```

安装固定 llama.cpp 运行库、启动已有 GGUF 的命令及限制见 [GGUF 使用说明](docs/gguf.md)。它复用相同数据、评分、HTTP 客户端和成本账本，不保留专门的历史探针流程。

## 代码阅读与检查

技能进化谱系图可从任一进化运行导出（每个 `original` 运行也已自带 `graphs/lineage.mmd`）。本命令额外标出最终保留和被选中的候选，输出文件不得已存在：

```bash
python -m repro render-lineage --run outputs/.../runs/seed-0/hotpotqa/original/<运行目录> --output outputs/reports/lineage/seed-0.mmd
```

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

当前 HotpotQA v2 批次位于 `outputs/provisional-hotpotqa/20261102T050827834273Z-ae48d3/`，汇总与 `handoff.md` 位于 `outputs/reports/hotpotqa-v2/20261102T105531089060Z-76f4bd/`。`outputs/` 不随 Git 提交，交接需复制完整批次和报告。当前 HotpotQA 数据身份位于 `manifests/hotpotqa-stratified-v2/`；旧小规模清单、报告和中断产物继续保留，旧 pilot/v1 说明归档到 `docs/archive/`。
