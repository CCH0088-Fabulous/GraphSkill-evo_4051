# 成员 A 复现与交付

成员 A 负责原方法和公共运行框架；B 负责改进选择器及消融，C 负责评分、失败分析与报告整合。任务原文见根目录 `nlp-project5.pdf`，分工与日程见 `group_div.md`。

## 当前进度

SearchQA已在Linux conda `jev` 完成seed0、42、127的新开发与冻结测试：12/12开发子运行、9/9测试子运行成功，保持PDF的100/30/50及三轮上限。测试EM/F1均值为无技能64.67/71.88、初始技能62.67/70.63、原方法选定技能63.33/69.73；原方法未优于无技能。证据和完整性检查见[本次审计](../outputs/reports/searchqa-linux/20261106T074952Z/audit-results.json)，逐种子表和技能图位于同目录。B的改进/消融、C的失败分类和机制分析仍由对应成员完成。

旧双数据集批次的无技能、初始技能、三轮原始 GraphSkillEvo 和选中技能重复验证已完成，共八个子运行。结果和限制见 [formal-results.md](formal-results.md)。这些是种子42的历史验证集成绩；该开发批次自身不执行测试。

以上 HotpotQA 属于旧的未分层 100/30/50 划分。新方案为 **300/150/1000**，已经完成 seed 0、42、127 的12个子运行，原方法各三轮；结果见 [HotpotQA v2 汇总](hotpotqa-v2-results.md)。选定技能验证 EM/F1 均值为70.89/82.36，测试未评估，整体规模相对PDF的限额差异继续披露。新结果与旧划分及中断批次独立保存。

## 1. 准备环境与数据

按 README 在已激活的 Python 环境中安装 `requirements.txt`。模型由使用者提供，根 `config.yaml` 管理路径、解码和实验方案。

```powershell
python -m repro doctor --config configs/searchqa.pilot.yaml
python -m repro show-config --config configs/searchqa.course.yaml
python -m repro validate-data --config configs/searchqa.course.yaml
python -m repro validate-data --config configs/hotpotqa.course.yaml
```

SearchQA 保持 100 train / 30 val / 50 test 和上游固定 ID。HotpotQA 新方案从官方 train 按难度各取 100 训练、50 验证，从官方 distractor dev 取 1000 测试；全部不重叠且固定数据 seed 42。用户已明确指定测试数量1000；原PDF仍有限额100/30/50，因此整体新规模继续标为暂定，不能按成绩换 ID。模型下载 revision 未独立核验，须保留文件指纹并披露来源限制。

```powershell
python scripts/prepare_hotpotqa.py --stratified
python -m repro validate-data --config config.yaml --preset course --preset hotpotqa.provisional
```

三份原始 ID 和参考清单在 `manifests/hotpotqa-stratified-v2/`，标准化数据在独立的 v2 目录，准备入口重复执行时只验证已有内容。v1 的200测试题、训练/验证数据和已中断运行完整保留；v2 的训练/验证内容不变，测试扩大为包含旧200题的1000题。

## 2. 基本验证

`repro smoke` 只做 synthetic 离线接线检查。`scripts/run_example.py` 只执行一个固定训练样例；`scripts/run_pilots.py` 使用 10 train / 5 val / 0 test 完成小规模流程，默认一轮。

```powershell
python -m repro smoke --dataset searchqa
python scripts/run_example.py
python scripts/run_pilots.py --set evolution.generations=2
```

两轮 pilot 可覆盖变异与交叉；这类结果不当作正式研究成绩。确认所有请求、失败和重试均有计量，检查输出截断和服务一致性。

## 3. 正式验证集对照

```powershell
python scripts/run_formal.py
```

统一入口保存一次配置快照，顺序运行两个数据集的 `no_skill`、`initial_skill`、`original`，再加载选定技能做 `selected_val`，结束后关闭自己创建的服务。两次重复验证使用同一种子，不是多种子复验。

当前课程方案为三轮、种群二、每轮两个新候选、训练评估 15 题、最多三条失败反馈；每条件预算为 600 次请求 / 200 万输出 token。到达预算后停止并保留失败产物。

后续实验种子已确定为 **`[0, 42, 127]`**。`experiment.seed` 是单次批次的整数种子，当前默认仍为 42；根配置注释记录完整种子方案。保持 `preparation.seed`、`preparation.pilot_seed`、固定 ID 和分区不变，各方法使用相同的三个实验种子。

当前 Linux conda `jev` 的 SearchQA 独立入口如下。它在同一环境重新运行三个种子的全部 A 开发条件，不调用 HotpotQA，也不复用旧 Windows 模型输出：

```bash
conda activate jev
python scripts/run_formal.py --dataset searchqa --seeds 0 42 127 --set experiment.output_root=outputs/linux-transformers
```

固定100/30/50、三轮及原course预算不变。每种子包含三项条件和选定技能重复验证，共12个开发子运行。
本次开发全部完成后，将打印的批次路径传给 `--test-from-batch`；入口冻结每种子的最终技能，单独完成九组50题测试。
三种子是团队实验计划，不是PDF额外要求；同一测试子集跨种子复用，不能当成150道独立测试题。

已有 seed 42 的完整批次。确认模型文件、数据、有效实验参数和方法语义一致后，可保留它并补跑 0、127；若改变公共协议，则在新协议下重跑全部三个种子，不混合不同设置的结果。补跑命令为：

```powershell
python scripts/run_formal.py --set experiment.seed=0
python scripts/run_formal.py --set experiment.seed=127
```

以上为待执行计划，不表示已启动或完成。每个新种子都包含无技能、初始技能、原方法进化及选定技能重复验证，不能只重跑进化而借用另一种子的基线。C 汇总逐种子结果、均值、样本标准差和配对差值。

上述旧方案复用仅适用于未改变的数据。新 HotpotQA 开发集三种子已完成；以下命令用于重新建立独立批次，运行状态见根 `MEMORY.md`：

```powershell
python scripts/run_formal.py --provisional-hotpotqa --seeds 0 42 127
```

每种子、每条件请求/输出预算上限为 3000 / 1000 万，其余三轮、每轮两个候选、训练评估15题、最多三条失败反馈及模型规则不变。扩大上限是为了容纳150题验证集，实际用量单独报告；B 后续比较需使用同一范围预算。此入口始终禁止测试推理，原 PDF 小规模 `course` 的边界仍保留。

## 4. 向 B/C 交接

已完成批次：`outputs/formal/20261101T135059023401Z-163af6/`。

新 HotpotQA v2 批次为 `outputs/provisional-hotpotqa/20261102T050827834273Z-ae48d3/`，重新汇总在 `outputs/reports/hotpotqa-v2/20261102T105531089060Z-76f4bd/`。后者包含 `results.json`、逐种子 CSV、相对路径 `handoff.md`、校验清单和可重算脚本；交接时需同时复制原始批次和报告目录。

- `handoff.md`：逐题预测、摘要和技能的相对链接，复制整批目录后仍可使用。
- `audit-results.json`、`results.json`：指标、成本和边界检查。
- `selection/`：自动选定技能的文件身份；各进化子运行的 `graphs/` 和批次中的 `selected-*.mmd`：技能图及谱系。
- 各子运行：配置、数据清单、模型指纹、原始提示与响应、调用和 GPU 采样。

C 基于选定技能的完整验证预测建立全部 EM=0 的失败清单、分类和诊断；B 据此设计训练失败选择器。每轮技能更新仍只使用训练失败材料。B 的改进和消融使用公共接口与匹配预算，不能用原选择器冒充未实现的方法。

## 5. 方法定稿后评估测试集

所有方法、模型、解码、预算和最终技能选择确定后，才显式开启独立测试。测试失败不再反馈给方法或提示词设计。

用户现已明确授权评估 A 的冻结条件。HotpotQA v2 使用以下入口，自动冻结开发批次中已经选好的全部三个种子技能，并比较无技能、初始技能和原方法，共九组，每组1000题：

```powershell
python scripts/run_formal.py --test-from-batch outputs/provisional-hotpotqa/20261102T050827834273Z-ae48d3
```

对于上述新Linux SearchQA批次，使用同一冻结入口，自动识别100/30/50方案：

```bash
python scripts/run_formal.py --test-from-batch outputs/linux-transformers/formal/<本次SearchQA开发批次>
```

SearchQA测试输出位于独立 `outputs/linux-transformers/heldout-searchqa/`，共三个种子 × 三条件 × 50题 = 450次题目评估。
该入口在任何测试推理前验证开发记录、选定技能、固定数据和模型身份；测试不执行进化，也不接受额外种子、数据集或参数覆盖。

测试模式不用 `--config`、`--set`、`--seeds` 覆盖；配置来自已完成开发批次，不能根据测试结果另选种子、技能或推理参数。输出为独立 `outputs/heldout-hotpotqa/` 批次，其中先保存冻结清单、技能与数据副本，再启动服务。B 的改进／消融仍由 B 完成，不能读取本测试的样例和失败来继续设计方法。

先启动相同配置的模型服务，再执行：

```powershell
python -m repro run --config configs/searchqa.course.yaml --condition no_skill --split test --set experiment.allow_test=true
python -m repro run --config configs/searchqa.course.yaml --condition initial_skill --split test --set experiment.allow_test=true
python -m repro evaluate --config configs/searchqa.course.yaml --condition original --skill "outputs/原方法运行目录/upstream/best_skill.md" --split test --set experiment.allow_test=true
```

HotpotQA 和 B 的改进/消融同样独立评估各自已选技能。C 汇总答案 EM/F1、延迟、实际调用和 token；进化轨迹中的重复训练预测不能混入最终测试均值。

## A 的验收

另外两位组员应能按 README 重跑一个小实验，并解释从训练反馈到最终评估的流程。任意基线数字须能追溯到配置、固定样例、技能和逐题输出。A 提供系统实现、复现设置、技能演化说明与工程限制；完整报告和演示由团队共同完成。
