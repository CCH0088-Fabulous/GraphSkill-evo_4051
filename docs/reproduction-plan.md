# 成员 A 复现与交付

成员 A 负责原方法和公共运行框架；B 负责改进选择器及消融，C 负责评分、失败分析与报告整合。任务原文见根目录 `nlp-project5.pdf`，分工与日程见 `group_div.md`。

## 当前进度

两数据集的无技能、初始技能、三轮原始 GraphSkillEvo 和选中技能重复验证已完成，共八个子运行。结果和限制见 [formal-results.md](formal-results.md)。这些是种子 42 的验证集成绩；测试集未向模型执行。

## 1. 准备环境与数据

按 README 在已激活的 Python 环境中安装 `requirements.txt`。模型由使用者提供，根 `config.yaml` 管理路径、解码和实验方案。

```powershell
python -m repro doctor --config configs/searchqa.pilot.yaml
python -m repro show-config --config configs/searchqa.course.yaml
python -m repro validate-data --config configs/searchqa.course.yaml
python -m repro validate-data --config configs/hotpotqa.course.yaml
```

正式数据为每套 100 train / 30 val / 50 test。SearchQA 使用上游固定 ID；HotpotQA 为公开 distractor 自选子集，`course_split_verified=false`，仍需由 C 确认课程一致性。不能根据成绩替换 ID。模型下载 revision 未独立核验，须保留文件指纹并披露来源限制。

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

已有 seed 42 的完整批次。确认模型文件、数据、有效实验参数和方法语义一致后，可保留它并补跑 0、127；若改变公共协议，则在新协议下重跑全部三个种子，不混合不同设置的结果。补跑命令为：

```powershell
python scripts/run_formal.py --set experiment.seed=0
python scripts/run_formal.py --set experiment.seed=127
```

以上为待执行计划，不表示已启动或完成。每个新种子都包含无技能、初始技能、原方法进化及选定技能重复验证，不能只重跑进化而借用另一种子的基线。C 汇总逐种子结果、均值、样本标准差和配对差值。

## 4. 向 B/C 交接

已完成批次：`outputs/formal/20261101T135059023401Z-163af6/`。

- `handoff.md`：逐题预测、摘要和技能的相对链接，复制整批目录后仍可使用。
- `audit-results.json`、`results.json`：指标、成本和边界检查。
- `selection/`：自动选定技能的文件身份；各进化子运行的 `graphs/` 和批次中的 `selected-*.mmd`：技能图及谱系。
- 各子运行：配置、数据清单、模型指纹、原始提示与响应、调用和 GPU 采样。

C 基于选定技能的完整验证预测建立全部 EM=0 的失败清单、分类和诊断；B 据此设计训练失败选择器。每轮技能更新仍只使用训练失败材料。B 的改进和消融使用公共接口与匹配预算，不能用原选择器冒充未实现的方法。

## 5. 方法定稿后评估测试集

所有方法、模型、解码、预算和最终技能选择确定后，才显式开启独立测试。测试失败不再反馈给方法或提示词设计。

先启动相同配置的模型服务，再执行：

```powershell
python -m repro run --config configs/searchqa.course.yaml --condition no_skill --split test --set experiment.allow_test=true
python -m repro run --config configs/searchqa.course.yaml --condition initial_skill --split test --set experiment.allow_test=true
python -m repro evaluate --config configs/searchqa.course.yaml --condition original --skill "outputs/原方法运行目录/upstream/best_skill.md" --split test --set experiment.allow_test=true
```

HotpotQA 和 B 的改进/消融同样独立评估各自已选技能。C 汇总答案 EM/F1、延迟、实际调用和 token；进化轨迹中的重复训练预测不能混入最终测试均值。

## A 的验收

另外两位组员应能按 README 重跑一个小实验，并解释从训练反馈到最终评估的流程。任意基线数字须能追溯到配置、固定样例、技能和逐题输出。A 提供系统实现、复现设置、技能演化说明与工程限制；完整报告和演示由团队共同完成。
