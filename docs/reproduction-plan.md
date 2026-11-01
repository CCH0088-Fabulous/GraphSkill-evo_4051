# 成员 A 的复现执行清单

## 已完成与尚待完成

原生 Windows、RTX 5090、固定上游源码、真实数据划分、统一配置、计量客户端及结果存储已经就绪。
2026-10-30 已将根配置切换到用户提供的 `E:\Qwen3.5-9B`，以 Transformers / BF16 / SDPA、8K 总上下文、关闭 thinking 运行真实 SearchQA 训练题。
第一条固定训练题答案为 `Bluebeard`，输出匹配；1 次调用，771 输入 / 105 输出 token，请求 3.94 秒，整卡显存采样峰值约 19.3 GiB。
产物在 `outputs/qwen35-example/20261030T013343516812Z-8e44c3/`，包含权重和 tokenizer SHA-256、实际请求、原始输出和配置。
随后完成了两个数据集的无技能、初始技能、一轮进化及选定技能的独立验证；详细结果和待处理的截断、重试问题见 `docs/pilot-results.md`。
后续两轮资格验收已通过，当时的公共设置保留为 v1 历史记录，具体规则见 `docs/frozen-settings-v1.md`；正式运行不再强制校验 v1。2026-11-01 已完成成员 A 的正式验证集对照，八个子运行全部成功，结果见 `docs/formal-results.md`；留出测试尚未执行。

该项目优化的是技能文本及图结构；模型通过推理执行技能、生成候选，权重固定，不进行梯度训练或微调。
下述实验都使用同一份 Qwen3.5-9B，执行器与技能优化器共用一个模型服务。

## 1. 固定可比较的设置

与 B、C 确认根 `config.yaml` 的模型、tokenizer、BF16、thinking、温度、上下文、种子、执行与算子输出上限、重试和预算。
当前 v1 为温度 0.6、thinking 关闭、总上下文 16384，问答与技能算子输出上限均为 4096，并启用可复现的结构重试种子规则。
先通过 pilot 检查实际长提示、截断、超时和资源消耗，再与组员约定正式设置；当前默认总窗口为 16K，不代表所有正式数据的算子提示都已验收。
当前模型文件已生成完整本地 SHA-256 指纹；补记下载来源的 revision，若无法确认，在报告中明确来源 revision 未核验。

```powershell
.\.venv\Scripts\python.exe -m repro show-config --config configs/searchqa.course.yaml
.\.venv\Scripts\python.exe -m repro validate-data --config configs/searchqa.course.yaml
.\.venv\Scripts\python.exe -m repro validate-data --config configs/hotpotqa.course.yaml
```

两个正式数据目录均为 train / val / test = 100 / 30 / 50。SearchQA 使用官方 GraphSkillEvo ID 清单。
HotpotQA 目前为官方公开 distractor 数据自行固定的子集，`course_split_verified=false`；与 C 确认是否有课程专用清单，或在报告中披露当前划分。
不得根据成绩更换 ID。若取得课程指定划分，应另建数据目录与清单，保留现有记录。

## 2. 双数据集小规模完整验收

2026-10-30 已使用温度 0.6、16K 上下文、360 秒超时完成一轮双数据集 pilot，具体设置和限制见 `docs/pilot-results.md`。
该早期批次使用临时覆盖；后续公共参数记录在历史 v1 中。下方普通命令读取当前根配置，历史精确设置以各运行的 `config.json` 为准。
如要用该批已完成的配置重跑两个数据集，可使用：

```powershell
.\.venv\Scripts\python.exe -u scripts/run_pilots.py --set model.temperature=0.6 --set model.max_context_tokens=16384 --set model.timeout_s=360
```

下列逐个运行示例读取当前根配置；服务和每条实验命令须使用一致模型设置。两轮资格验收可直接运行 `scripts/run_pilots.py --profile acceptance`：

第一个终端启动服务，后续命令放在第二个终端：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/serve_model.ps1 -Config config.yaml
```

```powershell
.\.venv\Scripts\python.exe -m repro doctor --config configs/searchqa.pilot.yaml --endpoint
.\.venv\Scripts\python.exe -m repro run --config configs/searchqa.pilot.yaml --condition no_skill
.\.venv\Scripts\python.exe -m repro run --config configs/searchqa.pilot.yaml --condition initial_skill
.\.venv\Scripts\python.exe -m repro run --config configs/searchqa.pilot.yaml --condition original
```

将配置名换成 `configs/hotpotqa.pilot.yaml`，同样运行三组。
pilot 每数据集 10 train / 5 val / 0 test，默认一轮；进化每次训练评估使用其中 5 题。
完整轨迹应包括初始化候选、训练执行与失败反馈、修改技能、验证评分、保留与最终技能保存。
检查所有请求均有计量、无未知 usage、无未说明的截断、无模型服务错误；记录调用、token、耗时和显存。
默认四类算子跨轮轮转，一轮只覆盖两个变异算子；预算允许时在开发阶段用两轮验证交叉路径：

```powershell
.\.venv\Scripts\python.exe -m repro run --config configs/searchqa.pilot.yaml --condition original --set evolution.generations=2
```

不把单题或 pilot 的分数写成正式研究结果。完整算子流程的消耗用于确定正式预算。

## 3. 运行 A 负责的三类正式条件

冻结设置后，将 pilot 配置换为 course 配置。A 负责两个数据集各三组，共六个条件运行：

| 条件 | 目的 | CLI condition |
|---|---|---|
| 无技能 | 测量模型本身的问答能力 | `no_skill` |
| 初始技能 | 测量固定初始技能的作用 | `initial_skill` |
| 原始 GraphSkillEvo | 用原失败选择器进化技能 | `original` |

```powershell
.\.venv\Scripts\python.exe -m repro run --config configs/searchqa.course.yaml --condition no_skill
.\.venv\Scripts\python.exe -m repro run --config configs/searchqa.course.yaml --condition initial_skill
.\.venv\Scripts\python.exe -m repro run --config configs/searchqa.course.yaml --condition original
```

HotpotQA 同样执行三组。course 默认三轮、每轮两个新候选，保留种群二；训练评估每次抽取既定训练集中的 15 题。
当前默认调用上限 600、生成 token 上限 2000000，包含失败与重试；达到预算会停止，不能把未完成运行当成正常完成。
需要多个种子时，先确定所有条件共用的种子清单，再用 `--set experiment.seed=数字` 独立运行，保留每次结果。

## 4. 将完整开发结果交给 B、C

独立加载原方法保存的最终技能，在完整固定开发集上评估：

```powershell
.\.venv\Scripts\python.exe -m repro evaluate --config configs/searchqa.course.yaml --condition original --skill "outputs/原方法运行ID/upstream/best_skill.md" --split val
```

HotpotQA 同样处理。C 从这次独立评估的 `predictions.jsonl` 建立全部 EM=0 的开发失败清单，核验 EM/F1、延迟和成本。
B 据此提出改进假设，实际技能更新仍只使用训练失败池。
B 实现并运行改进失败选择器和一项聚焦消融；接口位于 `src/repro/selectors/`，使用 `select(rows, *, k, seed)`。
`improved` / `ablation` 尚未实现，当前会明确报错，不能用原选择器代替后宣称已完成改进。
A 负责保证公共运行入口和日志可用，B 负责研究改动，C 负责评分和分析。
原方法、改进与消融匹配模型请求和生成 token 预算，并报告实际消耗，不能仅以相同轮数宣称公平。

## 5. 方法冻结后运行留出测试

技能选择、方法与所有配置冻结后，才打开 test；测试输出不再反馈给选择器或技能设计。
无技能和初始技能通过 `run` 独立测试，进化结果通过 `evaluate` 加载已选定技能，不在测试集上继续进化。

```powershell
.\.venv\Scripts\python.exe -m repro run --config configs/searchqa.course.yaml --condition no_skill --split test --set experiment.allow_test=true
.\.venv\Scripts\python.exe -m repro run --config configs/searchqa.course.yaml --condition initial_skill --split test --set experiment.allow_test=true
.\.venv\Scripts\python.exe -m repro evaluate --config configs/searchqa.course.yaml --condition original --skill "outputs/原方法运行ID/upstream/best_skill.md" --split test --set experiment.allow_test=true
```

HotpotQA 同样执行；B 的改进与消融也独立加载各自冻结技能。C 汇总两个数据集、团队五种条件的 EM/F1、延迟、调用、token 与资源记录。
初始化、修改、失败和重试计入优化成本；最终测试推理成本单列。失败记录和中断产物保留，不能混为已完成结果。

## 6. 完成 A 的交付物

- 可复现的环境、固定模型与数据清单、完整配置及运行命令。
- 两个数据集的三类基线，包含最终开发和留出测试逐题结果。
- 每轮技能、父子关系、验证分数、保留决定及选定的最终技能。
- 技能内部流程图与候选谱系图，两者均可追溯到原始产物。
- 调用、token、失败重试、耗时和 GPU 采样，以及报告中的复现设置、公共系统实现和限制。

```powershell
.\.venv\Scripts\python.exe -m repro summarize --run "outputs/最终评测运行ID"
.\.venv\Scripts\python.exe -m repro render-skill --skill "outputs/原方法运行ID/upstream/best_skill.md" --output "outputs/selected-skill.mmd"
```

验收标准：B、C 能按说明重跑一个小实验；任意结果能追溯到模型指纹、数据 ID、技能、请求和逐题输出。
成员 A 的六个正式验证条件及两次选定技能重复验证已完成，公共 v1 作为历史记录保留。下一步向 B/C 交付 `outputs/formal/20261101T135059023401Z-163af6/handoff.json` 所列产物，进行开发失败分析和改进方法实现；数据与测试边界继续固定。
