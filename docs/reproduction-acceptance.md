# B C 独立复现验收记录

依据 [分工中的 A 验收标准](../group_div.md)，B 和 C 各自按 README 完成一个小实验，A 核对结果能否追溯。两位成员必须独立执行并填写各自的运行路径；已有正式成绩、A 运行的 pilot 或离线 smoke 都不替代这项验收。

## 执行步骤

先按 [Linux 配置指南](environment.md) 创建并激活环境，再按 [README 路径表](../README.md#模型与数据设置) 设置 `config.yaml` 并复制两个 pilot。以下使用示例环境名 `graphskill`，可换成自己创建的名称；从仓库根目录执行。

```bash
conda activate graphskill
python -m pip check
python -m repro show-config --config configs/searchqa.pilot.yaml
python -m repro doctor --config configs/searchqa.pilot.yaml
python -m repro validate-data --config configs/searchqa.pilot.yaml
python -m repro validate-data --config configs/hotpotqa.pilot.yaml
python scripts/run_pilots.py --set evolution.generations=2
```

两个 pilot 均应为真实数据 10 train / 5 val / 0 test。该入口使用一份临时模型服务，依次运行两个数据集的无技能、初始技能、两轮原方法及选定技能重复验证，共八个子运行；两轮可覆盖默认的变异和交叉算子。程序结束后释放自己启动的服务。一张 GPU 上依次安排 B 和 C，不预先手动启动另一份模型服务。

pilot 是基础设施与完整流程验收，`research_results=false`，不是正式研究结果。HotpotQA pilot 是已有小规模工程数据；正式 HotpotQA 的分层数据和 300/150/1000 规模以 [hotpot_split.md](../hotpot_split.md) 为准，二者的成绩不混合。

程序打印 `outputs/pilot-check/<batch>/`。选定技能的独立验证运行位于：

```text
outputs/pilot-check/<batch>/runs/searchqa/selected_val/<run>/
outputs/pilot-check/<batch>/runs/hotpotqa/selected_val/<run>/
```

分别将实际目录传给汇总命令：

```bash
python -m repro summarize --run "outputs/pilot-check/<batch>/runs/searchqa/selected_val/<run>"
python -m repro summarize --run "outputs/pilot-check/<batch>/runs/hotpotqa/selected_val/<run>"
```

尖括号内容须替换为本次真实路径。汇总只读保存预测，不调用模型。不要用 `original` 目录中重复训练／验证预测的总体均值充当最终技能分数。

## 验收标准

| 检查内容 | 通过标准 |
|---|---|
| 批次 | `summary.json` 状态为 `completed`，八个子运行完成，`temporary_server_stopped=true` |
| 测试边界 | `scope.json` 的 `test_examples_used=false`；所有预测只属于 train/val |
| 两轮进化 | 两个 `original` 子运行的 `run_summary.json` 均为两轮，并保存 `upstream/best_skill.md` |
| 独立验证 | 两个 `selected_val` 子运行各 5 条唯一 val ID；有 EM/F1、延迟和对应技能 |
| 请求与用量 | 检查 `calls.jsonl` 与摘要，无 pending 或未知用量；错误与重试如实记录 |
| 截断 | 原文与标记保留，分数按既定规则为零；不为通过验收提高输出上限 |
| 可追溯性 | 可从一个验证结果找到配置、模型指纹、原始提示、响应及单题分数 |
| 演化解释 | 能区分初始化、父代、变异／交叉、验证选择、保留决定与最终技能 |

数值不要求达到某个 EM，也不要求与另一台机器完全相同。若模型或运行库不同，应记录区别，不能声称复现了同一环境下的研究数值。失败的运行原样保存，修复问题后用新目录重跑，并在记录中链接两次尝试。

## 执行人记录

截至本次文档更新，两位成员的独立验收均待本人执行；下表不是通过凭证。

| 成员 | 执行日期 | 本次 pilot 批次目录 | A 的复核记录 | 状态 |
|---|---|---|---|---|
| B | 待填写 | 待填写 | 待填写 | 待独立执行 |
| C | 待填写 | 待填写 | 待填写 | 待独立执行 |

每位执行人复制并填写以下记录：

```text
成员与日期：
解释器路径、Python 版本：
操作系统、GPU、依赖快照：
模型目录与 model_identity.json 指纹：
Git commit 与未提交修改说明：
实际命令：
pilot 批次目录与 summary.json 状态：
SearchQA selected_val 目录、n、EM、F1、延迟：
HotpotQA selected_val 目录、n、EM、F1、延迟：
总调用、输入/输出 token、失败/重试、pending/未知用量：
截断条数及计零分核对：
一条验证样例的原始提示、响应、result.json 路径：
候选图、lineage.mmd 与保留决定的位置：
失败尝试或环境差异：
A 的复核人、日期与结论：
```

## 交接范围

A 交付当前 Linux 的两套开发批次、各自审计、固定数据及 ID；完整路径见 [成员 A 初稿的证据索引](member-a-report.md#交付与证据索引)。`outputs/`、数据和权重被 Git 忽略，需单独传递，不能只发送仓库链接。

B 设计方法使用训练与开发材料。测试批次和 `test-failures-descriptive-only.csv` 只交给最终评测与描述分析使用，不纳入 B 的选择器开发材料。C 接手指标复核、失败分类与报告整合；A 在 B 的实际选择器接入后配合检查接口，并参与交叉检查与代码面试演练。
