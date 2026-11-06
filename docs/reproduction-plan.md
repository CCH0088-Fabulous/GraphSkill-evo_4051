# 成员 A 复现与交付

A 负责公共框架、原方法与基线，B 负责改进失败选择器和消融，C 负责评分、失败分析和报告整合。要求见 [课程 PDF](../nlp-project5.pdf) 与 [小组分工](../group_div.md)。

## 当前进度

SearchQA 和 HotpotQA 均已在原生 Linux conda `jev` 完成 seed 0、42、127 的 A 条件：每个数据集 12/12 开发子运行、9/9 冻结测试子运行完成，逐题重算审计通过。原方法未显示稳定优于无技能；完整结果、成本、技能演化与四个原始批次见 [成员 A 报告](member-a-report.md)。旧 Windows 成绩继续作为历史记录保留，不与当前 Linux 数值混用。

A 的复现设置、系统实现和演示初稿已整理；B/C 的独立小实验验收仍待本人执行。B 的方法与消融、C 的完整失败分类及团队报告仍未由本轮 A 实验代替。

## 环境与固定数据

按 [环境说明](environment.md) 激活已有 `jev`，确认本地模型和实际配置。以下命令只做数据校验，不调用模型：

```bash
python -m repro show-config --config configs/searchqa.course.yaml
python -m repro validate-data --config configs/searchqa.course.yaml
python -m repro validate-data --config config.yaml --preset course --preset hotpotqa.provisional
```

SearchQA 使用 PDF 的 100/30/50 和上游固定 ID。HotpotQA 按当前 [hotpot_split.md](../hotpot_split.md) 使用 300/150/1000：官方 train 每难度各 100 训练、50 验证，官方 distractor dev 取 1000 留出测试；原始 ID 互斥、上下文全部保留。数据抽样 seed 固定为 42，不随三个实验种子改变。

HotpotQA 的 `hotpotqa.provisional` 是保留的配置名称，历史清单还保留旧说明哈希及状态字段。当前说明已统一测试数为 1000，但字节哈希不同；`prepare_hotpotqa.py --stratified` 和新的正式开发预检目前会拒绝这项差异。新正式实验前需显式迁移协议版本，详见 [说明版本](configuration.md#hotpotqa-说明版本)。现有数据验证、两个小型 pilot 和已完成结果仍可读取；不要重写冻结清单绕过检查。

## 独立小实验验收

B、C 各按 [README](../README.md#b-c-独立复现验收) 执行一次，并填写 [验收记录](reproduction-acceptance.md)。两个真实 pilot 各为 10 train / 5 val / 0 test；两轮进化覆盖默认的变异与交叉，不当作正式研究数值。

无模型时的 `repro smoke` 仅检查 synthetic 离线接线，不能替代真实验收。A 不用已有批次代替 B/C 本人执行，也不在记录中预填通过。

## 正式开发入口

以下记录用于理解或将来建立新批次，不需要重跑已经完成的实验。SearchQA 的三种子入口是：

```bash
python scripts/run_formal.py --dataset searchqa --seeds 0 42 127 --set experiment.output_root=outputs/linux-transformers
```

每种子依次运行 `no_skill`、`initial_skill`、`original`，再独立加载最终技能评估 `selected_val`，共 12 个子运行。三轮进化、保留种群二、每轮最多两个新候选、训练评估 15 题、最多三条失败反馈；每条件上限 600 次调用 / 200 万输出 token。

HotpotQA 历史入口为 `scripts/run_formal.py --provisional-hotpotqa --seeds 0 42 127 --set experiment.output_root=outputs/linux-transformers`，当前受上述说明哈希检查限制。它使用 300/150/1000，预算为每条件 3000 次调用 / 1000 万输出 token，其余进化和模型规则与 SearchQA 相同。原 `course` profile 的 100/30/50 上限继续保留。

所有批次写入新目录并自动管理服务。各方法保持模型、解码、数据、种子与预算一致；B 的 `improved`、`ablation` 必须实际注册，未实现时显式失败，不能退回 `original`。

## 冻结测试与续跑

A 的两套测试已经完成。将来只有在对应方法与设置冻结后才对新开发批次运行：

```bash
python scripts/run_formal.py --test-from-batch "已完成开发批次的实际目录"
```

入口验证开发记录、固定数据、模型及选定技能，保存冻结副本，再执行三个种子 × 三条件的独立测试，不调用优化器。它自动识别 SearchQA 的 50 题或分层 HotpotQA 的 1000 题，拒绝 `--config`、`--set`、`--seeds` 等额外覆盖。

测试中断后可用 `python scripts/run_formal.py --resume-heldout "本次中断测试批次的实际目录"`：保留完成子运行，其余写入新子目录；不支持逐题续跑，也不恢复进化状态。失败、截断及未知用量原样保留，不能覆盖旧运行。HotpotQA 历史测试中断过一次，因此最终摘要的续跑耗时不能代表全批次耗时。

测试预测只用于最终评估与描述，不回流提示、选择器、调参或技能选择。三个种子重复使用相同测试 ID，不能扩大独立题数。

## 交接与剩余工作

完整批次含生效配置、环境与模型指纹、固定数据清单、逐题提示／响应、调用账本、技能、内部图与父子谱系。具体证据路径见 [成员 A 报告索引](member-a-report.md#交付与证据索引)。`outputs/`、原始数据和模型被 Git 忽略，交接需单独复制完整文件。

C 接收完整开发失败清单并分类、诊断、重算评分；B 使用训练和开发材料设计选择器，每次技能更新仍只接收训练失败。测试材料单独交给最终评测与描述分析。A 配合实际 B 方法接入检查，复核 B/C 验收，并参与团队报告、演示与代码面试演练。
