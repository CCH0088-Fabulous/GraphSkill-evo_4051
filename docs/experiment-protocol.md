# 实验运行约定

本文件约束共同运行方式；B 的研究改进只改变训练失败案例选择，C 维护评分与分析。

## 固定条件

原方法、改进和消融共享模型文件、precision、thinking、解码、输入处理、上下文、生成上限、超时、重试及调用/token 预算。GPU 小时是实测资源记录与课程规划上限，不用于候选排名；相同轮数不能代替预算匹配。

SearchQA 使用 `course` profile 的 100/30/50。HotpotQA 按 [hotpot_split.md](../hotpot_split.md) 使用分层 300/150/1000，配置名 `hotpotqa.provisional` 与 `custom` profile 继续保留以兼容历史记录；历史状态字段不改写。两者均最多三轮、每轮最多两个新候选。`synthetic` profile 仅用于标记清楚的合成接线检查，不计入真实实验成绩。

初始化种群与每轮新候选数不同。默认保留种群为二，包含提供的初始技能与模型生成的候选；每轮两个新候选，按 `mutation_other`、`mutation_graph`、`crossover_other`、`crossover_graph` 跨轮轮转。一轮 pilot 验证接线，两轮才能覆盖四种默认算子。初始化及各轮分别限制算子尝试；HTTP 重试由统一 client 负责，每次请求单独记账。

## 划分与反馈

先冻结 ID、来源 revision 及原始/标准化数据校验值，再查看结果。SearchQA 使用上游发布的固定 ID；HotpotQA 从官方 train 按难度分配互斥训练/验证，从官方 distractor dev 选留出测试，提交三份原始 ID。所有给定上下文均输入求解器，不附加外部检索、参考答案或 supporting-fact 标签。

HotpotQA 当前说明与历史清单的字节哈希不同，新的分层准备及正式开发预检会拒绝；处理方式与限制见 [说明版本](configuration.md#hotpotqa-说明版本)。本次文档更新不重写固定数据、清单或已完成输出。

训练执行与训练参考答案形成优化反馈。验证成绩用于父代、幸存者和最终技能选择，验证失败供诊断；重复使用验证集的选择偏差须在报告披露。进化 dataloader 只持有训练和验证，冻结测试独立执行且不调用优化器。测试失败不用于继续开发方法。

本项目报告答案 EM/F1，不将其称为 HotpotQA 支持事实或联合指标。三个实验种子共用同一批固定题，跨种子记录不增加独立题数。

## 可追溯计量

模型请求统一经过 `inference/client.py`；初始化、修订、格式错误、失败与重试都必须计量。未知 usage 保留预留并标记缺失，不能补成零。相同技能内容的验证缓存仅在同一次独立运行内使用，上游明确标记复用，不算新独立评测。

问答截断保留原文、空计分答案、EM/F1 为零；优化器截断或格式不合格在原尝试预算内重试，不增加输出上限。单题 latency 包含失败重试等待，`last_request_latency_s` 单列最后请求。GPU 采样记录设备显存、利用率、功率与墙钟，不能当作 CUDA 内核时间或独占显存。

结果表来自独立加载最终技能的评测；进化中重复采样的训练预测不能混入测试均值。保留失败尝试、分段历史与原始证据，新运行使用新目录。

## 验收与当前进度

两个数据集的 A 条件均已在 Linux `jev` 完成三个种子的开发与冻结测试，逐题重算审计通过；结果与限制见 [成员 A 报告](member-a-report.md)。原方法未显示稳定测试优势，B 的改进／消融及 C 的完整诊断仍待完成。

B、C 各自按 [README](../README.md#b-c-独立复现验收) 运行真实双数据集两轮 pilot 并填写 [验收记录](reproduction-acceptance.md)。该流程不含测试题。`repro smoke` 是 synthetic 接线检查，单题 `run_example.py` 是模型检查，均不代替完整验收。

早期 Windows 开发报告见 [单种子结果](formal-results.md)、[分层 HotpotQA 结果](hotpotqa-v2-results.md)；更早记录见 [历史归档](archive/README.md)。其中旧参数与状态只解释当时的证据。
