# 正式实验公共设置 v1

2026-10-30 发布。范围为公共实验协议与成员 A 的 `no_skill`、`initial_skill`、`original` 三条件。
B 的 `improved` / `ablation` 尚未实现或批准，接入后需要新版本，不能覆盖 v1 或假称已完成整个团队的方法冻结。

## 公共参数

| 项目 | v1 设置 |
|---|---|
| 模型 | `E:\Qwen3.5-9B`，权重、tokenizer 与配置 SHA-256 固定 |
| 后端 | 原生 Windows，Transformers，BF16 / SDPA，CUDA 单卡串行，thinking 关闭 |
| 上下文 / 输出 | 总窗口 16384；问答与技能算子均最多生成 4096 token |
| 采样 | temperature 0.6、top_p 1.0、top_k 50 |
| 实验种子 | 42；v1 为单种子设置，新增种子需要新版本 |
| HTTP | 超时 360 秒，最多重试一次，每次计量 |
| 正式数据 | 两数据集各固定 100 train / 30 val / 50 test |
| 正式进化 | 三轮、保留种群二、每轮两个新候选；每次训练评估 15 题，最多取 3 条失败反馈 |
| 算子 | mutation_other、mutation_graph、crossover_other、crossover_graph 跨轮轮转 |
| 尝试上限 | graph_retry_attempts=2；初始化最多 6 次，进化每轮阶段总计最多 12 次逻辑尝试 |
| 优化预算 | 每数据集/条件/种子最多 600 次请求、2000000 输出 token，包含失败与重试 |
| 测试 | 默认关闭，方法和最终技能选择冻结后显式开启独立测试评估 |

原方法、改进和消融共享公共设置及上限，并报告实际消耗；基础基线无需人为重复请求来耗尽优化预算。
三轮在不复用验证缓存时约有 330 次问答，加最多 42 次算子逻辑尝试；全部成功且每次生成 4096 token 时约 152 万输出 token。
因此正式额度从早期暂定的 60 万调整到 200 万。网络失败和未知用量保守预留仍可能触发硬预算；耗尽即停止并保留部分产物，不能标记为正常完成。
40 GPU 小时仍是规划参考。

## 截断与重试

- 问答 `finish_reason=length`：保留原文，标记 `status=truncated`、`output_truncated=true`、`failure_type=output_truncated`；计分答案为空、EM/F1 为零。
  不自动续写、扩限、重跑或删除样本。训练选择器可选中此类失败，但反馈明确标为输出预算失败。
- 算子 `finish_reason=length`：拒绝整次不完整响应，不手工修复或接受其中片段；在原阶段尝试上限内换样本重试，完整计量。
- `optimizer_retry_v1`：每个 phase 内每个算子 stage 的首次请求沿用实验种子；后续尝试取紧凑 JSON 序列化
  `["optimizer_retry_v1", base_seed, phase, stage, stage_attempt]` 的 SHA-256 前 32 位作为无符号整数。次数及种子写入账本，重跑序列一致。
- 网络重试沿用相同请求体和种子，仅传输异常、429、5xx 可重试；不消耗新的算子逻辑尝试。`operator_attempt` / `seed_attempt` 与 `http_attempt` 分开记录。
- 缺失 usage 明确为未知，并保守预留输出额度，不填成零。任何预算耗尽都停止；不替换失败选择器。

## 验收证据

通过批次：`outputs/pilot-check/20261030T032905811231Z-e0ee70/`，检查器 `acceptance-audit.json` 状态 `passed`。
两数据集各用固定 10 train / 5 val / 0 test、每次训练评估 10 题、最多 3 条失败反馈，完成两轮及四种算子、两个基础条件与选定技能独立验证，八个子运行全部完成。
共 177 次请求，321632 输入 / 39520 输出 token，无 HTTP 失败和未知用量。墙钟约 24 分 24 秒，设备总显存采样峰值约 21.8 GiB（含其他应用，非精确 CUDA 内核测量）；服务已关闭。
SearchQA 同一训练题在两轮各达到一次 4096 上限，均明确标记并计零分；HotpotQA 与优化器无长度截断。
SearchQA 图交叉和 HotpotQA 初始化各一次格式重试，经派生种子成功恢复；未放宽上游校验或改写模型输出。
两数据集最终仍选初始技能，独立验证 EM/F1 为 SearchQA 0.8/0.8、HotpotQA 1.0/1.0；这些是小样本验收结果，不是正式研究结果。
先前失败批次 `20261030T030322660351Z-2f0b35` 保留，不计入通过结果。
验收最大问答输入 3600 token、算子输入 3801 token；第三轮及正式数据可能产生不同长度，运行时仍严格检查输入加请求输出窗口，不静默截断输入。
HotpotQA 是公开 distractor 数据自行固定的子集，`course_split_verified=false`；模型下载 revision 未独立核验，文件指纹已固定。

## 冻结与运行

`config.yaml` 保存公共参数，`configs/*.course.yaml` 保存正式数据/规模/预算，并引用 `manifests/experiment-v1/manifest.json`。
版本目录保存六份完整配置快照、模型指纹、数据清单、上游版本以及代码、脚本、依赖校验值；`manifests/active.json` 防止正式数据意外关闭冻结校验。
正式运行在发出模型请求前核对最终生效设置、受保护文件和包版本，再重算模型指纹及验证数据内容；产物保存 `frozen_protocol.json`。
只有输出目录、批准种子和显式最终测试开关允许变化。其他变化发布新版本，旧版拒绝覆盖。

```powershell
.\.venv\Scripts\python.exe -m repro verify-freeze --config configs/searchqa.course.yaml
.\.venv\Scripts\python.exe -m repro verify-freeze --config configs/hotpotqa.course.yaml
```

发布本版使用 `repro freeze --version v1 --acceptance-run outputs/pilot-check/20261030T032905811231Z-e0ee70`。
后续先启动模型服务，再在另一个终端运行正式条件；本次任务未启动正式研究运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/serve_model.ps1 -Config config.yaml
.\.venv\Scripts\python.exe -m repro run --config configs/searchqa.course.yaml --condition original
```
