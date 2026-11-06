# Linux 运行入口

当前使用原生 Linux conda `jev`，已完成 SearchQA 与 HotpotQA 的三个种子开发及冻结测试。无需重建现有环境或重新下载模型。

```bash
conda activate jev
python -m pip check
python -m repro doctor --config configs/searchqa.pilot.yaml
```

根 `config.yaml` 的 `model.local_path` 默认指向 `/home/admin1/Qwen3.5-9B`；换机器时修改该项，并清空或同步本机 `.env` / 环境变量中的 `REPRO_MODEL_PATH`。数据目录见 [README 路径表](README.md#模型与数据设置)。Linux 依赖快照为 [requirements/lock.linux.txt](requirements/lock.linux.txt)；`requirements.txt` 引用 Windows 锁，不用于 Linux 安装。GGUF 已弃用，新环境使用 Transformers。

- [环境与服务](docs/environment.md)：已有 `jev`、新机器准备、模型路径和自动服务管理。
- [B/C 独立复现](README.md#b-c-独立复现验收)：真实小实验命令与验收要求。
- [成员 A 报告](docs/member-a-report.md)：当前结果、技能演化和原始证据。
- [实验与交接](docs/reproduction-plan.md)：正式入口、冻结测试与团队分工。

HotpotQA 使用 [hotpot_split.md](hotpot_split.md) 的 300/150/1000。当前说明已修订，历史清单仍锁定旧哈希；新的分层准备／正式开发需先处理 [协议版本](docs/configuration.md#hotpotqa-说明版本)。已有 pilot 与完成结果不受影响。保留固定数据、ID 和所有历史输出，不修改清单绕过校验。
