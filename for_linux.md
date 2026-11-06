# Linux 配置入口

从 [Linux 配置指南](docs/environment.md) 开始：准备驱动和 CUDA 编译工具、创建 Python 3.12 环境、先安装 PyTorch，再通过 `requirements.txt` 安装项目与扩展。

- [安装命令](README.md#linux-安装)：Linux 统一安装入口和依赖锁。
- [模型与数据路径](README.md#模型与数据设置)：在根 `config.yaml` 指定本地模型并复制固定数据。
- [独立复现验收](docs/reproduction-acceptance.md)：两个真实数据集的两轮小实验。

GGUF 已弃用，新环境使用 Transformers。HotpotQA 按 [hotpot_split.md](hotpot_split.md) 使用 300/150/1000；新正式开发前还需处理 [说明版本](docs/configuration.md#hotpotqa-说明版本)，不改写历史清单绕过校验。
