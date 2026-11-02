# GGUF 备用后端

GGUF 的检查、运行库安装和服务启动统一由 `scripts/gguf.py` 提供。主实验使用 Transformers 加载 Qwen3.5-9B；GGUF 保留为后续量化模型的备用路线。

## 配置

在根 `config.yaml` 的 `presets.local.gguf.model` 中设置已有 `.gguf` 文件的路径、服务名称、上下文和生成参数。`configs/local.gguf.yaml` 只是该方案的入口，不重复保存参数。
若 `.env` 或进程环境中存在非空模型覆盖项，它们仍优先于 YAML；先检查实际生效配置。

```powershell
python -m repro show-config --config configs/local.gguf.yaml
python scripts/gguf.py inspect
python scripts/gguf.py inspect --sha256
```

`inspect` 只读取本地文件，检查 GGUF 结构、张量范围和分片是否完整；`--sha256` 会额外读取整个文件计算哈希。这些检查不加载模型，也不生成回答。

## 运行库与服务

运行库版本及归档、可执行文件 SHA-256 固定在 `tools/llama.cpp/runtime.lock.json`。需要准备运行库时执行：

```powershell
python scripts/gguf.py install
python scripts/gguf.py serve
```

安装器仅下载锁定的官方 Windows x64 CUDA 运行库归档，校验文件后解压；已有不同内容的文件会被拒绝覆盖。它不会下载模型权重，也不会改写版本锁。此锁对应 CUDA 13.3 的 Windows 运行库，机器须具有兼容的 NVIDIA 驱动。

服务只加载配置指向的本地 GGUF，监听本机并提供 OpenAI 兼容接口；按 `Ctrl+C` 结束。可使用 `--config` 选择另一份 GGUF 配置。公开启动入口只使用版本锁内的运行库并核验可执行文件 SHA-256，保证与实验记录中的运行库来源一致。
日常修改模型位置和服务端口应通过 YAML 完成；临时使用 `--model-path` 或 `--port` 覆盖时，客户端实验配置也须指向相同模型和地址。

## 基本验证与实验入口

服务启动后，在另一终端使用已有公共入口检查服务，或运行固定 pilot 验证集的无技能条件：

```powershell
python -m repro doctor --config config.yaml --preset searchqa.pilot --preset local.gguf --endpoint
python -m repro run --config config.yaml --preset searchqa.pilot --preset local.gguf --condition no_skill
```

模型请求仍经公共计量客户端，数据划分、测试开关、预测和调用日志沿用原实验框架。切换为不同权重或量化模型后应单独建立比较批次，不能把旧 Transformers 成绩当成该后端的运行结果。

早期 `probe_gguf.py`、`check_gguf_pilot.py`、独立安装脚本和 PowerShell 启动包装已被本入口替代；既有实验产物保留在本地 `outputs/`，不参与当前代码执行。
