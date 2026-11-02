# 环境与模型服务

当前已验证路线是原生 Windows、Python 3.13.11、CUDA、Transformers、BF16 / SDPA。使用已激活的 Python 环境；工具不要求环境一定叫 `.venv`。未保留未经验收的 Linux/vLLM 安装脚本，也不需要 WSL。

## 环境安装

```powershell
python scripts/fetch_sources.py
python -m pip install -r requirements.txt
python -m pip check
python -m repro doctor --config configs/searchqa.pilot.yaml
```

`requirements.txt` 引用 Windows 依赖锁，再安装项目和固定上游源码。安装过程不会重写锁文件。若尚无独立环境，可自行创建并激活 venv 或 Conda 环境，再执行上述命令。

项目脚本不下载模型权重。`config.yaml` 的 `model.local_path` 必须指向完整本地模型目录，包含配置、tokenizer、权重及索引引用的所有分片。`models/` 下若仍有早期非权重缓存，它不能替代完整模型，也不会与当前模型混用。

## Transformers 服务

```powershell
python -m repro.serving.transformers_launcher --config config.yaml
```

Windows 用户也可使用等价的简短包装器：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/serve_model.ps1 -Config config.yaml
```

包装器使用当前环境的 `python`，也可用 `-Python` 显式指定解释器。模型和服务设置以 YAML 为准。

服务只读完整本地模型，显式禁止 Hub 下载；当前固定 BF16、SDPA、逻辑 `cuda:0` 和串行生成。`model.base_url` 控制本机地址和端口，`max_context_tokens` 是输入加输出的总窗口。输入超限会报错，不自动截断提示。

提供 `GET /health`、`GET /v1/models` 和 `POST /v1/chat/completions`。实验客户端在执行前核对模型名、路径、上下文和 thinking 设置；每次请求显式携带生成上限，token 用量按实际 tokenizer ID 计数。

## 检查与服务生命周期

`doctor` 检查当前解释器、依赖、固定源码、GPU、数据和完整模型；加 `--endpoint` 可核对已启动的服务。

```powershell
python -m repro doctor --config configs/searchqa.pilot.yaml --endpoint
```

静态模型文件存在不等同于服务就绪：只有基础设施、权重和服务检查都成功时，`experiment_ready` 才为真。

`run_example.py`、`run_pilots.py` 和 `run_formal.py` 管理自己启动的临时服务，拒绝占用已有服务端口，结束后只关闭自身进程。逐组 `repro run/evaluate` 则连接已启动服务。

GGUF 的文件检查、固定运行库安装和启动统一见 [gguf.md](gguf.md)。正式结果与资源测量见 [formal-results.md](formal-results.md)；硬件和模型变化必须在对照中明确记录。
