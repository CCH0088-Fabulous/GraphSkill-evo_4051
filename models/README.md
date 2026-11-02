# 本地模型目录

基础设施脚本不下载模型权重。当前共享配置使用用户提供的完整 `E:\Qwen3.5-9B`。
仓库内 `models/Qwen3.5-9B` 仅为早期固定版本的非权重资源缓存，不与外部模型目录混用。

后续请放置完整 Hugging Face 模型文件，包括模型配置、tokenizer 和全部 safetensors 分片。
启动脚本只加载本地完整目录，不会根据模型名称自动联网补文件。

```powershell
python -m repro.serving.transformers_launcher --config config.yaml
```

模型名 `Qwen/Qwen3.5-9B` 是 HTTP API 中的固定标识，模型路径是磁盘目录；两者分别配置。

GGUF 文件放在仓库外，通过根配置的 `local.gguf` preset 指定；检查、运行库安装和启动统一使用 `scripts/gguf.py`，见 [GGUF 说明](../docs/gguf.md)。
