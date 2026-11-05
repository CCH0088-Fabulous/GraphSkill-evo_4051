下面是更新后的清单。**说明文件已修好，统一使用 `hotpot_split.md`，不需要再复制旧文件名。**

代码、数据、权重和历史输出可共用；Linux 需独立环境。以下先按现有 **Transformers 路线**准备，批量并行目前还需要代码适配。

1. **进入 Linux，确认挂载目录和 GPU**

   `/mnt/shared` 是示例，换成实际挂载点：

   ```bash
   PROJECT_DIR="/mnt/shared/GraphSkill-evo_4051"
   MODEL_DIR="/mnt/shared/Qwen3.5-9B"

   cd "$PROJECT_DIR"
   nvidia-smi
   test -r "$MODEL_DIR/config.json"
   ```

   保留现有 `data/`、`manifests/`、`outputs/`，无需重新下载权重或抽样。

2. **创建 Linux 专用 Python 环境**

   建议 Python 3.12，环境放在 Linux 本机文件系统，不复用 Windows `.venv`：

   ```bash
   python3.12 -m venv "$HOME/.venvs/graphskill-linux"
   source "$HOME/.venvs/graphskill-linux/bin/activate"
   python -m pip install --upgrade pip setuptools wheel
   ```

3. **安装 CUDA PyTorch**

   根据 Linux 驱动，在 [PyTorch 官方页面](https://pytorch.org/get-started/locally/)选择匹配的 Linux/CUDA 安装命令。

   **当前 `requirements.txt` 引用 Windows 锁，不要直接用于 Linux。** 仓库尚无经过验证的 Linux 依赖锁。

4. **安装项目和加速依赖**

   ```bash
   python scripts/fetch_sources.py
   python -m pip install -e ".[data,dev,server]" -e ./third_party/GraphSkillEvo

   python -m pip install ninja packaging
   python -m pip install "flash-linear-attention[cuda]"
   python -m pip install causal-conv1d --no-build-isolation

   python -m pip check
   ```

   加速包必须与所选 PyTorch/CUDA 配套；没有匹配 wheel 时，可能需要 CUDA Toolkit、`nvcc` 和编译工具。具体安装方式以 [FLA 官方说明](https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md)为准，不能把“安装成功”直接当作性能验收完成。

5. **指定 Linux 模型路径**

   ```bash
   export REPRO_MODEL_PATH="$MODEL_DIR"
   export HF_HUB_OFFLINE=1
   export TRANSFORMERS_OFFLINE=1
   ```

   这个覆盖适用于普通开发入口，避免将共享 `config.yaml` 的路径反复改成不同系统格式。

6. **核验环境、数据和基本流程**

   ```bash
   python -c "import torch, triton, fla, causal_conv1d; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.cuda.is_bf16_supported())"

   python -m repro show-config --config config.yaml --preset course --preset hotpotqa.provisional
   python -m repro validate-data --config config.yaml --preset course --preset hotpotqa.provisional
   python -m repro doctor --config config.yaml --preset course --preset hotpotqa.provisional

   python -m pytest -q
   python -m repro smoke --dataset hotpotqa
   ```

   确认模型路径正确、CUDA/BF16 可用，数据仍为 **300／150／1000**。

7. **用训练样例验收实际推理**

   ```bash
   python scripts/run_example.py \
     --config configs/hotpotqa.pilot.yaml \
     --max-tokens 4096
   ```

   检查服务日志中不再出现之前四条 reference PyTorch 慢路径警告，再用固定训练样例测稳定吞吐和显存。

   **当前仍是单题串行。** 若目标是并行提速，需要先补客户端和服务端的批量处理，按并发 `1 → 2 → 4` 验收；不要同时加载三份模型。

8. **验收通过后，固定 Linux 依赖版本**

   ```bash
   python -m pip freeze --exclude-editable > requirements/lock.linux.txt
   ```

   同时记录 Python、驱动、CUDA、推理后端和并发设置。正式对照使用同一套固定环境。

9. **建立独立 Linux 实验批次**

   以下是现有串行入口：

   ```bash
   python scripts/run_formal.py --provisional-hotpotqa --seeds 0 42 127 \
     --set experiment.output_root=outputs/linux-transformers
   ```

   开发批次完成、技能选定后，再测试：

   ```bash
   python scripts/run_formal.py --test-from-batch "新Linux开发批次的实际目录"
   ```

   **不要直接续用旧 Windows 测试计划**：它保留了 Windows 模型路径。若只想迁移旧技能、避免重跑进化，需要先补路径重定位和身份校验入口，不能手改历史计划绕过检查。

本次只提供清单，实验仍保持停止。