from __future__ import annotations

import json

import httpx

from repro import diagnostics


def _model_files(path):
    path.mkdir()
    (path / "config.json").write_text('{"model_type":"qwen3_5"}', encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")


def test_sharded_model_requires_every_referenced_shard(tmp_path):
    model = tmp_path / "model"
    _model_files(model)
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "part1.safetensors", "b": "part2.safetensors"}}), encoding="utf-8"
    )
    (model / "part1.safetensors").write_bytes(b"weights")
    complete, message = diagnostics.validate_model_directory(model)
    assert not complete
    assert "part2.safetensors" in message
    (model / "part2.safetensors").write_bytes(b"weights")
    assert diagnostics.validate_model_directory(model)[0]


def test_model_index_cannot_escape_local_model_directory(tmp_path):
    model = tmp_path / "model"
    _model_files(model)
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "../outside.safetensors"}}), encoding="utf-8"
    )
    (tmp_path / "outside.safetensors").write_bytes(b"weights")
    assert not diagnostics.validate_model_directory(model)[0]


def test_weights_deferred_does_not_make_infrastructure_unready(tmp_path, monkeypatch):
    for name in (
        "check_python",
        "check_dependencies",
        "check_upstream",
        "check_gpu",
        "check_inference_runtime",
        "check_data",
    ):
        monkeypatch.setattr(diagnostics, name, lambda *args: diagnostics._check("required", "ok", "ready"))
    monkeypatch.delenv("REPRO_MODEL_PATH", raising=False)
    report = diagnostics.collect_diagnostics(tmp_path)
    assert report["infrastructure_ready"]
    assert not report["model_ready"]
    assert not report["experiment_ready"]
    assert report["checks"][-1]["status"] == "deferred"


def test_gpu_output_must_contain_valid_device(monkeypatch):
    monkeypatch.setattr(diagnostics, "run_command", lambda *args, **kwargs: (0, "unexpected output"))
    assert diagnostics.check_gpu()["status"] != "ok"


def test_native_runtime_checks_current_interpreter_without_environment_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics.sys, "executable", "custom-environment/python.exe")
    monkeypatch.setattr(diagnostics.platform, "system", lambda: "Windows")

    def run(command, **kwargs):
        assert command[:2] == ["custom-environment/python.exe", "-c"]
        return 0, json.dumps({"gpu_available": True, "bf16_matmul_result": 8.0})

    monkeypatch.setattr(diagnostics, "run_command", run)
    assert diagnostics.check_inference_runtime(tmp_path)["status"] == "ok"


def test_data_readiness_checks_manifest_and_all_splits(tmp_path):
    from repro.contracts import QAExample
    from repro.datasets.manifests import freeze_splits

    data = tmp_path / "splits"
    examples = {
        split: [QAExample(f"{split}-1", "searchqa", split, "question", "context", ("answer",))]
        for split in ("train", "val", "test")
    }
    manifest = freeze_splits("searchqa", examples, data, {"train": 1, "val": 1, "test": 1})
    config = {"data": {"dataset": "hotpotqa", "split_dir": "splits", "manifest": "fixed.json"}}
    assert diagnostics.check_data(tmp_path, config)["status"] == "error"
    config["data"]["dataset"] = "searchqa"
    assert diagnostics.check_data(tmp_path, config)["status"] == "missing"
    (tmp_path / "fixed.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert diagnostics.check_data(tmp_path, config)["status"] == "ok"
    with (data / "train.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    assert diagnostics.check_data(tmp_path, config)["status"] == "error"


def test_doctor_rejects_synthetic_dataset_with_course_profile(tmp_path):
    from repro.contracts import QAExample
    from repro.datasets.manifests import freeze_splits

    examples = {
        split: [
            QAExample(
                f"{split}-1", "hotpotqa", split, "question", "context", ("answer",), {"synthetic": True}
            )
        ]
        for split in ("train", "val", "test")
    }
    freeze_splits("hotpotqa", examples, tmp_path / "splits", {"train": 1, "val": 1, "test": 1})
    config = {"data": {"dataset": "hotpotqa", "split_dir": "splits"}, "experiment": {"profile": "course"}}
    assert diagnostics.check_data(tmp_path, config)["status"] == "error"


def test_endpoint_with_embedded_credentials_is_rejected():
    result = diagnostics.check_model_endpoint({"model": {"base_url": "http://secret:password@localhost/v1"}})
    assert result["status"] == "error"
    assert "password" not in result["message"]


def _llama_runtime_files(root):
    runtime = root / "tools/llama.cpp"
    runtime.mkdir(parents=True)
    executable = runtime / "b1234/llama-server.exe"
    executable.parent.mkdir()
    executable.write_bytes(b"local executable fixture")
    (runtime / "runtime.lock.json").write_text(
        json.dumps(
            {
                "release": "b1234",
                "executable": "tools/llama.cpp/b1234/llama-server.exe",
            }
        ),
        encoding="utf-8",
    )
    return executable


def test_gguf_runtime_is_independent_of_torch(tmp_path, monkeypatch):
    executable = _llama_runtime_files(tmp_path)

    def run(command, **kwargs):
        assert command[0] == str(executable)
        return (0, "version: b1234") if command[-1] == "--version" else (0, "CUDA0: RTX 5090")

    monkeypatch.setattr(diagnostics, "run_command", run)
    report = diagnostics.check_inference_runtime(tmp_path, {"model": {"backend": "llama_cpp"}})
    assert report["status"] == "ok"
    assert report["backend"] == "llama_cpp"
    assert "torch" not in report


def test_gguf_runtime_rejects_missing_cuda_device(tmp_path, monkeypatch):
    _llama_runtime_files(tmp_path)
    monkeypatch.setattr(diagnostics, "run_command", lambda *args, **kwargs: (0, "CPU only"))
    assert (
        diagnostics.check_inference_runtime(tmp_path, {"model": {"backend": "llama_cpp"}})["status"]
        == "error"
    )


def test_gguf_model_readiness_rejects_corrupt_header(tmp_path):
    model = tmp_path / "corrupt.gguf"
    model.write_bytes(b"not GGUF weights")
    report = diagnostics.check_model(tmp_path, {"model": {"backend": "llama_cpp", "local_path": str(model)}})
    assert report["status"] == "error"
    assert "successful inference" in report["note"]


def test_llama_endpoint_context_must_match_global_config(monkeypatch):
    factory = httpx.Client

    def respond(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "local-gguf"}]})
        return httpx.Response(
            200,
            json={
                "build_info": "b1234",
                "total_slots": 1,
                "default_generation_settings": {
                    "n_ctx": 4096,
                    "params": {
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                },
            },
        )

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: factory(
            **kwargs,
            transport=httpx.MockTransport(respond),
        ),
    )
    config = {
        "model": {
            "backend": "llama_cpp",
            "name": "local-gguf",
            "max_context_tokens": 8192,
            "enable_thinking": False,
            "base_url": "http://127.0.0.1:8000/v1",
        }
    }
    report = diagnostics.check_model_endpoint(config)
    assert report["status"] == "error"
    assert report["reported_context_tokens"] == 4096
    config["model"]["max_context_tokens"] = 4096
    assert diagnostics.check_model_endpoint(config)["status"] == "ok"


def test_reasoning_parser_flags_do_not_prove_thinking_mode(monkeypatch):
    factory = httpx.Client

    def respond(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "local-gguf"}]})
        return httpx.Response(
            200,
            json={
                "default_generation_settings": {
                    "n_ctx": 8192,
                    "params": {"reasoning_format": "none", "reasoning_in_content": False},
                }
            },
        )

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: factory(
            **kwargs,
            transport=httpx.MockTransport(respond),
        ),
    )
    result = diagnostics.check_model_endpoint(
        {
            "model": {
                "backend": "llama_cpp",
                "name": "local-gguf",
                "max_context_tokens": 8192,
                "enable_thinking": False,
                "base_url": "http://127.0.0.1:8000/v1",
            }
        }
    )
    assert result["status"] == "ok"
    assert result["reasoning_parser_format"] == "none"
    assert result["reported_enable_thinking"] is None
    assert result["thinking_mode_verified"] is False
