"""Check run labels, path independence, and invalid evaluation conditions."""

import json
from copy import deepcopy

import httpx
import pytest

from repro.config import DEFAULTS, PROJECT_ROOT, load_config
from repro.contracts import QAExample
from repro.datasets.manifests import freeze_splits
from repro.experiments.baselines import run_baseline
from repro.inference.client import ChatClient


def config(tmp_path, condition="no_skill"):
    return load_config(
        PROJECT_ROOT / "configs/offline.searchqa.yaml",
        [
            f"experiment.condition={condition}",
            f"experiment.output_root={tmp_path.as_posix()}",
        ],
    )


def test_course_run_allows_local_model_path_but_rejects_changed_data_identity(tmp_path, monkeypatch):
    from repro.experiments.context import prepare_run

    source = {"revision": "source-revision", "raw_files": {"train": {"sha256": "raw-hash"}}}
    data = tmp_path / "data"
    item = QAExample("a", "searchqa", "train", "question", "context", ("answer",))
    manifest = freeze_splits("searchqa", {"train": [item]}, data, {"train": 1}, source=source)
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(manifest), encoding="utf-8")
    cfg = deepcopy(DEFAULTS)
    cfg["data"].update(split_dir=str(data), manifest=str(reference))
    cfg["model"]["local_path"] = "D:/another-machine/model"
    cfg["experiment"]["output_root"] = str(tmp_path / "runs")
    monkeypatch.setattr("repro.experiments.context.capture_model_identity", lambda *args, **kwargs: {})
    monkeypatch.setattr("repro.experiments.context.source_snapshot", lambda: {})
    monkeypatch.setattr("repro.experiments.context.environment_snapshot", lambda **kwargs: {})
    store, client = prepare_run(cfg)
    client.close()
    assert not (store.path / "frozen_protocol.json").exists()

    manifest["source"]["raw_files"]["train"]["sha256"] = "changed-raw-hash"
    (data / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source checksums"):
        prepare_run(cfg)


def test_baseline_resolves_data_from_project_root_outside_cwd(tmp_path, monkeypatch):
    cfg = config(tmp_path / "runs")
    monkeypatch.chdir(tmp_path)
    run = run_baseline(cfg, offline=True)
    summary = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["n"] == 3
    assert summary["synthetic"]
    assert summary["backend_offline"]
    assert summary["failed_predictions"] == 0


def test_http_backend_does_not_make_synthetic_data_real(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg["model"]["backend"] = "transformers"

    def respond(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": cfg["model"]["name"]}]})
        if request.url.path == "/health":
            return httpx.Response(200, json={"backend": "transformers", "enable_thinking": False})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "<answer>Paris</answer>"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            },
        )

    monkeypatch.setattr(
        "repro.experiments.context.ChatClient",
        lambda settings, ledger, seed: ChatClient(
            settings,
            ledger,
            seed=seed,
            transport=httpx.MockTransport(respond),
        ),
    )
    run = run_baseline(cfg)
    summary = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["synthetic"] is True
    assert summary["backend_offline"] is False


@pytest.mark.parametrize("condition", ["no_skill", "initial_skill"])
def test_saved_skill_cannot_be_mislabeled_as_baseline(tmp_path, condition):
    fake_evolved = tmp_path / "evolved.md"
    fake_evolved.write_text("evolved artifact", encoding="utf-8")
    with pytest.raises(ValueError):
        run_baseline(config(tmp_path, condition), saved_skill=fake_evolved, offline=True)


def test_preflight_failure_is_saved_without_fake_predictions(tmp_path, monkeypatch):
    def missing_service(request):
        raise httpx.ConnectError("not running", request=request)

    monkeypatch.setattr(
        "repro.experiments.context.ChatClient",
        lambda settings, ledger, seed: ChatClient(
            settings,
            ledger,
            seed=seed,
            transport=httpx.MockTransport(missing_service),
        ),
    )
    with pytest.raises(RuntimeError, match="Model service is unavailable"):
        run_baseline(config(tmp_path))
    summary = json.loads(next(tmp_path.glob("*/run_summary.json")).read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["saved_predictions"] == 0
    assert summary["usage"]["calls"] == 0
