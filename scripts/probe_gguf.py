"""Try an existing local GGUF on training examples, then release the trial GPU server."""

from __future__ import annotations

import json
import subprocess
import time

import httpx
from dotenv import load_dotenv

from repro.artifacts.store import RunStore
from repro.config import PROJECT_ROOT, load_config
from repro.datasets.io import load_split
from repro.inference.accounting import UsageLedger
from repro.inference.client import ChatClient
from repro.integration.qa_adapter import execute_examples
from repro.integration.source import ensure_upstream
from repro.serving.llama_launcher import build_command


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    settings = load_config(PROJECT_ROOT / "configs/local.gguf.yaml")["model"]
    settings.update(
        max_completion_tokens=256,
        timeout_s=180,
        max_retries=0,
    )
    store = RunStore(PROJECT_ROOT / "outputs/gguf-probe")
    argv = build_command(settings)
    if "--predict" not in argv and "--n-predict" not in argv:
        argv.extend(["--predict", "4096"])
    store.save_json("server_command.json", {"argv": argv, "model_was_already_local": True})
    ledger = UsageLedger(
        max_calls=10, max_completion_tokens=30000, sink=lambda row: store.append_jsonl("calls.jsonl", row)
    )
    client = ChatClient(settings, ledger)
    with httpx.Client(timeout=2, trust_env=False) as health_client:
        try:
            health_client.get("http://127.0.0.1:8000/health").raise_for_status()
        except httpx.HTTPError:
            pass
        else:
            raise RuntimeError("Port 8000 already has a service; it was left untouched")
    stdout = (store.path / "server.stdout.log").open("w", encoding="utf-8")
    stderr = (store.path / "server.stderr.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        argv, stdout=stdout, stderr=stderr, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
    )
    print(f"Probe directory: {store.path}; server PID: {process.pid}", flush=True)
    summary = {"model": settings, "training_examples_only": True, "research_comparison": False}
    try:
        with httpx.Client(timeout=3, trust_env=False) as health_client:
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"llama-server exited with code {process.returncode}; inspect {store.path}"
                    )
                try:
                    response = health_client.get("http://127.0.0.1:8000/health")
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("GGUF model did not become ready within 180 seconds")
                time.sleep(1)
            props = health_client.get("http://127.0.0.1:8000/props")
            props.raise_for_status()
            store.save_json("server_props.json", props.json())
        store.save_json("service.json", client.verify_service())
        sanity = client.generate(
            [
                {"role": "system", "content": "Return only the final answer inside <answer> tags."},
                {
                    "role": "user",
                    "content": (
                        "Context: Paris is the capital of France. "
                        "Question: What is the capital of France?"
                    ),
                },
            ],
            max_tokens=64,
            metadata={"phase": "protocol_sanity"},
        )
        summary["sanity_response"] = sanity.content
        print(f"Sanity response: {sanity.content}", flush=True)
        for dataset in ("searchqa", "hotpotqa"):
            examples = load_split(PROJECT_ROOT / "data/pilot" / dataset, "train", dataset)[:2]
            rows = execute_examples(examples, "", client, store, "no_skill", max_tokens=256)
            summary[dataset] = {
                "ids": [row["id"] for row in rows],
                "n": len(rows),
                "em": sum(row["em"] for row in rows) / len(rows),
                "f1": sum(row["f1"] for row in rows) / len(rows),
                "failed": sum(row["status"] != "ok" for row in rows),
            }
            print(f"{dataset}: {summary[dataset]}", flush=True)
        ensure_upstream()
        from graphskillevo.graph_skill import assert_graph_skill
        from graphskillevo.operators import EvolutionOperators

        from repro.integration.model_bridge import OptimizerBridge, scoped_upstream

        initial_path = (
            PROJECT_ROOT / "third_party/GraphSkillEvo/graphskillevo/initial_skills/searchqa/init_skill.md"
        )
        bridge = OptimizerBridge(client, store)
        operators = EvolutionOperators(
            env_name="searchqa", max_completion_tokens=4096, reasoning_effort=None, graph_retry_attempts=1
        )
        with scoped_upstream(bridge, selector_name="original", dataset="searchqa"):
            with bridge.phase("probe_initialization", 2):
                skills, record = operators.initialize_population(
                    n=1,
                    base_skill=initial_path.read_text(encoding="utf-8"),
                    task_context="Answer questions using supplied context. Return <answer>...</answer>.",
                )
        assert_graph_skill(skills[0])
        store.save_skill("generated_probe", skills[0], origin="real_gguf_initialization")
        store.save_json("initialization.json", record)
        summary["valid_graph_generated"] = True
        print("Real model generated a structurally valid skill graph.", flush=True)
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        client.close()
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        stdout.close()
        stderr.close()
        summary.update(usage=ledger.summary(), server_stopped=True)
        store.save_json("probe_summary.json", summary)
        print(
            json.dumps(
                {
                    "path": str(store.path),
                    "valid_graph_generated": summary.get("valid_graph_generated", False),
                    "usage": ledger.summary(),
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
