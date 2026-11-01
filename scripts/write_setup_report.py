"""Save environment checks and references to completed model verifications."""

from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from repro.config import PROJECT_ROOT, load_config
from repro.datasets.manifests import file_sha256, verify_manifest
from repro.diagnostics import collect_diagnostics


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    target = PROJECT_ROOT / "outputs/verification"
    target.mkdir(parents=True, exist_ok=True)
    native = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/check_native_qwen.py")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    native_report = json.loads(native.stdout)
    (target / "native-qwen.json").write_text(json.dumps(native_report, indent=2) + "\n", encoding="utf-8")
    diagnostics = {
        dataset: collect_diagnostics(
            PROJECT_ROOT, load_config(PROJECT_ROOT / f"configs/{dataset}.course.yaml")
        )
        for dataset in ("searchqa", "hotpotqa")
    }
    manifest_summaries = {}
    for dataset in ("searchqa", "hotpotqa"):
        root = PROJECT_ROOT / "data/processed" / dataset
        manifest = verify_manifest(root)
        manifest_summaries[dataset] = {
            "counts": {split: info["count"] for split, info in manifest["splits"].items()},
            "manifest_sha256": file_sha256(root / "split_manifest.json"),
            "source": manifest["source"],
        }
    suites = list(ET.parse(target / "pytest.xml").getroot().iter("testsuite"))
    tests = {
        key: sum(int(suite.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    report = {
        "verified_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "platform": "native_windows",
        "wsl_required": False,
        "setup_downloads_model_weights": False,
        "real_9b_experiment_run": False,
        "selected_model": load_config(PROJECT_ROOT / "config.yaml")["model"],
        "shared_config_sha256": file_sha256(PROJECT_ROOT / "config.yaml"),
        "diagnostics": diagnostics,
        "datasets": manifest_summaries,
        "native_architecture_check": native_report,
        "pytest": tests,
        "dependency_lock_sha256": file_sha256(PROJECT_ROOT / "requirements/lock.windows.txt"),
    }
    asset_manifest = PROJECT_ROOT / "models/Qwen3.5-9B/asset_manifest.json"
    if asset_manifest.is_file():
        report["non_weight_model_assets"] = json.loads(asset_manifest.read_text(encoding="utf-8"))
    inspection = target / "gguf-inspection.json"
    if inspection.is_file():
        report["existing_gguf_inspection"] = json.loads(inspection.read_text(encoding="utf-8"))
    for directory, filename, key in (
        ("gguf-probe", "probe_summary.json", "real_gguf_protocol_probe"),
        ("gguf-pilot-check", "summary.json", "real_gguf_evolution_verification"),
    ):
        completed = sorted((PROJECT_ROOT / "outputs" / directory).glob(f"*/{filename}"))
        if completed:
            report[key] = {
                "path": str(completed[-1]),
                "result": json.loads(completed[-1].read_text(encoding="utf-8")),
            }
    path = target / "setup-report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)
    print(
        json.dumps(
            {
                "infrastructure_ready": all(item["infrastructure_ready"] for item in diagnostics.values()),
                "model_ready": any(item["model_ready"] for item in diagnostics.values()),
                "tests": tests,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
