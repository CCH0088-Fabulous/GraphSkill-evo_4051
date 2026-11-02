from copy import deepcopy

import pytest

from repro.config import DEFAULTS, load_config, validate_config


def test_pilot_configs_load():
    for name in (
        "searchqa.pilot",
        "hotpotqa.pilot",
        "searchqa.course",
        "hotpotqa.course",
        "offline.searchqa",
        "offline.hotpotqa",
    ):
        load_config(f"configs/{name}.yaml")


def test_course_candidate_budget_and_gpu_reference_are_separate():
    cfg = deepcopy(DEFAULTS)
    cfg["budget"]["gpu_hours_reference"] = 200
    validate_config(cfg)
    cfg["evolution"]["new_candidates_per_round"] = 3
    with pytest.raises(ValueError, match="2 new candidates"):
        validate_config(cfg)


def test_provisional_hotpot_protocol_has_separate_identity_and_budgets():
    legacy = load_config("configs/hotpotqa.course.yaml")
    current = load_config("config.yaml", presets=["course", "hotpotqa.provisional"])
    assert legacy["experiment"]["profile"] == "course"
    assert legacy["data"]["manifest"] == "manifests/data-v1/hotpotqa.json"
    assert current["experiment"]["profile"] == "custom"
    assert not current["experiment"]["allow_test"]
    assert current["data"]["manifest"] == "manifests/hotpotqa-stratified-v2/hotpotqa.json"
    assert current["preparation"]["hotpotqa_test_count"] == 1000
    assert current["model"] == legacy["model"]
    assert current["evolution"] == legacy["evolution"]
    assert current["budget"]["max_calls"] == 5 * legacy["budget"]["max_calls"]
    assert current["budget"]["max_completion_tokens"] == 5 * legacy["budget"]["max_completion_tokens"]


@pytest.mark.parametrize("key,value", [("hotpotqa_train_per_level", 0), ("hotpotqa_test_count", True)])
def test_stratified_preparation_counts_require_positive_integers(key, value):
    config = deepcopy(DEFAULTS)
    config["preparation"][key] = value
    with pytest.raises(ValueError, match=key):
        validate_config(config)


def test_typo_override_rejected():
    with pytest.raises(ValueError, match="Unknown override"):
        load_config("configs/searchqa.pilot.yaml", ["model.misspelled=1"])


def test_improved_condition_cannot_silently_run_original_selector():
    cfg = deepcopy(DEFAULTS)
    cfg["experiment"]["condition"] = "improved"
    with pytest.raises(ValueError, match="not a fallback"):
        validate_config(cfg)


def test_shared_base_deep_merge_and_relative_file_resolution(tmp_path, monkeypatch):
    shared = tmp_path / "shared.yaml"
    shared.write_text("model:\n  name: shared-model\n  temperature: 0.3\nbudget:\n  max_calls: 30\n")
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "middle.yaml").write_text("_base_: ../shared.yaml\nmodel:\n  timeout_s: 75\n")
    experiment = profiles / "experiment.yaml"
    experiment.write_text("_base_: middle.yaml\nmodel:\n  temperature: 0.0\n")
    monkeypatch.chdir(tmp_path.parent)
    result = load_config(experiment)
    assert result["model"]["name"] == "shared-model"
    assert result["model"]["temperature"] == 0.0
    assert result["model"]["timeout_s"] == 75
    assert result["model"]["max_retries"] == DEFAULTS["model"]["max_retries"]
    assert result["budget"]["max_calls"] == 30
    assert "_base_" not in result


def test_named_presets_compose_from_shared_config_and_cli_can_override(tmp_path):
    shared = tmp_path / "shared.yaml"
    shared.write_text(
        "model:\n  name: shared-model\npresets:\n"
        "  course:\n    evolution:\n      generations: 3\n"
        "  hotpotqa:\n    data:\n      dataset: hotpotqa\n      split_dir: data/hotpotqa\n",
        encoding="utf-8",
    )
    child = tmp_path / "profile.yaml"
    child.write_text("_base_: shared.yaml\n_presets_: [course, hotpotqa]\n", encoding="utf-8")
    config = load_config(child, ["evolution.generations=2"])
    assert config["model"]["name"] == "shared-model"
    assert config["data"]["dataset"] == "hotpotqa"
    assert config["data"]["split_dir"] == "data/hotpotqa"
    assert config["evolution"]["generations"] == 2
    assert "presets" not in config and "_presets_" not in config
    assert load_config(shared, presets=["course", "hotpotqa"])["evolution"]["generations"] == 3


def test_unknown_or_malformed_presets_are_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  name: x\n_presets_: [missing]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown preset"):
        load_config(path)
    path.write_text("model:\n  name: x\npresets:\n  broken:\n    model:\n      typoo: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="typoo"):
        load_config(path)
    path.write_text("model:\n  name: x\n_presets_: course\n", encoding="utf-8")
    with pytest.raises(ValueError, match="_presets_ must be a list"):
        load_config(path)


def test_cli_can_select_root_presets_without_a_separate_config_file(capsys):
    import json

    from repro.cli import main

    assert main([
        "show-config", "--config", "config.yaml", "--preset", "course", "--preset", "hotpotqa.course",
        "--set", "experiment.seed=17",
    ]) == 0
    config = json.loads(capsys.readouterr().out)
    assert config["data"]["dataset"] == "hotpotqa"
    assert config["data"]["manifest"] == "manifests/data-v1/hotpotqa.json"
    assert config["evolution"]["generations"] == 3
    assert config["experiment"]["seed"] == 17
    assert main([
        "show-config", "--config", "config.yaml", "--preset", "course", "--preset", "hotpotqa.course",
        "--set", "data.manifest=null",
    ]) == 2
    assert "data manifest cannot be overridden" in capsys.readouterr().out
    assert main([
        "show-config", "--config", "config.yaml", "--preset", "course", "--preset", "hotpotqa.course",
        "--preset", "searchqa.pilot",
    ]) == 2
    assert "data manifest cannot be overridden" in capsys.readouterr().out


def test_child_config_cannot_disable_a_selected_data_reference(tmp_path):
    parent = tmp_path / "parent.yaml"
    parent.write_text("data:\n  manifest: fixed.json\n", encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text("_base_: parent.yaml\ndata:\n  manifest: null\n", encoding="utf-8")
    with pytest.raises(ValueError, match="data manifest cannot be overridden"):
        load_config(child)


@pytest.mark.parametrize("dataset", ["searchqa", "hotpotqa"])
def test_existing_entrypoints_resolve_central_presets(dataset):
    course = load_config(f"configs/{dataset}.course.yaml")
    pilot = load_config(f"configs/{dataset}.pilot.yaml")
    offline = load_config(f"configs/offline.{dataset}.yaml")
    assert course["data"]["manifest"] == f"manifests/data-v1/{dataset}.json"
    assert (course["evolution"]["generations"], course["budget"]["max_calls"]) == (3, 600)
    assert pilot["data"]["split_dir"] == f"data/pilot/{dataset}"
    assert offline["experiment"]["profile"] == "synthetic"
    assert offline["data"]["split_dir"] == f"tests/fixtures/{dataset}"


@pytest.mark.parametrize("dataset", ["searchqa", "hotpotqa"])
@pytest.mark.parametrize("profile", ["course", "pilot", "offline"])
def test_existing_entrypoints_match_direct_root_presets(dataset, profile):
    if profile == "course":
        names = ["course", f"{dataset}.course"]
    elif profile == "offline":
        names = ["offline", f"{dataset}.offline"]
    else:
        names = [f"{dataset}.pilot"]
    entry = f"offline.{dataset}" if profile == "offline" else f"{dataset}.{profile}"
    assert load_config(f"configs/{entry}.yaml") == load_config("config.yaml", presets=names)


def test_declared_data_identity_cannot_be_disabled_from_cli(tmp_path):
    path = tmp_path / "formal.yaml"
    path.write_text("data:\n  manifest: recorded-data.json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="data manifest cannot be overridden"):
        load_config(path, ["data.manifest=null"])


def test_environment_and_cli_override_inheritance_in_order(tmp_path, monkeypatch):
    shared = tmp_path / "shared.yaml"
    shared.write_text("model:\n  name: shared-model\n")
    child = tmp_path / "experiment.yaml"
    child.write_text("_base_: shared.yaml\nmodel:\n  name: experiment-model\n")
    monkeypatch.setenv("REPRO_MODEL_NAME", "environment-model")
    assert load_config(child)["model"]["name"] == "environment-model"
    assert load_config(child, ["model.name=cli-model"])["model"]["name"] == "cli-model"


@pytest.mark.parametrize("cycle", ["self", "indirect"])
def test_inheritance_cycle_fails_with_chain(tmp_path, cycle):
    first = tmp_path / "first.yaml"
    first.write_text("_base_: first.yaml\n" if cycle == "self" else "_base_: second.yaml\n")
    if cycle == "indirect":
        (tmp_path / "second.yaml").write_text("_base_: first.yaml\n")
    with pytest.raises(ValueError, match="inheritance cycle"):
        load_config(first)


@pytest.mark.parametrize("contents", ["", "# comment only\n", "{}\n", "[]\n"])
def test_empty_or_non_mapping_config_is_rejected(tmp_path, contents):
    config = tmp_path / "empty.yaml"
    config.write_text(contents)
    with pytest.raises(ValueError, match="mapping|empty"):
        load_config(config)


def test_bad_option_in_base_cannot_be_hidden_by_experiment(tmp_path):
    (tmp_path / "shared.yaml").write_text("model:\n  max_contex_tokens: 1024\n")
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text("_base_: shared.yaml\nmodel:\n  max_context_tokens: 8192\n")
    with pytest.raises(ValueError, match="max_contex_tokens"):
        load_config(experiment)


@pytest.mark.parametrize("value", ["null", "[]", "false", "''"])
def test_base_must_be_one_nonempty_path_string(tmp_path, value):
    path = tmp_path / "config.yaml"
    path.write_text(f"_base_: {value}\n")
    with pytest.raises(ValueError, match="_base_ must be a nonempty file path string"):
        load_config(path)


def test_standalone_config_keeps_builtin_defaults_without_implicit_project_base(tmp_path):
    path = tmp_path / "standalone.yaml"
    path.write_text("experiment:\n  seed: 9\n")
    assert load_config(path)["model"] == DEFAULTS["model"]


@pytest.mark.parametrize("field,value", [("backend", "unknown"), ("max_context_tokens", 0)])
def test_backend_and_context_are_validated(field, value):
    config = deepcopy(DEFAULTS)
    config["model"][field] = value
    with pytest.raises(ValueError, match=field):
        validate_config(config)


@pytest.mark.parametrize(
    "section,key", [("model", "max_completion_tokens"), ("evolution", "operator_max_completion_tokens")]
)
def test_output_limit_must_leave_room_for_input(section, key):
    config = deepcopy(DEFAULTS)
    config[section][key] = config["model"]["max_context_tokens"]
    with pytest.raises(ValueError, match="leave room for input"):
        validate_config(config)


def test_dotenv_precedence_is_shared_by_cli_and_server(tmp_path, monkeypatch):
    monkeypatch.setattr("repro.config.PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("REPRO_MODEL_NAME", raising=False)
    (tmp_path / ".env").write_text("REPRO_MODEL_NAME=dotenv-model\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("model:\n  name: yaml-model\n", encoding="utf-8")
    assert load_config(tmp_path / "config.yaml")["model"]["name"] == "dotenv-model"
    monkeypatch.setenv("REPRO_MODEL_NAME", "process-model")
    assert load_config(tmp_path / "config.yaml")["model"]["name"] == "process-model"
    assert load_config(tmp_path / "config.yaml", ["model.name=cli-model"])["model"]["name"] == "cli-model"
