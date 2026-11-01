from copy import deepcopy

import pytest

from repro.config import DEFAULTS
from repro.inference.service import validate_llama_properties


def properties():
    return {
        "model_alias": "test",
        "total_slots": 1,
        "build_info": "b123-abcd",
        "default_generation_settings": {"n_ctx": 8192, "params": {"reasoning_format": "none"}},
    }


def settings():
    result = deepcopy(DEFAULTS["model"])
    result["name"] = "test"
    return result


def test_unreported_thinking_is_not_falsely_verified():
    info = validate_llama_properties(properties(), settings(), expected_build="b123-abcd")
    assert info["reported_enable_thinking"] is None


@pytest.mark.parametrize(
    "field,value", [("total_slots", 2), ("build_info", "different"), ("model_alias", "different")]
)
def test_wrong_service_rejected(field, value):
    props = properties()
    props[field] = value
    with pytest.raises(ValueError):
        validate_llama_properties(props, settings(), expected_build="b123-abcd")


def test_wrong_context_and_thinking_rejected():
    props = properties()
    props["default_generation_settings"]["n_ctx"] = 4096
    with pytest.raises(ValueError, match="context"):
        validate_llama_properties(props, settings())
    props = properties()
    props["chat_template_kwargs"] = {"enable_thinking": True}
    with pytest.raises(ValueError, match="thinking mode"):
        validate_llama_properties(props, settings())
