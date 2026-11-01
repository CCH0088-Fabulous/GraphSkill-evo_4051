"""Load and execute a local Transformers model without hub access."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repro.diagnostics import validate_model_directory


@dataclass(frozen=True)
class Completion:
    content: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


class ContextLimitError(ValueError):
    """The request exceeds the configured input-plus-output context budget."""


class LocalTextModel:
    """One resident model, one serialized generation stream, and no implicit truncation."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        torch_module: Any,
        model_name: str,
        max_model_len: int = 16384,
        enable_thinking: bool = False,
        local_path: Path | None = None,
    ) -> None:
        if max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch_module
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.enable_thinking = enable_thinking
        self.local_path = local_path
        self._lock = threading.Lock()

    @classmethod
    def from_directory(
        cls,
        local_path: Path,
        *,
        model_name: str = "Qwen/Qwen3.5-9B",
        max_model_len: int = 16384,
        enable_thinking: bool = False,
    ) -> LocalTextModel:
        local_path = Path(local_path).expanduser().resolve()
        complete, detail = validate_model_directory(local_path)
        if not complete:
            raise ValueError(f"Incomplete local model: {detail}")
        # Set before importing either library. from_pretrained calls also explicitly forbid network access.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        import torch
        import transformers

        if not torch.cuda.is_available():
            raise RuntimeError("A working CUDA PyTorch installation is required for this BF16 server")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The configured CUDA GPU does not support BF16")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(local_path), local_files_only=True, trust_remote_code=False
        )
        config = transformers.AutoConfig.from_pretrained(
            str(local_path), local_files_only=True, trust_remote_code=False
        )
        # Qwen3.5 uses the multimodal conditional-generation class even for text-only prompts.
        if getattr(config, "vision_config", None) is not None:
            factory = getattr(transformers, "AutoModelForMultimodalLM", None)
            if factory is None:
                factory = transformers.AutoModelForImageTextToText
        else:
            factory = transformers.AutoModelForCausalLM
        text_config = getattr(config, "text_config", config)
        actual_limit = getattr(text_config, "max_position_embeddings", max_model_len)
        if actual_limit and max_model_len > int(actual_limit):
            raise ValueError(f"Requested context {max_model_len} exceeds model limit {actual_limit}")
        model = factory.from_pretrained(
            str(local_path),
            config=config,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
        ).eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            torch_module=torch,
            model_name=model_name,
            max_model_len=max_model_len,
            enable_thinking=enable_thinking,
            local_path=local_path,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 50,
        seed: int | None = None,
    ) -> Completion:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if temperature < 0 or not 0 < top_p <= 1 or top_k < 0:
            raise ValueError("Invalid temperature, top_p or top_k")
        # Tokenization is serialized as well; some tokenizer implementations maintain mutable state.
        with self._lock:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=self.enable_thinking,
            )
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            if prompt_tokens + max_new_tokens > self.max_model_len:
                raise ContextLimitError(
                    f"Input has {prompt_tokens} tokens and requests {max_new_tokens} output tokens; "
                    f"the total exceeds context limit {self.max_model_len}. No input was truncated."
                )
            inputs = inputs.to(self.model.device)
            generate_options: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id,
            }
            if temperature > 0:
                generate_options.update(temperature=temperature, top_p=top_p, top_k=top_k)
                if seed is not None:
                    self.torch.manual_seed(seed)
                    self.torch.cuda.manual_seed_all(seed)
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, **generate_options)
            generated = output[0, prompt_tokens:]
            completion_tokens = int(generated.shape[-1])
            eos_ids = self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id
            if isinstance(eos_ids, int):
                eos_ids = [eos_ids]
            ended = completion_tokens > 0 and int(generated[-1]) in (eos_ids or [])
            finish_reason = "stop" if ended or completion_tokens < max_new_tokens else "length"
            return Completion(
                content=self.tokenizer.decode(generated, skip_special_tokens=True),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
            )
