"""Verify Qwen3.5 CUDA kernels with a tiny random in-memory model, without loading weights."""

from __future__ import annotations

import json
import os
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def main() -> None:
    import torch
    import transformers
    from transformers import AutoModelForMultimodalLM, Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this architecture check")
    torch.manual_seed(42)
    text = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [3, 3, 2],
        },
    )
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=64,
        num_position_embeddings=16,
    )
    config = Qwen3_5Config(
        text_config=text,
        vision_config=vision,
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        vision_end_token_id=123,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = (
        AutoModelForMultimodalLM.from_config(config, dtype=torch.bfloat16, attn_implementation="sdpa")
        .to("cuda")
        .eval()
    )
    input_ids = torch.tensor([[1, 13, 14, 15]], device="cuda")
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=3,
            min_new_tokens=3,
            do_sample=False,
            use_cache=True,
        )
    torch.cuda.synchronize()
    if output.shape != (1, 7):
        raise RuntimeError(f"Unexpected generation shape: {tuple(output.shape)}")
    print(
        json.dumps(
            {
                "check": "tiny_random_qwen3_5_cuda_generation",
                "status": "ok",
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(0),
                "cuda": torch.version.cuda,
                "dtype": str(next(model.parameters()).dtype),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "linear_attention_layers": 3,
                "full_attention_layers": 1,
                "generated_tokens": int(output.shape[1] - input_ids.shape[1]),
                "use_cache": True,
                "elapsed_s": round(time.perf_counter() - started, 4),
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
                "real_9b_weights_tested": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
