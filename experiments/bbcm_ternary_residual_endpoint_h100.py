#!/usr/bin/env python3
"""H100 endpoint gate for BBCM's INT8 base plus 2-bit ternary residual."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM

try:
    from experiments.bbcm_int8_shared_base_h100 import (
        EXPECTED_BASELINE_NLL,
        EXPECTED_DATA_MANIFEST_SHA256,
        paired_difference,
        quantize_int8_per_output,
        store_scale_as_bfloat16,
    )
    from experiments.reflex_swiglu_lm_screen import (
        MODEL,
        MODEL_REVISION,
        TokenFile,
        evaluate,
        sha256_file,
        validate_data_ledger,
    )
except ModuleNotFoundError:
    from bbcm_int8_shared_base_h100 import (
        EXPECTED_BASELINE_NLL,
        EXPECTED_DATA_MANIFEST_SHA256,
        paired_difference,
        quantize_int8_per_output,
        store_scale_as_bfloat16,
    )
    from reflex_swiglu_lm_screen import (
        MODEL,
        MODEL_REVISION,
        TokenFile,
        evaluate,
        sha256_file,
        validate_data_ledger,
    )


def quantize_int8_plus_ternary_residual(
    weight: torch.Tensor,
    lloyd_iterations: int = 4,
) -> tuple[torch.Tensor, dict[str, Any]]:
    source = weight.detach().float()
    base, base_stats = quantize_int8_per_output(source)
    residual = source - base
    residual_max = residual.abs().amax(dim=1, keepdim=True)
    nonzero_rows = residual_max > 0
    scale = store_scale_as_bfloat16((2.0 / 3.0) * residual_max, nonzero_rows)
    codes = torch.zeros_like(residual)
    for _ in range(lloyd_iterations):
        assigned = residual.abs() >= scale / 2.0
        codes = torch.sign(residual) * assigned
        counts = assigned.sum(dim=1, keepdim=True).clamp_min(1)
        updated_scale = (residual.abs() * assigned).sum(dim=1, keepdim=True) / counts
        scale = store_scale_as_bfloat16(updated_scale, nonzero_rows)
    assigned = residual.abs() >= scale / 2.0
    codes = torch.sign(residual) * assigned
    decoded = base + codes * scale
    error = decoded - source
    denominator = torch.linalg.vector_norm(source).clamp_min(torch.finfo(torch.float32).tiny)
    return decoded, {
        "weights": source.numel(),
        "output_rows": source.shape[0],
        "base_relative_l2_error": base_stats["relative_l2_error"],
        "combined_relative_l2_error": float(torch.linalg.vector_norm(error) / denominator),
        "combined_mean_squared_error": float(error.square().mean()),
        "max_absolute_error": float(error.abs().max()),
        "zero_code_fraction": float((codes == 0).float().mean()),
        "positive_code_fraction": float((codes > 0).float().mean()),
        "negative_code_fraction": float((codes < 0).float().mean()),
        "max_delta_scale": float(scale.max()),
        "base_scale_storage_dtype": base_stats["scale_storage_dtype"],
        "base_scales_exactly_bfloat16_representable": base_stats[
            "scales_exactly_bfloat16_representable"
        ],
        "delta_scale_storage_dtype": "bfloat16",
        "delta_scales_exactly_bfloat16_representable": bool(
            torch.equal(scale, scale.to(torch.bfloat16).to(torch.float32))
        ),
    }


def quantize_model(model: torch.nn.Module) -> dict[str, Any]:
    records = []
    for layer_index, layer in enumerate(model.model.layers):
        for projection_name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(layer.mlp, projection_name)
            decoded, statistics = quantize_int8_plus_ternary_residual(projection.weight)
            projection.weight.data.copy_(decoded.to(projection.weight.dtype))
            records.append({"layer": layer_index, "projection": projection_name, **statistics})
    total_weights = sum(record["weights"] for record in records)
    return {
        "matrices": len(records),
        "weights": total_weights,
        "combined_error_lower_for_every_matrix": all(
            record["combined_relative_l2_error"] < record["base_relative_l2_error"]
            for record in records
        ),
        "weighted_combined_mse": sum(
            record["combined_mean_squared_error"] * record["weights"] for record in records
        ) / total_weights,
        "max_combined_relative_l2_error": max(
            record["combined_relative_l2_error"] for record in records
        ),
        "zero_code_fraction": sum(
            record["zero_code_fraction"] * record["weights"] for record in records
        ) / total_weights,
        "positive_code_fraction": sum(
            record["positive_code_fraction"] * record["weights"] for record in records
        ) / total_weights,
        "negative_code_fraction": sum(
            record["negative_code_fraction"] * record["weights"] for record in records
        ) / total_weights,
        "records": records,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    versions_valid = (
        torch.__version__ == "2.5.1+cu124"
        and torch.version.cuda == "12.4"
        and transformers.__version__ == "4.57.6"
    )
    if not versions_valid:
        raise RuntimeError("software drift")
    if sha256_file(args.data_manifest) != EXPECTED_DATA_MANIFEST_SHA256:
        raise RuntimeError("manifest drift")
    data_ledger = validate_data_ledger(
        args.data_manifest, args.train_file, args.validation_file, args.sequence_length
    )
    validation_file = TokenFile(args.validation_file, args.sequence_length)
    protocol_valid = (
        args.sequence_length == 512
        and args.batch_size == 32
        and args.batches == 128
        and validation_file.sequence_count == 4096
    )
    if not protocol_valid:
        raise RuntimeError("evaluation geometry drift")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=MODEL_REVISION,
        dtype=torch.float32,
        attn_implementation="sdpa",
    ).to("cuda")
    model.config.use_cache = False
    architecture_valid = (
        len(model.model.layers) == 30
        and model.config.hidden_size == 576
        and model.config.intermediate_size == 1536
    )
    baseline = evaluate(model, validation_file, args.batches, args.batch_size, torch.device("cuda"))
    quantization = quantize_model(model)
    decoded = evaluate(model, validation_file, args.batches, args.batch_size, torch.device("cuda"))
    relative_degradation = (decoded["loss"] - baseline["loss"]) / baseline["loss"]
    interval = paired_difference(decoded, baseline)
    finite_values = [
        baseline["loss"],
        decoded["loss"],
        relative_degradation,
        quantization["weighted_combined_mse"],
        quantization["max_combined_relative_l2_error"],
        quantization["zero_code_fraction"],
        quantization["positive_code_fraction"],
        quantization["negative_code_fraction"],
    ]
    gates = {
        "protocol_valid": protocol_valid and versions_valid and architecture_valid and data_ledger["valid"],
        "baseline_matches_frozen_anchor": abs(baseline["loss"] - EXPECTED_BASELINE_NLL) <= 1e-7,
        "combined_relative_nll_degradation_at_most_0p05_percent": relative_degradation <= 0.0005,
        "combined_error_lower_for_all_90_matrices":
            quantization["matrices"] == 90
            and quantization["combined_error_lower_for_every_matrix"],
        "finite_with_zero_and_nonzero_codes":
            all(math.isfinite(value) for value in finite_values)
            and 0.0 < quantization["zero_code_fraction"] < 1.0
            and quantization["positive_code_fraction"] > 0.0
            and quantization["negative_code_fraction"] > 0.0,
        "all_base_and_delta_scales_use_ledgered_bfloat16_storage": all(
            record["base_scale_storage_dtype"] == "bfloat16"
            and record["base_scales_exactly_bfloat16_representable"]
            and record["delta_scale_storage_dtype"] == "bfloat16"
            and record["delta_scales_exactly_bfloat16_representable"]
            for record in quantization["records"]
        ),
    }
    preregistration = Path("results/bbcm-ternary-residual-endpoint-h100-v2-preregistration.md")
    return {
        "candidate": "BBCM INT8 plus ternary residual endpoint",
        "scope": "exact-BF16-scale pretrained endpoint reconstruction gate only",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "preregistration_sha256": hashlib.sha256(preregistration.read_bytes()).hexdigest(),
        "data_manifest_sha256": sha256_file(args.data_manifest),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "data_ledger": data_ledger,
        "baseline": baseline,
        "decoded": decoded,
        "relative_nll_degradation": relative_degradation,
        "paired_interval": interval,
        "quantization": quantization,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("data/reflex-lm-screen/train.uint16.bin"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/reflex-lm-screen/validation.uint16.bin"))
    parser.add_argument("--data-manifest", type=Path, default=Path("results/reflex-swiglu-lm-data-manifest.json"))
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batches", type=int, default=128)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--output", type=Path, default=Path("results/bbcm-ternary-residual-endpoint-h100-v2.json"))
    args = parser.parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": payload["gates"], "relative_nll_degradation": payload["relative_nll_degradation"], "all_gates_pass": payload["all_gates_pass"]}, indent=2))
    raise SystemExit(0 if payload["all_gates_pass"] else 1)


if __name__ == "__main__":
    main()
