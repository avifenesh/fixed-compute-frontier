#!/usr/bin/env python3
"""Real-checkpoint quality gate for BBCM's per-row INT8 shared base."""

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
    from experiments.reflex_swiglu_lm_screen import (
        MODEL,
        MODEL_REVISION,
        TokenFile,
        evaluate,
        sha256_file,
        validate_data_ledger,
    )
except ModuleNotFoundError:  # Direct execution places experiments/ on sys.path.
    from reflex_swiglu_lm_screen import (
        MODEL,
        MODEL_REVISION,
        TokenFile,
        evaluate,
        sha256_file,
        validate_data_ledger,
    )


EXPECTED_BASELINE_NLL = 2.8295718903541565
EXPECTED_DATA_MANIFEST_SHA256 = "955cc3d3be7992840da95881450e4483b2dee4a9567b062b17e1ebe061c817a4"


def store_scale_as_bfloat16(scale: torch.Tensor, nonzero_rows: torch.Tensor) -> torch.Tensor:
    """Round scale metadata to its ledgered BF16 representation, then decode in FP32."""
    rounded = scale.to(torch.bfloat16).to(torch.float32)
    smallest_normal = torch.full_like(rounded, torch.finfo(torch.bfloat16).tiny)
    rounded = torch.where(nonzero_rows & (rounded == 0), smallest_normal, rounded)
    return torch.where(nonzero_rows, rounded, torch.ones_like(rounded))


def quantize_int8_per_output(weight: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    source = weight.detach().float()
    max_abs = source.abs().amax(dim=1, keepdim=True)
    nonzero_rows = max_abs > 0
    scale = store_scale_as_bfloat16(max_abs / 127.0, nonzero_rows)
    codes = torch.round(source / scale).clamp(-127, 127)
    decoded = codes * scale
    error = decoded - source
    return decoded, {
        "weights": source.numel(),
        "output_rows": source.shape[0],
        "mean_squared_error": float(error.square().mean()),
        "relative_l2_error": float(
            torch.linalg.vector_norm(error)
            / torch.linalg.vector_norm(source).clamp_min(torch.finfo(torch.float32).tiny)
        ),
        "max_absolute_error": float(error.abs().max()),
        "max_scale": float(scale.max()),
        "scale_storage_dtype": "bfloat16",
        "scales_exactly_bfloat16_representable": bool(
            torch.equal(scale, scale.to(torch.bfloat16).to(torch.float32))
        ),
    }


def quantize_model_ffns(model: torch.nn.Module) -> dict[str, Any]:
    records = []
    for layer_index, layer in enumerate(model.model.layers):
        for projection_name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(layer.mlp, projection_name)
            decoded, statistics = quantize_int8_per_output(projection.weight)
            projection.weight.data.copy_(decoded.to(dtype=projection.weight.dtype))
            records.append({"layer": layer_index, "projection": projection_name, **statistics})
    total_weights = sum(record["weights"] for record in records)
    weighted_mse = sum(record["mean_squared_error"] * record["weights"] for record in records) / total_weights
    return {
        "matrices": len(records),
        "weights": total_weights,
        "weighted_mean_squared_error": weighted_mse,
        "max_relative_l2_error": max(record["relative_l2_error"] for record in records),
        "max_absolute_error": max(record["max_absolute_error"] for record in records),
        "records": records,
    }


def paired_difference(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    candidate_losses = np.asarray(candidate["per_batch_loss"], dtype=np.float64)
    baseline_losses = np.asarray(baseline["per_batch_loss"], dtype=np.float64)
    difference = candidate_losses - baseline_losses
    standard_error = float(difference.std(ddof=1) / math.sqrt(difference.size))
    mean = float(difference.mean())
    return {
        "candidate_minus_baseline_mean": mean,
        "standard_error": standard_error,
        "lower_95": mean - 1.96 * standard_error,
        "upper_95": mean + 1.96 * standard_error,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("frozen gate requires H100")
    versions_valid = (
        torch.__version__ == "2.5.1+cu124"
        and torch.version.cuda == "12.4"
        and transformers.__version__ == "4.57.6"
    )
    if not versions_valid:
        raise RuntimeError("frozen software versions do not match")
    if sha256_file(args.data_manifest) != EXPECTED_DATA_MANIFEST_SHA256:
        raise RuntimeError("data manifest hash drift")
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
        raise RuntimeError("frozen evaluation geometry drift")
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
    quantization = quantize_model_ffns(model)
    decoded = evaluate(model, validation_file, args.batches, args.batch_size, torch.device("cuda"))
    relative_degradation = (decoded["loss"] - baseline["loss"]) / baseline["loss"]
    interval = paired_difference(decoded, baseline)
    finite = all(
        math.isfinite(value)
        for value in (
            baseline["loss"],
            decoded["loss"],
            relative_degradation,
            quantization["weighted_mean_squared_error"],
            quantization["max_relative_l2_error"],
            quantization["max_absolute_error"],
        )
    )
    gates = {
        "protocol_valid": protocol_valid and versions_valid and architecture_valid and data_ledger["valid"],
        "baseline_matches_frozen_anchor": abs(baseline["loss"] - EXPECTED_BASELINE_NLL) <= 1e-7,
        "int8_relative_nll_degradation_at_most_0p2_percent": relative_degradation <= 0.002,
        "finite_and_exactly_90_matrices": finite and quantization["matrices"] == 90,
        "all_scales_use_ledgered_bfloat16_storage": all(
            record["scale_storage_dtype"] == "bfloat16"
            and record["scales_exactly_bfloat16_representable"]
            for record in quantization["records"]
        ),
    }
    preregistration = Path("results/bbcm-int8-shared-base-h100-v2-preregistration.md")
    return {
        "candidate": "BBCM INT8 shared base",
        "scope": "real-checkpoint exact-BF16-scale decoded-weight quality gate; no routed deltas or runtime claim",
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
        "int8_decoded": decoded,
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
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--output", type=Path, default=Path("results/bbcm-int8-shared-base-h100-v2.json"))
    args = parser.parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": payload["gates"], "relative_nll_degradation": payload["relative_nll_degradation"], "all_gates_pass": payload["all_gates_pass"]}, indent=2))
    raise SystemExit(0 if payload["all_gates_pass"] else 1)


if __name__ == "__main__":
    main()
