#!/usr/bin/env python3
"""Fatal real-checkpoint endpoint for route-remapped W4 codes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM

from experiments.reflex_swiglu_lm_screen import (
    MODEL,
    MODEL_REVISION,
    TokenFile,
    evaluate,
    sha256_file,
    validate_data_ledger,
)


OUTPUT = Path("results/route-remapped-w4-endpoint.json")
PREREGISTRATION = Path("results/route-remapped-w4-endpoint-preregistration.md")
MANIFEST = Path("results/route-remapped-w4-endpoint-integrity-manifest.json")
DATA_MANIFEST = Path("results/reflex-swiglu-lm-data-manifest.json")
TRAIN_FILE = Path("data/reflex-lm-screen/train.uint16.bin")
VALIDATION_FILE = Path("data/reflex-lm-screen/validation.uint16.bin")
EXPECTED_BASELINE = 2.8295718897134066
GROUP_SIZE = 128
EXPECTED_LEDGER = {
    "matrices": 90,
    "weights": 79_626_240,
    "groups": 668_160,
    "code_bits": 318_504_960,
    "scale_bits": 10_690_560,
    "resident_bits": 329_195_520,
    "bf16_baseline_bits": 1_274_019_840,
}

DEPENDENCIES = {
    "endpoint_source": Path(__file__),
    "preregistration": PREREGISTRATION,
    "reflex_source": Path("experiments/reflex_swiglu_lm_screen.py"),
    "data_manifest": DATA_MANIFEST,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_integrity() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text())
    checks = {
        name: manifest.get(name + "_sha256") == file_sha256(path)
        for name, path in DEPENDENCIES.items()
    }
    checks.update({
        "python_version": manifest.get("python_version") == platform.python_version(),
        "numpy_version": manifest.get("numpy_version") == np.__version__,
        "torch_version": manifest.get("torch_version") == torch.__version__,
        "cuda_version": manifest.get("cuda_version") == torch.version.cuda,
        "transformers_version": manifest.get("transformers_version") == transformers.__version__,
    })
    if not all(checks.values()):
        raise ValueError(f"invalid route-remapped W4 integrity: {checks}")
    return {
        "valid": True, "checks": checks,
        "manifest_sha256": file_sha256(MANIFEST),
    }


def bf16_scale(maximum: torch.Tensor) -> torch.Tensor:
    nonzero = maximum > 0
    rounded = (maximum / 7.0).to(torch.bfloat16).to(torch.float32)
    smallest = torch.full_like(rounded, torch.finfo(torch.bfloat16).tiny)
    return torch.where(nonzero, torch.maximum(rounded, smallest), torch.ones_like(rounded))


def all_numeric_values_finite(value: Any) -> bool:
    """Recursively reject non-finite numeric evidence; ignore labels/booleans."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float, np.number)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(all_numeric_values_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_numeric_values_finite(item) for item in value)
    return True


def quantize_w4_group128(weight: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    source = weight.detach().float()
    rows, columns = source.shape
    decoded_parts = []
    records = []
    for start in range(0, columns, GROUP_SIZE):
        stop = min(start + GROUP_SIZE, columns)
        block = source[:, start:stop]
        scale = bf16_scale(block.abs().amax(dim=1, keepdim=True))
        codes = torch.round(block / scale).clamp(-7, 7)
        decoded = codes * scale
        error = decoded - block
        decoded_parts.append(decoded)
        records.append({
            "start": start,
            "stop": stop,
            "values": block.numel(),
            "scales": rows,
            "mse": float(error.square().mean()),
            "relative_l2": float(
                torch.linalg.vector_norm(error)
                / torch.linalg.vector_norm(block).clamp_min(
                    torch.finfo(torch.float32).tiny
                )
            ),
            "scale_bf16_roundtrip": bool(
                torch.equal(scale, scale.to(torch.bfloat16).to(torch.float32))
            ),
            "codes_are_exact_integers": bool(torch.equal(codes, codes.round())),
            "codes_are_finite": bool(torch.isfinite(codes).all()),
            "minimum_code": int(codes.min()),
            "maximum_code": int(codes.max()),
            "scales_are_finite_and_positive": bool(
                torch.isfinite(scale).all() and (scale > 0).all()
            ),
            "all_finite": bool(
                torch.isfinite(block).all()
                and torch.isfinite(scale).all()
                and torch.isfinite(codes).all()
                and torch.isfinite(decoded).all()
                and torch.isfinite(error).all()
            ),
        })
    decoded_weight = torch.cat(decoded_parts, dim=1)
    error = decoded_weight - source
    scale_count = sum(record["scales"] for record in records)
    return decoded_weight, {
        "shape": [rows, columns],
        "weights": source.numel(),
        "groups": scale_count,
        "code_bits": 4 * source.numel(),
        "scale_bits": 16 * scale_count,
        "resident_bits": 4 * source.numel() + 16 * scale_count,
        "mean_squared_error": float(error.square().mean()),
        "relative_l2_error": float(
            torch.linalg.vector_norm(error)
            / torch.linalg.vector_norm(source).clamp_min(
                torch.finfo(torch.float32).tiny
            )
        ),
        "max_absolute_error": float(error.abs().max()),
        "all_scales_bf16_roundtrip": all(
            record["scale_bf16_roundtrip"] for record in records
        ),
        "all_codes_valid": all(
            record["codes_are_finite"]
            and record["codes_are_exact_integers"]
            and -7 <= record["minimum_code"] <= record["maximum_code"] <= 7
            for record in records
        ),
        "all_scales_finite_and_positive": all(
            record["scales_are_finite_and_positive"] for record in records
        ),
        "all_values_finite": all(record["all_finite"] for record in records),
        "group_count_exact": scale_count == rows * math.ceil(columns / GROUP_SIZE),
        "group_records": records,
    }


def validate_topology(records: list[dict[str, Any]]) -> dict[str, Any]:
    expected_shapes = {
        "gate_proj": [1536, 576],
        "up_proj": [1536, 576],
        "down_proj": [576, 1536],
    }
    expected_keys = {
        (layer, projection) for layer in range(30) for projection in expected_shapes
    }
    actual_keys = {(record["layer"], record["projection"]) for record in records}
    matrix_checks = []
    for record in records:
        rows, columns = record["shape"]
        expected_groups = [
            {
                "start": start,
                "stop": min(start + GROUP_SIZE, columns),
                "values": rows * (min(start + GROUP_SIZE, columns) - start),
                "scales": rows,
            }
            for start in range(0, columns, GROUP_SIZE)
        ]
        observed_groups = [
            {key: group[key] for key in ("start", "stop", "values", "scales")}
            for group in record["group_records"]
        ]
        matrix_checks.append(
            record["projection"] in expected_shapes
            and record["shape"] == expected_shapes.get(record["projection"])
            and record["weights"] == rows * columns
            and observed_groups == expected_groups
            and sum(group["values"] for group in record["group_records"])
            == record["weights"]
            and record["groups"] == rows * len(expected_groups)
        )
    return {
        "exact_matrix_key_set": len(records) == len(expected_keys) and actual_keys == expected_keys,
        "exact_shapes_and_group_boundaries": len(matrix_checks) == 90 and all(matrix_checks),
        "gate_up_tail_width_is_64": all(
            record["group_records"][-1]["stop"]
            - record["group_records"][-1]["start"] == 64
            for record in records if record["projection"] in {"gate_proj", "up_proj"}
        ),
        "gate_up_groups_per_row_is_5": all(
            len(record["group_records"]) == 5
            for record in records if record["projection"] in {"gate_proj", "up_proj"}
        ),
        "down_groups_per_row_is_12": all(
            len(record["group_records"]) == 12
            for record in records if record["projection"] == "down_proj"
        ),
    }


def quantize_model(model: torch.nn.Module) -> dict[str, Any]:
    records = []
    for layer_index, layer in enumerate(model.model.layers):
        for projection_name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(layer.mlp, projection_name)
            decoded, statistics = quantize_w4_group128(projection.weight)
            projection.weight.data.copy_(decoded.to(projection.weight.dtype))
            records.append({
                "layer": layer_index,
                "projection": projection_name,
                **statistics,
            })
    weights = sum(record["weights"] for record in records)
    resident_bits = sum(record["resident_bits"] for record in records)
    topology = validate_topology(records)
    return {
        "matrices": len(records),
        "weights": weights,
        "groups": sum(record["groups"] for record in records),
        "code_bits": 4 * weights,
        "scale_bits": sum(record["scale_bits"] for record in records),
        "resident_bits": resident_bits,
        "resident_bits_per_weight": resident_bits / weights,
        "bf16_baseline_bits": 16 * weights,
        "resident_fraction_of_bf16": resident_bits / (16 * weights),
        "weighted_mse": sum(
            record["mean_squared_error"] * record["weights"] for record in records
        ) / weights,
        "max_relative_l2_error": max(record["relative_l2_error"] for record in records),
        "max_absolute_error": max(record["max_absolute_error"] for record in records),
        "all_scales_bf16_roundtrip": all(
            record["all_scales_bf16_roundtrip"] for record in records
        ),
        "all_codes_valid": all(record["all_codes_valid"] for record in records),
        "all_scales_finite_and_positive": all(
            record["all_scales_finite_and_positive"] for record in records
        ),
        "all_values_finite": all(record["all_values_finite"] for record in records),
        "all_group_counts_exact": all(record["group_count_exact"] for record in records),
        "topology": topology,
        "records": records,
    }


def paired(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    delta = np.asarray(candidate["per_batch_loss"], dtype=np.float64) - np.asarray(
        baseline["per_batch_loss"], dtype=np.float64
    )
    mean = float(delta.mean())
    standard_error = float(delta.std(ddof=1) / math.sqrt(delta.size))
    return {
        "candidate_minus_baseline_mean": mean,
        "standard_error": standard_error,
        "lower_95": mean - 1.96 * standard_error,
        "upper_95": mean + 1.96 * standard_error,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    integrity = validate_integrity()
    ledger = validate_data_ledger(
        args.data_manifest, args.train_file, args.validation_file,
        args.sequence_length,
    )
    validation = TokenFile(args.validation_file, args.sequence_length)
    protocol = {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "batches": args.batches,
        "validation_sequences": validation.sequence_count,
        "group_size": GROUP_SIZE,
        "all_ffn_matrices": True,
        "train_file": str(args.train_file),
        "validation_file": str(args.validation_file),
        "data_manifest": str(args.data_manifest),
        "output": str(args.output),
        "seed": args.seed,
    }
    protocol_valid = protocol == {
        "model": "HuggingFaceTB/SmolLM2-135M",
        "model_revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        "sequence_length": 512,
        "batch_size": 32,
        "batches": 128,
        "validation_sequences": 4096,
        "group_size": 128,
        "all_ffn_matrices": True,
        "train_file": "data/reflex-lm-screen/train.uint16.bin",
        "validation_file": "data/reflex-lm-screen/validation.uint16.bin",
        "data_manifest": "results/reflex-swiglu-lm-data-manifest.json",
        "output": "results/route-remapped-w4-endpoint.json",
        "seed": 131,
    }
    if not protocol_valid:
        raise ValueError("route-remapped W4 protocol drift")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=MODEL_REVISION, dtype=torch.float32,
        attn_implementation="sdpa",
    ).to("cuda")
    model.config.use_cache = False
    architecture = {
        "layers": len(model.model.layers),
        "hidden_size": model.config.hidden_size,
        "intermediate_size": model.config.intermediate_size,
    }
    if architecture != {
        "layers": 30, "hidden_size": 576, "intermediate_size": 1536,
    }:
        raise ValueError(f"unexpected model architecture: {architecture}")
    device = torch.device("cuda")
    baseline = evaluate(model, validation, args.batches, args.batch_size, device)
    quantization = quantize_model(model)
    decoded = evaluate(model, validation, args.batches, args.batch_size, device)
    relative_degradation = (decoded["loss"] - baseline["loss"]) / baseline["loss"]
    interval = paired(decoded, baseline)
    evidence_finite = all_numeric_values_finite({
        "baseline": baseline,
        "decoded": decoded,
        "paired_interval": interval,
        "quantization": quantization,
        "relative_nll_degradation": relative_degradation,
    })
    observed_ledger = {
        key: quantization[key] for key in EXPECTED_LEDGER
    }
    gates = {
        "integrity_protocol_data_valid": (
            integrity["valid"] and protocol_valid and ledger["valid"]
        ),
        "baseline_anchor_exact": abs(baseline["loss"] - EXPECTED_BASELINE) <= 1e-7,
        "relative_nll_degradation_at_most_0p2_percent": (
            relative_degradation <= 0.002
        ),
        "paired_upper_at_most_0p2_percent_baseline": (
            interval["upper_95"] <= 0.002 * baseline["loss"]
        ),
        "all_90_matrices_finite_and_bf16_scales_exact": (
            quantization["matrices"] == 90
            and quantization["all_scales_bf16_roundtrip"]
            and quantization["all_scales_finite_and_positive"]
            and quantization["all_codes_valid"]
            and quantization["all_values_finite"]
            and quantization["all_group_counts_exact"]
            and all(quantization["topology"].values())
            and evidence_finite
            and all(math.isfinite(value) for value in (
                baseline["loss"], decoded["loss"], relative_degradation,
                quantization["weighted_mse"],
                quantization["max_relative_l2_error"],
                quantization["max_absolute_error"],
            ))
        ),
        "resident_ledger_exact": (
            observed_ledger == EXPECTED_LEDGER
            and
            quantization["resident_bits"]
            == quantization["code_bits"] + quantization["scale_bits"]
            and quantization["bf16_baseline_bits"] == 16 * quantization["weights"]
        ),
    }
    payload = {
        "schema": "route-remapped-w4-endpoint-v1",
        "source_sha256": file_sha256(Path(__file__)),
        "preregistration_sha256": file_sha256(PREREGISTRATION),
        "formal_integrity": integrity,
        "protocol": protocol,
        "architecture": architecture,
        "data_ledger": ledger,
        "device": torch.cuda.get_device_name(0),
        "runtime": {
            "python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
        },
        "baseline": baseline,
        "decoded_w4_group128": decoded,
        "relative_nll_degradation": relative_degradation,
        "paired_interval": interval,
        "quantization": quantization,
        "expected_ledger": EXPECTED_LEDGER,
        "observed_ledger": observed_ledger,
        "all_numeric_evidence_finite": evidence_finite,
        "gates": gates,
        "endpoint_pass": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=TRAIN_FILE)
    parser.add_argument("--validation-file", type=Path, default=VALIDATION_FILE)
    parser.add_argument("--data-manifest", type=Path, default=DATA_MANIFEST)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batches", type=int, default=128)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    result = run(parser.parse_args())
    print(json.dumps({
        "relative_nll_degradation": result["relative_nll_degradation"],
        "resident_bits_per_weight": result["quantization"]["resident_bits_per_weight"],
        "gates": result["gates"],
        "endpoint_pass": result["endpoint_pass"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
