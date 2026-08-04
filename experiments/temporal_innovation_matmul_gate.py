#!/usr/bin/env python3
"""CPU-only Stage-0 gate for temporal-innovation dense projections."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT / "results" / "temporal-innovation-matmul-stage0-preregistration.md"
)
MODEL_REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
MODEL_FILE_SHA256 = "7aaff6661428bed033abba9522bec81938678642cca3181fe752b6ca9e1e540f"
SEED = 260726
SEQUENCE_LENGTH = 192
TRANSITIONS_PER_LANE = 16
BITS = (8, 6, 4)
TOPK_FRACTIONS = (0.10, 0.25, 0.50)
PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
ATTENTION_PROJECTIONS = frozenset(("q_proj", "k_proj", "v_proj", "o_proj"))
FFN_PROJECTIONS = frozenset(("gate_proj", "up_proj", "down_proj"))
LANES = ("prose", "dialogue", "code", "random_tokens")
LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_algebra_check(
    *, width: int = 96, output_width: int = 80, sparse_fraction: float = 0.05
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    weight = torch.randn(output_width, width, generator=generator, dtype=torch.float32)
    previous = torch.randn(width, generator=generator, dtype=torch.float32)
    delta = torch.zeros(width, dtype=torch.float32)
    changed = torch.randperm(width, generator=generator)[: math.ceil(width * sparse_fraction)]
    delta[changed] = torch.randn(changed.numel(), generator=generator)
    current = previous + delta

    previous_output = F.linear(previous, weight)
    full_output = F.linear(current, weight)
    incremental_output = previous_output + F.linear(delta, weight)
    relative_error = float(
        torch.linalg.vector_norm(full_output - incremental_output)
        / torch.linalg.vector_norm(full_output)
    )

    gamma = torch.randn(width, generator=generator, dtype=torch.float32)
    epsilon = 1e-5
    previous_scale = torch.rsqrt(previous.square().mean() + epsilon)
    current_scale = torch.rsqrt(current.square().mean() + epsilon)
    previous_normalized_output = F.linear(previous * gamma * previous_scale, weight)
    full_normalized_output = F.linear(current * gamma * current_scale, weight)
    incremental_normalized_output = (
        (current_scale / previous_scale) * previous_normalized_output
        + current_scale * F.linear(delta * gamma, weight)
    )
    normalized_relative_error = float(
        torch.linalg.vector_norm(
            full_normalized_output - incremental_normalized_output
        )
        / torch.linalg.vector_norm(full_normalized_output)
    )

    fanout = F.linear(delta, weight)
    return {
        "input_width": width,
        "output_width": output_width,
        "changed_coordinates": int(changed.numel()),
        "changed_fraction": float(changed.numel() / width),
        "dense_relative_error": relative_error,
        "rmsnorm_relative_error": normalized_relative_error,
        "fanout_nonzero_fraction": float((fanout != 0).float().mean()),
        "fanout_minimum_absolute_change": float(fanout.abs().min()),
        "passes": relative_error < 1e-6 and normalized_relative_error < 1e-6,
    }


def selected_layer_indices(layer_indices: Iterable[int]) -> tuple[int, ...]:
    ordered = sorted(set(layer_indices))
    if not ordered:
        raise ValueError("no decoder layers found")
    positions = (0.0, 0.25, 0.50, 0.75, 1.0)
    return tuple(
        sorted({ordered[round((len(ordered) - 1) * position)] for position in positions})
    )


def transition_indices(sequence_length: int) -> torch.Tensor:
    if sequence_length < 2:
        raise ValueError("sequence must have at least two positions")
    count = min(TRANSITIONS_PER_LANE, sequence_length - 1)
    indices = np.linspace(1, sequence_length - 1, count, dtype=np.int64)
    return torch.from_numpy(np.unique(indices))


def quantize_per_coordinate(
    inputs: torch.Tensor, bits: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if bits < 2:
        raise ValueError("signed quantization needs at least two bits")
    qmax = (1 << (bits - 1)) - 1
    maximum = inputs.abs().amax(dim=(0, 1))
    scale = maximum / qmax
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = torch.round(inputs / scale).clamp(-qmax, qmax).to(torch.int16)
    reconstructed = codes.to(inputs.dtype) * scale
    return codes, reconstructed, scale


def _relative_l2(reference: torch.Tensor, tested: torch.Tensor) -> torch.Tensor:
    numerator = torch.linalg.vector_norm(reference - tested, dim=1)
    denominator = torch.linalg.vector_norm(reference, dim=1).clamp_min(1e-12)
    return numerator / denominator


def projection_records(
    *,
    inputs: torch.Tensor,
    module: nn.Linear,
    layer: int,
    projection: str,
) -> list[dict[str, Any]]:
    if inputs.ndim != 3:
        raise ValueError(f"expected [batch, tokens, channels], got {tuple(inputs.shape)}")
    if inputs.shape[0] != len(LANES):
        raise ValueError(f"expected {len(LANES)} lanes, got {inputs.shape[0]}")

    values = inputs.detach().to(device="cpu", dtype=torch.float32)
    weight = module.weight.detach().to(device="cpu", dtype=torch.float32)
    bias = None
    if module.bias is not None:
        bias = module.bias.detach().to(device="cpu", dtype=torch.float32)
    indices = transition_indices(values.shape[1])
    previous = values[:, indices - 1, :]
    current = values[:, indices, :]
    deltas = current - previous
    flat_delta = deltas.reshape(-1, deltas.shape[-1])
    flat_current = current.reshape(-1, current.shape[-1])
    exact_delta_output = F.linear(flat_delta, weight)
    exact_current_output = F.linear(flat_current, weight, bias)

    topk_errors: dict[float, torch.Tensor] = {}
    for fraction in TOPK_FRACTIONS:
        count = max(1, math.ceil(flat_delta.shape[1] * fraction))
        positions = torch.topk(flat_delta.abs(), count, dim=1).indices
        sparse_delta = torch.zeros_like(flat_delta)
        sparse_delta.scatter_(1, positions, flat_delta.gather(1, positions))
        tested_output = F.linear(sparse_delta, weight)
        topk_errors[fraction] = _relative_l2(exact_delta_output, tested_output)

    quantized_metrics: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for bits in BITS:
        codes, reconstructed, _ = quantize_per_coordinate(values, bits)
        code_previous = codes[:, indices - 1, :]
        code_current = codes[:, indices, :]
        changed = (code_current != code_previous).float().mean(dim=2).reshape(-1)
        reconstructed_current = reconstructed[:, indices, :].reshape(
            -1, reconstructed.shape[-1]
        )
        tested_output = F.linear(reconstructed_current, weight, bias)
        distortion = _relative_l2(exact_current_output, tested_output)
        quantized_metrics[bits] = (changed, distortion)

    exact_changed = (flat_delta != 0).float().mean(dim=1)
    records: list[dict[str, Any]] = []
    row = 0
    for lane_index, lane in enumerate(LANES):
        for token_index in indices.tolist():
            record = {
                "layer": layer,
                "projection": projection,
                "group": (
                    "attention" if projection in ATTENTION_PROJECTIONS else "ffn"
                ),
                "lane": lane,
                "token_index": int(token_index),
                "input_width": int(weight.shape[1]),
                "output_width": int(weight.shape[0]),
                "exact_changed_fraction": float(exact_changed[row]),
                "topk_relative_delta_error": {
                    f"{fraction:.2f}": float(topk_errors[fraction][row])
                    for fraction in TOPK_FRACTIONS
                },
                "quantized": {
                    str(bits): {
                        "changed_code_fraction": float(quantized_metrics[bits][0][row]),
                        "relative_output_distortion": float(
                            quantized_metrics[bits][1][row]
                        ),
                    }
                    for bits in BITS
                },
            }
            records.append(record)
            row += 1
    return records


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty metric")
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "mean": float(array.mean()),
        "maximum": float(array.max()),
    }


def _aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "exact_changed_fraction": _stats(
            record["exact_changed_fraction"] for record in records
        ),
        "topk_relative_delta_error": {
            f"{fraction:.2f}": _stats(
                record["topk_relative_delta_error"][f"{fraction:.2f}"]
                for record in records
            )
            for fraction in TOPK_FRACTIONS
        },
        "quantized": {
            str(bits): {
                "changed_code_fraction": _stats(
                    record["quantized"][str(bits)]["changed_code_fraction"]
                    for record in records
                ),
                "relative_output_distortion": _stats(
                    record["quantized"][str(bits)]["relative_output_distortion"]
                    for record in records
                ),
            }
            for bits in BITS
        },
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("no projection records were captured")
    summaries: dict[str, Any] = {
        "all": _aggregate_records(records),
        "groups": {},
        "projections": {},
        "layers": {},
        "lanes": {},
    }
    for group in ("attention", "ffn"):
        summaries["groups"][group] = _aggregate_records(
            [record for record in records if record["group"] == group]
        )
    for projection in PROJECTIONS:
        subset = [record for record in records if record["projection"] == projection]
        if subset:
            summaries["projections"][projection] = _aggregate_records(subset)
    for layer in sorted({record["layer"] for record in records}):
        summaries["layers"][str(layer)] = _aggregate_records(
            [record for record in records if record["layer"] == layer]
        )
    for lane in LANES:
        summaries["lanes"][lane] = _aggregate_records(
            [record for record in records if record["lane"] == lane]
        )
    return summaries


def gate_decision(summary: dict[str, Any]) -> dict[str, Any]:
    groups = summary["groups"]

    exact_groups = {}
    for group in ("attention", "ffn"):
        stats = groups[group]["exact_changed_fraction"]
        exact_groups[group] = stats["median"] <= 0.10 and stats["p90"] <= 0.20
    free_exact_reuse = all(exact_groups.values())

    raw_top10_groups = {}
    for group in ("attention", "ffn"):
        stats = groups[group]["topk_relative_delta_error"]["0.10"]
        raw_top10_groups[group] = stats["median"] <= 0.05 and stats["p90"] <= 0.10
    raw_top10_passes = all(raw_top10_groups.values())

    quantized: dict[str, Any] = {}
    for bits in BITS:
        group_passes = {}
        for group in ("attention", "ffn"):
            metrics = groups[group]["quantized"][str(bits)]
            changed = metrics["changed_code_fraction"]
            distortion = metrics["relative_output_distortion"]
            group_passes[group] = (
                changed["median"] <= 0.10
                and changed["p90"] <= 0.20
                and distortion["median"] <= 0.01
                and distortion["p90"] <= 0.02
            )
        quantized[str(bits)] = {
            "groups": group_passes,
            "event_quantized_slack": all(group_passes.values())
            and raw_top10_passes,
        }

    any_quantized = any(
        result["event_quantized_slack"] for result in quantized.values()
    )
    return {
        "status": (
            "joint_training_hypothesis_only"
            if any_quantized
            else "reject_exact_and_pretrained_slack"
        ),
        "algebraic_exact_composition_rejected": True,
        "free_exact_reuse": free_exact_reuse,
        "exact_groups": exact_groups,
        "raw_top10_passes": raw_top10_passes,
        "raw_top10_groups": raw_top10_groups,
        "quantized": quantized,
        "gpu_rental_justified": False,
    }


def resource_ledger(
    modules: dict[str, nn.Linear], *, decoder_layers: int
) -> dict[str, Any]:
    required = set(PROJECTIONS)
    missing = required.difference(modules)
    if missing:
        raise ValueError(f"missing projection shapes: {sorted(missing)}")

    output_elements = sum(modules[name].out_features for name in PROJECTIONS)
    unique_input_elements = (
        modules["q_proj"].in_features
        + modules["o_proj"].in_features
        + modules["gate_proj"].in_features
        + modules["down_proj"].in_features
    )
    naive_elements = unique_input_elements + output_elements
    kv_already_cached = modules["k_proj"].out_features + modules["v_proj"].out_features
    lower_bound_elements = naive_elements - kv_already_cached
    return {
        "decoder_layers": decoder_layers,
        "projection_shapes": {
            name: {
                "input": modules[name].in_features,
                "output": modules[name].out_features,
            }
            for name in PROJECTIONS
        },
        "per_layer": {
            "unique_previous_input_elements": unique_input_elements,
            "previous_output_elements": output_elements,
            "naive_incremental_state_elements": naive_elements,
            "lower_bound_elements_excluding_existing_kv": lower_bound_elements,
            "naive_bf16_bytes": naive_elements * 2,
            "lower_bound_bf16_bytes_excluding_existing_kv": lower_bound_elements * 2,
        },
        "all_layers": {
            "naive_bf16_bytes": naive_elements * 2 * decoder_layers,
            "lower_bound_bf16_bytes_excluding_existing_kv": (
                lower_bound_elements * 2 * decoder_layers
            ),
        },
        "omitted_from_logical_weight_read_fraction": [
            "dense output-vector reads and writes",
            "changed-index encoding",
            "column-gather inefficiency",
            "support divergence across requests",
            "kernel launch and synchronization",
        ],
        "hardware_boundary": (
            "Arbitrary changed columns are gathers, not a dense Tensor-Core GEMM; "
            "only direct H100/H200 kernels can establish latency or cost."
        ),
    }


def _fixed_panel(tokenizer: Any) -> torch.Tensor:
    texts = {
        "prose": (
            "A system is understood by tracing what changes, what stays invariant, "
            "and which resource pays for each transformation. Dense computation can "
            "hide repeated work, but an exact shortcut must preserve every dependency. "
        ),
        "dialogue": (
            "User: What changed since the previous step? Assistant: The state changed, "
            "but the contract did not. User: Can we reuse the result? Assistant: Only "
            "if the update remains sparse after every transformation. "
        ),
        "code": (
            "def update(weight, previous, delta):\n"
            "    projected = previous + weight @ delta\n"
            "    assert projected.shape == previous.shape\n"
            "    return projected\n"
        ),
    }
    lanes: list[list[int]] = []
    bos = tokenizer.bos_token_id
    for lane in LANES[:-1]:
        base = tokenizer.encode(texts[lane], add_special_tokens=False)
        if not base:
            raise RuntimeError(f"tokenizer produced no tokens for {lane}")
        prefix = [] if bos is None else [bos]
        needed = SEQUENCE_LENGTH - len(prefix)
        repeated = (base * math.ceil(needed / len(base)))[:needed]
        lanes.append(prefix + repeated)

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    special = set(tokenizer.all_special_ids)
    candidates = [index for index in range(tokenizer.vocab_size) if index not in special]
    positions = torch.randint(
        0, len(candidates), (SEQUENCE_LENGTH,), generator=generator
    ).tolist()
    lanes.append([candidates[position] for position in positions])
    return torch.tensor(lanes, dtype=torch.long)


def _projection_modules(model: nn.Module) -> dict[int, dict[str, nn.Linear]]:
    found: dict[int, dict[str, nn.Linear]] = {}
    for name, module in model.named_modules():
        projection = name.rsplit(".", 1)[-1]
        if projection not in PROJECTIONS or not isinstance(module, nn.Linear):
            continue
        match = LAYER_PATTERN.search(name)
        if match is None:
            continue
        layer = int(match.group(1))
        if projection in found.setdefault(layer, {}):
            raise RuntimeError(f"duplicate {projection} in layer {layer}")
        found[layer][projection] = module
    complete = {
        layer: modules
        for layer, modules in found.items()
        if set(modules) == set(PROJECTIONS)
    }
    if not complete:
        raise RuntimeError("no complete decoder projection groups found")
    return complete


def run_model_gate(model_dir: Path, *, threads: int) -> dict[str, Any]:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch.set_num_threads(threads)
    torch.manual_seed(SEED)

    model_file = model_dir / "model.safetensors"
    actual_model_sha = sha256(model_file)
    if actual_model_sha != MODEL_FILE_SHA256:
        raise RuntimeError(
            f"checkpoint hash mismatch: expected {MODEL_FILE_SHA256}, got {actual_model_sha}"
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    projection_modules = _projection_modules(model)
    layers = selected_layer_indices(projection_modules)
    first_layer_modules = projection_modules[min(projection_modules)]

    records: list[dict[str, Any]] = []
    handles = []
    for layer in layers:
        for projection, module in projection_modules[layer].items():
            def capture(
                captured_module: nn.Module,
                arguments: tuple[torch.Tensor, ...],
                *,
                captured_layer: int = layer,
                captured_projection: str = projection,
            ) -> None:
                if not isinstance(captured_module, nn.Linear):
                    raise TypeError("capture hook was attached to a non-linear module")
                records.extend(
                    projection_records(
                        inputs=arguments[0],
                        module=captured_module,
                        layer=captured_layer,
                        projection=captured_projection,
                    )
                )

            handles.append(module.register_forward_pre_hook(capture))

    input_ids = _fixed_panel(tokenizer)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    elapsed = time.perf_counter() - started

    summary = summarize_records(records)
    decision = gate_decision(summary)
    decoder_layer_count = len(projection_modules)
    return {
        "schema_version": 1,
        "claim": "temporal-innovation update of arbitrary dense projections",
        "model": {
            "reference": f"HuggingFaceTB/SmolLM2-360M@{MODEL_REVISION}",
            "local_path": str(model_dir),
            "model_safetensors_sha256": actual_model_sha,
            "decoder_layers": decoder_layer_count,
            "sampled_layers": list(layers),
        },
        "execution": {
            "device": "cpu",
            "dtype": "float32",
            "threads": threads,
            "seed": SEED,
            "sequence_length": SEQUENCE_LENGTH,
            "lanes": list(LANES),
            "transitions_per_lane": TRANSITIONS_PER_LANE,
            "forward_seconds": elapsed,
            "torch_version": torch.__version__,
        },
        "identity": {
            "source_sha256": sha256(Path(__file__)),
            "preregistration_sha256": sha256(PREREGISTRATION),
        },
        "exact_algebra": exact_algebra_check(),
        "resource_ledger": resource_ledger(
            first_layer_modules, decoder_layers=decoder_layer_count
        ),
        "summary": summary,
        "decision": decision,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()

    if arguments.self_test:
        payload = {
            "source_sha256": sha256(Path(__file__)),
            "preregistration_sha256": sha256(PREREGISTRATION),
            "exact_algebra": exact_algebra_check(),
        }
    else:
        if arguments.model_dir is None or arguments.output is None:
            parser.error("--model-dir and --output are required unless --self-test is used")
        payload = run_model_gate(arguments.model_dir, threads=arguments.threads)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(payload, indent=2) + "\n")

    concise = {
        "exact_algebra": payload["exact_algebra"],
        "decision": payload.get("decision"),
        "output": None if arguments.output is None else str(arguments.output),
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
