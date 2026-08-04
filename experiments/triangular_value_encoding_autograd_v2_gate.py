#!/usr/bin/env python3
"""Numerical and operator-cost gate for the TVE custom backward."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
from typing import Any, Callable
import unittest

import numpy as np
import torch

from experiments.triangular_value_encoding_attention import serving_value_encoding
from experiments.triangular_value_encoding_autograd import triangular_value_encoding_autograd


OUTPUT = Path("results/triangular-value-encoding-autograd-v2.json")
PREREGISTRATION = Path("results/triangular-value-encoding-autograd-v2-preregistration.md")
MANIFEST = Path("results/triangular-value-encoding-autograd-v2-integrity-manifest.json")
AUTOGRAD_SOURCE = Path("experiments/triangular_value_encoding_autograd.py")
ATTENTION_SOURCE = Path("experiments/triangular_value_encoding_attention.py")
AUTOGRAD_TEST = Path("tests/test_triangular_value_encoding_autograd.py")
CASES = (
    {"name": "tail", "seed": 3181, "shape": (2, 2, 65, 64), "query_groups": 3},
    {"name": "pilot_37m", "seed": 4261, "shape": (32, 2, 512, 64), "query_groups": 3},
    {"name": "smollm2_360m", "seed": 5347, "shape": (8, 5, 512, 64), "query_groups": 3},
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_integrity() -> dict[str, bool]:
    manifest = json.loads(MANIFEST.read_text())
    paths = {
        "gate_source": Path(__file__),
        "autograd_source": AUTOGRAD_SOURCE,
        "attention_source": ATTENTION_SOURCE,
        "preregistration": PREREGISTRATION,
        "autograd_test": AUTOGRAD_TEST,
    }
    checks = {
        name: manifest.get(name + "_sha256") == sha256_file(path)
        for name, path in paths.items()
    }
    if not all(checks.values()):
        raise ValueError(f"invalid autograd v2 integrity: {checks}")
    return checks


def torch_surrogate(
    values: torch.Tensor,
    output_weight: torch.Tensor,
    query_groups: int,
) -> torch.Tensor:
    head_dim = values.shape[-1]
    kv_heads = values.shape[1]
    pieces = []
    for start in range(0, head_dim, 16):
        block = values[..., start:start + 16]
        physical = torch.stack([
            output_weight[
                start:start + 16,
                kv_head * query_groups * head_dim + start:
                kv_head * query_groups * head_dim + start + 16,
            ]
            for kv_head in range(kv_heads)
        ])
        coefficient = (
            torch.tril(physical.to(block.dtype), diagonal=-1).float() / 0.125
        ).to(block.dtype)
        feature = (block.float() * block.float().abs()).to(block.dtype)
        delta = torch.matmul(
            coefficient[None, :, None], feature.unsqueeze(-1)
        ).squeeze(-1)
        pieces.append((block.float() + delta.float()).to(block.dtype))
    return torch.cat(pieces, dim=-1)


def current_training_path(
    values: torch.Tensor,
    output_weight: torch.Tensor,
    query_groups: int,
) -> torch.Tensor:
    differentiable = torch_surrogate(values, output_weight, query_groups)
    batch, kv_heads, tokens, head_dim = values.shape
    token_major = values.transpose(1, 2).contiguous().view(
        batch * tokens, kv_heads, head_dim
    )
    serving = serving_value_encoding(
        token_major,
        output_weight.to(torch.bfloat16),
        query_groups=query_groups,
        block_size=16,
        tau=0.125,
    ).view(batch, tokens, kv_heads, head_dim).transpose(1, 2)
    return serving.detach() + (differentiable - differentiable.detach())


def active_weight_mask(
    hidden: int,
    kv_heads: int,
    query_groups: int,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(hidden, hidden, device=device, dtype=torch.bool)
    for kv_head in range(kv_heads):
        representative = kv_head * query_groups * head_dim
        for start in range(0, head_dim, 16):
            for target in range(1, 16):
                mask[
                    start + target,
                    representative + start:representative + start + target,
                ] = True
    return mask


def vector_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    left = actual.double().flatten()
    right = expected.double().flatten()
    difference = left - right
    expected_norm = torch.linalg.vector_norm(right)
    actual_norm = torch.linalg.vector_norm(left)
    denominator = max(float(expected_norm), torch.finfo(torch.float64).tiny)
    cosine_denominator = max(
        float(expected_norm * actual_norm), torch.finfo(torch.float64).tiny
    )
    return {
        "relative_l2": float(torch.linalg.vector_norm(difference)) / denominator,
        "cosine": float(torch.dot(left, right)) / cosine_denominator,
        "rmse": float(torch.sqrt(torch.mean(difference.square()))),
        "max_abs": float(difference.abs().max()),
    }


def make_tensors(case: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, kv_heads, tokens, head_dim = case["shape"]
    hidden = kv_heads * case["query_groups"] * head_dim
    generator = torch.Generator(device="cuda").manual_seed(case["seed"])
    values = torch.randn(
        batch, kv_heads, tokens, head_dim,
        device="cuda", dtype=torch.bfloat16, generator=generator,
    )
    output_weight = torch.randn(
        hidden, hidden, device="cuda", dtype=torch.float32, generator=generator,
    ) * 0.01
    upstream = torch.randn(
        batch, kv_heads, tokens, head_dim,
        device="cuda", dtype=torch.bfloat16, generator=generator,
    )
    return values, output_weight, upstream


def correctness_case(case: dict[str, Any]) -> dict[str, Any]:
    base_values, base_weight, upstream = make_tensors(case)
    actual_values = base_values.detach().clone().requires_grad_(True)
    actual_weight = base_weight.detach().clone().requires_grad_(True)
    expected_values = base_values.detach().clone().requires_grad_(True)
    expected_weight = base_weight.detach().clone().requires_grad_(True)
    actual = triangular_value_encoding_autograd(
        actual_values, actual_weight, query_groups=case["query_groups"]
    )
    expected = current_training_path(
        expected_values, expected_weight, case["query_groups"]
    )
    forward_exact = torch.equal(actual, expected)
    (actual.float() * upstream.float()).sum().backward()
    (expected.float() * upstream.float()).sum().backward()

    batch, kv_heads, tokens, head_dim = case["shape"]
    hidden = kv_heads * case["query_groups"] * head_dim
    mask = active_weight_mask(
        hidden, kv_heads, case["query_groups"], head_dim, actual.device
    )
    value_metrics = vector_metrics(actual_values.grad, expected_values.grad)
    weight_metrics = vector_metrics(actual_weight.grad[mask], expected_weight.grad[mask])
    outside_support_exact = (
        actual_weight.grad[~mask].count_nonzero().item() == 0
        and expected_weight.grad[~mask].count_nonzero().item() == 0
    )
    finite = all(
        torch.isfinite(tensor).all().item()
        for tensor in (
            actual,
            actual_values.grad,
            actual_weight.grad,
            expected_values.grad,
            expected_weight.grad,
        )
    ) and all(
        math.isfinite(value)
        for metrics in (value_metrics, weight_metrics)
        for value in metrics.values()
    )
    gates = {
        "active_weight_slot_count_exact": int(mask.sum())
        == (960 if case["name"] in {"tail", "pilot_37m"} else 2400),
        "forward_bit_exact": forward_exact,
        "outside_weight_support_exact_zero": outside_support_exact,
        "value_relative_l2_at_most_0_005": value_metrics["relative_l2"] <= 0.005,
        "value_cosine_at_least_0_99998": value_metrics["cosine"] >= 0.99998,
        "value_max_abs_at_most_0_0625": value_metrics["max_abs"] <= 0.0625,
        "weight_relative_l2_at_most_0_005": weight_metrics["relative_l2"] <= 0.005,
        "weight_cosine_at_least_0_99998": weight_metrics["cosine"] >= 0.99998,
        "weight_rmse_at_most_0_05": weight_metrics["rmse"] <= 0.05,
        "all_finite": finite,
    }
    return {
        "case": case,
        "active_weight_slots": int(mask.sum()),
        "value_gradient": value_metrics,
        "active_weight_gradient": weight_metrics,
        "gates": gates,
        "correctness_pass": all(gates.values()),
    }


def timed_call(
    function: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
    values: torch.Tensor,
    weight: torch.Tensor,
    upstream: torch.Tensor,
    query_groups: int,
) -> float:
    values.grad = None
    weight.grad = None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = function(values, weight, query_groups)
    (output.float() * upstream.float()).sum().backward()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def custom_path(
    values: torch.Tensor,
    output_weight: torch.Tensor,
    query_groups: int,
) -> torch.Tensor:
    return triangular_value_encoding_autograd(
        values, output_weight, query_groups=query_groups
    )


def timing_case(case: dict[str, Any]) -> dict[str, Any]:
    base_values, base_weight, upstream = make_tensors(case)
    values = base_values.detach().clone().requires_grad_(True)
    weight = base_weight.detach().clone().requires_grad_(True)
    arms = {
        "current_duplicate_path": current_training_path,
        "custom_single_forward": custom_path,
    }
    for _ in range(10):
        for function in arms.values():
            timed_call(function, values, weight, upstream, case["query_groups"])
    torch.cuda.synchronize()
    peak_by_arm = {}
    for name, function in arms.items():
        torch.cuda.reset_peak_memory_stats()
        timed_call(function, values, weight, upstream, case["query_groups"])
        peak_by_arm[name] = torch.cuda.max_memory_allocated()
    samples = {name: [] for name in arms}
    pairs = []
    generator = random.Random(case["seed"] + 10_000)
    for trial in range(40):
        order = list(arms)
        generator.shuffle(order)
        pair = {"trial": trial, "order": order, "milliseconds": {}}
        for name in order:
            elapsed = timed_call(
                arms[name], values, weight, upstream, case["query_groups"]
            )
            samples[name].append(elapsed)
            pair["milliseconds"][name] = elapsed
        pair["custom_over_current_ratio"] = (
            pair["milliseconds"]["custom_single_forward"]
            / pair["milliseconds"]["current_duplicate_path"]
        )
        pairs.append(pair)
    medians = {name: float(np.median(values)) for name, values in samples.items()}
    ratio = float(np.median([
        pair["custom_over_current_ratio"] for pair in pairs
    ]))
    memory_ok = (
        peak_by_arm["custom_single_forward"]
        <= peak_by_arm["current_duplicate_path"]
    )
    return {
        "case": case,
        "warmups_per_arm": 10,
        "trials_per_arm": 40,
        "milliseconds": {
            name: {
                "median": medians[name],
                "p95": float(np.percentile(values, 95)),
                "samples": values,
            }
            for name, values in samples.items()
        },
        "paired_trials": pairs,
        "median_paired_custom_over_current_ratio": ratio,
        "peak_allocated_bytes": peak_by_arm,
        "custom_peak_not_above_current": memory_ok,
        "timing_pass": ratio <= 0.75 and memory_ok,
    }


def signed_closed_form() -> bool:
    spec = importlib.util.spec_from_file_location("tve_autograd_bound_test", AUTOGRAD_TEST)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load bound autograd test")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case = module.TriangularValueEncodingAutogradTest(
        "test_one_physical_slot_has_exact_local_forward_and_backward_support"
    )
    result = unittest.TestResult()
    case.run(result)
    return result.testsRun == 1 and not result.skipped and result.wasSuccessful()


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    integrity = validate_integrity()
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    correctness = [correctness_case(case) for case in CASES]
    timing = [timing_case(case) for case in CASES[1:]]
    exact = signed_closed_form()
    gates = {
        "signed_closed_form_exact": exact,
        "all_correctness_cases_pass": all(case["correctness_pass"] for case in correctness),
        "all_operator_timing_cases_pass": all(case["timing_pass"] for case in timing),
    }
    payload = {
        "schema": "triangular-value-encoding-autograd-v2-gate-v1",
        "device": torch.cuda.get_device_name(0),
        "source_sha256": sha256_file(Path(__file__)),
        "autograd_source_sha256": sha256_file(
            Path("experiments/triangular_value_encoding_autograd.py")
        ),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "integrity": integrity,
        "integrity_manifest_sha256": sha256_file(MANIFEST),
        "disclosure": "v1 elementwise gradient gate failed",
        "correctness": correctness,
        "timing": timing,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    payload = run(parser.parse_args().output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
