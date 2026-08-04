#!/usr/bin/env python3
"""Formal numerical and operator-cost gate for hybrid TVE autograd v3."""

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

import experiments.triangular_value_encoding_autograd_v2_gate as v2
from experiments.triangular_value_encoding_autograd_hybrid import (
    triangular_value_encoding_autograd_hybrid,
)


OUTPUT = Path("results/triangular-value-encoding-autograd-v3.json")
PREREGISTRATION = Path(
    "results/triangular-value-encoding-autograd-v3-preregistration.md"
)
MANIFEST = Path(
    "results/triangular-value-encoding-autograd-v3-integrity-manifest.json"
)
HYBRID_SOURCE = Path("experiments/triangular_value_encoding_autograd_hybrid.py")
HYBRID_TEST = Path("tests/test_triangular_value_encoding_autograd_hybrid.py")
V2_RESULT = Path("results/triangular-value-encoding-autograd-v2.json")
V2_MANIFEST = Path(
    "results/triangular-value-encoding-autograd-v2-integrity-manifest.json"
)
V2_PREREGISTRATION = Path(
    "results/triangular-value-encoding-autograd-v2-preregistration.md"
)
CASES = (
    {"name": "tail", "seed": 7411, "shape": (2, 2, 65, 64), "query_groups": 3},
    {"name": "pilot_37m", "seed": 8537, "shape": (32, 2, 512, 64), "query_groups": 3},
    {"name": "smollm2_360m", "seed": 9661, "shape": (8, 5, 512, 64), "query_groups": 3},
)

DEPENDENCIES = {
    "gate_source": Path(__file__),
    "hybrid_source": HYBRID_SOURCE,
    "hybrid_test": HYBRID_TEST,
    "preregistration": PREREGISTRATION,
    "reference_autograd_source": Path(
        "experiments/triangular_value_encoding_autograd.py"
    ),
    "reference_autograd_test": Path(
        "tests/test_triangular_value_encoding_autograd.py"
    ),
    "reference_attention_source": Path(
        "experiments/triangular_value_encoding_attention.py"
    ),
    "v2_gate_source": Path(
        "experiments/triangular_value_encoding_autograd_v2_gate.py"
    ),
    "v2_preregistration": V2_PREREGISTRATION,
    "v2_manifest": V2_MANIFEST,
    "v2_result": V2_RESULT,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_integrity() -> dict[str, bool]:
    manifest = json.loads(MANIFEST.read_text())
    checks = {
        name: manifest.get(name + "_sha256") == sha256_file(path)
        for name, path in DEPENDENCIES.items()
    }
    prior = json.loads(V2_RESULT.read_text())
    expected_prior_cases = {
        "tail": {"seed": 3181, "shape": [2, 2, 65, 64], "query_groups": 3},
        "pilot_37m": {
            "seed": 4261, "shape": [32, 2, 512, 64], "query_groups": 3,
        },
        "smollm2_360m": {
            "seed": 5347, "shape": [8, 5, 512, 64], "query_groups": 3,
        },
    }
    expected_prior_gate_names = {
        "active_weight_slot_count_exact", "all_finite", "forward_bit_exact",
        "outside_weight_support_exact_zero", "value_cosine_at_least_0_99998",
        "value_max_abs_at_most_0_0625", "value_relative_l2_at_most_0_005",
        "weight_cosine_at_least_0_99998", "weight_relative_l2_at_most_0_005",
        "weight_rmse_at_most_0_05",
    }
    prior_cases = {
        case.get("case", {}).get("name"): case
        for case in prior.get("correctness", [])
    }
    prior_failure_profile = (
        len(prior.get("correctness", [])) == len(expected_prior_cases)
        and set(prior_cases) == set(expected_prior_cases)
        and all(
        case.get("case") == {"name": name, **expected_prior_cases[name]}
        and set(case.get("gates", {})) == expected_prior_gate_names
        and case["gates"]["weight_rmse_at_most_0_05"] is False
        and all(
            case["gates"][gate] is True
            for gate in expected_prior_gate_names - {"weight_rmse_at_most_0_05"}
        )
        for name, case in prior_cases.items()
        )
    )
    checks.update({
        "v2_formal_no_go_disclosed": prior.get("all_gates_pass") is False,
        "v2_failure_is_weight_rmse_only": (
            prior.get("gates", {}).get("all_correctness_cases_pass") is False
            and prior_failure_profile
            and prior.get("gates", {}).get("signed_closed_form_exact") is True
            and prior.get("gates", {}).get("all_operator_timing_cases_pass") is True
        ),
        "v2_recorded_integrity_all_true": all(
            prior.get("integrity", {}).values()
        ) and bool(prior.get("integrity")),
        "v2_current_integrity_all_true": all(v2.validate_integrity().values()),
        "v2_manifest_matches_bound_result": (
            prior.get("integrity_manifest_sha256") == sha256_file(V2_MANIFEST)
        ),
    })
    if not all(checks.values()):
        raise ValueError(f"invalid hybrid v3 integrity: {checks}")
    return checks


def make_tensors(case: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, kv_heads, tokens, head_dim = case["shape"]
    hidden = kv_heads * case["query_groups"] * head_dim
    generator = torch.Generator(device="cuda").manual_seed(case["seed"])
    values = torch.randn(
        batch, kv_heads, tokens, head_dim,
        device="cuda", dtype=torch.bfloat16, generator=generator,
    )
    weight = torch.randn(
        hidden, hidden, device="cuda", dtype=torch.float32, generator=generator,
    ) * 0.01
    upstream = torch.randn(
        batch, kv_heads, tokens, head_dim,
        device="cuda", dtype=torch.bfloat16, generator=generator,
    )
    return values, weight, upstream


def hybrid_path(
    values: torch.Tensor, output_weight: torch.Tensor, query_groups: int
) -> torch.Tensor:
    return triangular_value_encoding_autograd_hybrid(
        values, output_weight, query_groups=query_groups
    )


def correctness_case(case: dict[str, Any]) -> dict[str, Any]:
    base_values, base_weight, upstream = make_tensors(case)
    actual_values = base_values.detach().clone().requires_grad_(True)
    actual_weight = base_weight.detach().clone().requires_grad_(True)
    expected_values = base_values.detach().clone().requires_grad_(True)
    expected_weight = base_weight.detach().clone().requires_grad_(True)
    actual = hybrid_path(actual_values, actual_weight, case["query_groups"])
    expected = v2.current_training_path(
        expected_values, expected_weight, case["query_groups"]
    )
    forward_exact = torch.equal(actual, expected)
    (actual.float() * upstream.float()).sum().backward()
    (expected.float() * upstream.float()).sum().backward()

    batch, kv_heads, _, head_dim = case["shape"]
    hidden = kv_heads * case["query_groups"] * head_dim
    mask = v2.active_weight_mask(
        hidden, kv_heads, case["query_groups"], head_dim, actual.device
    )
    value_metrics = v2.vector_metrics(actual_values.grad, expected_values.grad)
    dense_weight_exact = torch.equal(actual_weight.grad, expected_weight.grad)
    active_weight_exact = torch.equal(
        actual_weight.grad[mask], expected_weight.grad[mask]
    )
    outside_support_exact = (
        actual_weight.grad[~mask].count_nonzero().item() == 0
        and expected_weight.grad[~mask].count_nonzero().item() == 0
    )
    finite = all(
        torch.isfinite(tensor).all().item()
        for tensor in (
            actual, actual_values.grad, actual_weight.grad,
            expected_values.grad, expected_weight.grad,
        )
    ) and all(math.isfinite(value) for value in value_metrics.values())
    expected_slots = 960 if case["name"] in {"tail", "pilot_37m"} else 2400
    gates = {
        "active_weight_slot_count_exact": int(mask.sum()) == expected_slots,
        "forward_bit_exact": forward_exact,
        "dense_weight_gradient_bit_exact": dense_weight_exact,
        "active_weight_gradient_bit_exact": active_weight_exact,
        "outside_weight_support_exact_zero": outside_support_exact,
        "value_relative_l2_at_most_0_005": value_metrics["relative_l2"] <= 0.005,
        "value_cosine_at_least_0_99998": value_metrics["cosine"] >= 0.99998,
        "value_max_abs_at_most_0_0625": value_metrics["max_abs"] <= 0.0625,
        "all_finite": finite,
    }
    return {
        "case": case,
        "active_weight_slots": int(mask.sum()),
        "value_gradient": value_metrics,
        "dense_weight_gradient_bit_exact": dense_weight_exact,
        "active_weight_gradient_bit_exact": active_weight_exact,
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


def timing_case(case: dict[str, Any]) -> dict[str, Any]:
    base_values, base_weight, upstream = make_tensors(case)
    values = base_values.detach().clone().requires_grad_(True)
    weight = base_weight.detach().clone().requires_grad_(True)
    arms = {
        "current_duplicate_path": v2.current_training_path,
        "hybrid_single_forward": hybrid_path,
    }
    for _ in range(10):
        for function in arms.values():
            timed_call(function, values, weight, upstream, case["query_groups"])
    torch.cuda.synchronize()
    peaks = {}
    for name, function in arms.items():
        torch.cuda.reset_peak_memory_stats()
        timed_call(function, values, weight, upstream, case["query_groups"])
        peaks[name] = torch.cuda.max_memory_allocated()
    samples = {name: [] for name in arms}
    pairs = []
    generator = random.Random(case["seed"] + 10_000)
    for trial in range(40):
        order = list(arms)
        generator.shuffle(order)
        milliseconds = {}
        for name in order:
            elapsed = timed_call(
                arms[name], values, weight, upstream, case["query_groups"]
            )
            samples[name].append(elapsed)
            milliseconds[name] = elapsed
        pairs.append({
            "trial": trial,
            "order": order,
            "milliseconds": milliseconds,
            "hybrid_over_current_ratio": (
                milliseconds["hybrid_single_forward"]
                / milliseconds["current_duplicate_path"]
            ),
        })
    ratio = float(np.median([
        pair["hybrid_over_current_ratio"] for pair in pairs
    ]))
    memory_ok = peaks["hybrid_single_forward"] <= peaks["current_duplicate_path"]
    return {
        "case": case,
        "warmups_per_arm": 10,
        "trials_per_arm": 40,
        "milliseconds": {
            name: {
                "median": float(np.median(values_)),
                "p95": float(np.percentile(values_, 95)),
                "samples": values_,
            }
            for name, values_ in samples.items()
        },
        "paired_trials": pairs,
        "median_paired_hybrid_over_current_ratio": ratio,
        "peak_allocated_bytes": peaks,
        "hybrid_peak_not_above_current": memory_ok,
        "timing_pass": ratio <= 0.75 and memory_ok,
    }


def signed_closed_form() -> bool:
    spec = importlib.util.spec_from_file_location(
        "tve_hybrid_bound_test", HYBRID_TEST
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load bound hybrid test")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case = module.TriangularValueEncodingAutogradHybridTest(
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
    gates = {
        "hybrid_signed_closed_form_exact": signed_closed_form(),
        "all_correctness_cases_pass": all(
            case["correctness_pass"] for case in correctness
        ),
        "all_operator_timing_cases_pass": all(
            case["timing_pass"] for case in timing
        ),
    }
    payload = {
        "schema": "triangular-value-encoding-autograd-v3-hybrid-gate-v1",
        "device": torch.cuda.get_device_name(0),
        "source_sha256": sha256_file(Path(__file__)),
        "hybrid_source_sha256": sha256_file(HYBRID_SOURCE),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "integrity": integrity,
        "integrity_manifest_sha256": sha256_file(MANIFEST),
        "disclosure": "v2 failed frozen active-weight-gradient RMSE gates",
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
