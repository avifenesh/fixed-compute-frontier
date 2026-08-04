#!/usr/bin/env python3
"""Frozen H100 latency gate for Feedback-SwiGLU."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


def baseline_swiglu(
    x: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    return (F.silu(x @ gate.T) * (x @ up.T)) @ down.T


def feedback_swiglu(
    x: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    compress: torch.Tensor,
    expand: torch.Tensor,
) -> torch.Tensor:
    g = x @ gate.T
    u = x @ up.T
    z0 = F.silu(g) * u
    batch = z0.shape[0]
    groups, group_hidden, _ = compress.shape
    grouped_z0 = z0.reshape(batch, groups, group_hidden)
    state = torch.tanh(torch.einsum("bgh,ghq->bgq", grouped_z0, compress))
    delta = torch.einsum("bgq,ghq->bgh", state, expand).reshape(batch, -1)
    return (F.silu(g + delta) * u) @ down.T


def timed_call(fn: Callable[[], torch.Tensor]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = fn()
    end.record()
    end.synchronize()
    if output.numel() == 0:
        raise AssertionError("empty output")
    return float(start.elapsed_time(end) * 1000.0)


def timing_summary(values: list[float]) -> dict[str, float | int]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "samples": len(values),
        "min_us": float(tensor.min()),
        "p10_us": float(torch.quantile(tensor, 0.1)),
        "median_us": float(tensor.median()),
        "mean_us": float(tensor.mean()),
        "p90_us": float(torch.quantile(tensor, 0.9)),
        "max_us": float(tensor.max()),
    }


def make_weights(
    dimension: int,
    hidden: int,
    state: int,
    groups: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    gate = torch.randn(hidden, dimension, device=device, dtype=dtype) / dimension**0.5
    up = torch.randn(hidden, dimension, device=device, dtype=dtype) / dimension**0.5
    down = torch.randn(dimension, hidden, device=device, dtype=dtype) / hidden**0.5
    if hidden % groups:
        raise ValueError(f"hidden={hidden} must be divisible by groups={groups}")
    group_hidden = hidden // groups
    compress = (
        torch.randn(groups, group_hidden, state, device=device, dtype=dtype)
        / group_hidden**0.5
    )
    expand = (
        torch.randn(groups, group_hidden, state, device=device, dtype=dtype)
        * (0.1 / state**0.5)
    )
    return gate, up, down, compress, expand


def max_row_relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm((actual.float() - expected.float()), dim=-1)
    denominator = torch.linalg.vector_norm(expected.float(), dim=-1).clamp_min(1e-12)
    return float((numerator / denominator).max())


def driver_version() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).splitlines()[0]
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"unavailable: {exc}"


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    device_name = torch.cuda.get_device_name(0)
    batches = [int(value) for value in args.batches.split(",")]

    exact_equal_hidden = (3 * args.dimension * args.hidden) // (
        3 * args.dimension + 2 * args.state
    )
    total_alignment = args.hidden_alignment * args.groups
    equal_hidden = (exact_equal_hidden // total_alignment) * total_alignment

    base_weights = make_weights(
        args.dimension, args.hidden, args.state, args.groups, dtype, device
    )
    equal_weights = make_weights(
        args.dimension, equal_hidden, args.state, args.groups, dtype, device
    )

    compiled_baseline = torch.compile(baseline_swiglu, fullgraph=True, mode="reduce-overhead")
    compiled_feedback = torch.compile(feedback_swiglu, fullgraph=True, mode="reduce-overhead")

    cells: list[dict[str, object]] = []
    zero_feedback_errors: list[float] = []
    for batch in batches:
        print(f"benchmark B={batch}", flush=True)
        x = torch.randn(batch, args.dimension, device=device, dtype=dtype)
        gate, up, down, compress, expand = base_weights
        eq_gate, eq_up, eq_down, eq_compress, eq_expand = equal_weights
        zero_expand = torch.zeros_like(expand)

        base_fn = lambda: compiled_baseline(x, gate, up, down)
        same_fn = lambda: compiled_feedback(x, gate, up, down, compress, expand)
        equal_fn = lambda: compiled_feedback(
            x, eq_gate, eq_up, eq_down, eq_compress, eq_expand
        )

        with torch.inference_mode():
            baseline_output = base_fn().clone()
            zero_output = compiled_feedback(
                x, gate, up, down, compress, zero_expand
            ).clone()
            torch.cuda.synchronize()
            zero_feedback_errors.append(max_row_relative(zero_output, baseline_output))
            for _ in range(args.warmup):
                base_fn()
                same_fn()
                equal_fn()
            torch.cuda.synchronize()

            timings = {"baseline": [], "same_width": [], "equal_parameter": []}
            functions = {
                "baseline": base_fn,
                "same_width": same_fn,
                "equal_parameter": equal_fn,
            }
            for _ in range(args.samples):
                order = list(functions)
                random.shuffle(order)
                for name in order:
                    timings[name].append(timed_call(functions[name]))

        summaries = {name: timing_summary(values) for name, values in timings.items()}
        baseline_median = float(summaries["baseline"]["median_us"])
        cells.append(
            {
                "batch": batch,
                "timings": summaries,
                "same_width_over_baseline_median":
                    float(summaries["same_width"]["median_us"]) / baseline_median,
                "equal_parameter_over_baseline_median":
                    float(summaries["equal_parameter"]["median_us"]) / baseline_median,
                "zero_feedback_max_row_relative_error": zero_feedback_errors[-1],
            }
        )

    frozen_batches = {1, 8, 32, 128}
    protocol_valid = (
        "H100" in device_name
        and args.dimension == 4096
        and args.hidden == 14336
        and args.state == 8
        and args.groups == 8
        and set(batches) == frozen_batches
        and args.warmup >= 40
        and args.samples >= 200
        and equal_hidden == 14272
    )
    key_cells = [cell for cell in cells if cell["batch"] in {1, 8}]
    gates = {
        "protocol_valid": protocol_valid,
        "zero_feedback_inclusion_below_0p2_percent": max(zero_feedback_errors) <= 0.002,
        "same_width_within_1p02_on_key_cells": all(
            cell["same_width_over_baseline_median"] <= 1.02 for cell in key_cells
        ),
        "same_width_within_1p05_on_full_grid": all(
            cell["same_width_over_baseline_median"] <= 1.05 for cell in cells
        ),
        "equal_parameter_within_1p02_on_key_cells": all(
            cell["equal_parameter_over_baseline_median"] <= 1.02 for cell in key_cells
        ),
        "equal_parameter_within_1p05_on_full_grid": all(
            cell["equal_parameter_over_baseline_median"] <= 1.05 for cell in cells
        ),
    }
    source = Path(__file__)
    prereg = Path("results/feedback-swiglu-h100-preregistration.md")
    return {
        "candidate": "Feedback-SwiGLU",
        "scope": "stock torch.compile H100 execution gate only",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "preregistration_sha256": hashlib.sha256(prereg.read_bytes()).hexdigest(),
        "args": {**vars(args), "output": str(args.output)},
        "device": device_name,
        "device_properties": str(torch.cuda.get_device_properties(0)),
        "driver_version": driver_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "ledger": {
            "baseline_hidden": args.hidden,
            "exact_equal_parameter_hidden": exact_equal_hidden,
            "aligned_equal_parameter_hidden": equal_hidden,
            "feedback_groups": args.groups,
            "features_per_baseline_group": args.hidden // args.groups,
            "adds_tensor_parallel_collective": False,
            "same_width_overhead_fraction": 2 * args.state / (3 * args.dimension),
            "baseline_parameters": 3 * args.dimension * args.hidden,
            "equal_parameter_candidate_parameters":
                equal_hidden * (3 * args.dimension + 2 * args.state),
        },
        "cells": cells,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=14336)
    parser.add_argument("--state", type=int, default=8)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--hidden-alignment", type=int, default=8)
    parser.add_argument("--batches", default="1,8,32,128")
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/feedback-swiglu-h100.json"),
    )
    args = parser.parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": payload["gates"], "all_gates_pass": payload["all_gates_pass"]}, indent=2))
    raise SystemExit(0 if payload["all_gates_pass"] else 1)


if __name__ == "__main__":
    main()
