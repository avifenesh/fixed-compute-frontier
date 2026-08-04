#!/usr/bin/env python3
"""Fused single-H100 serving gate for group-local Feedback-SwiGLU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
from pathlib import Path
from typing import Callable

import torch
import triton
import triton.language as tl


@triton.jit
def baseline_activation_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    silu_gate = gate * tl.sigmoid(gate)
    tl.store(out_ptr + offsets, silu_gate * up, mask=mask)


@triton.jit
def feedback_activation_kernel(
    gate_ptr,
    up_ptr,
    compress_ptr,
    expand_ptr,
    out_ptr,
    GROUP_HIDDEN: tl.constexpr,
    STATE: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    batch_index = tl.program_id(0)
    group_index = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_HIDDEN)
    mask = offsets < GROUP_HIDDEN
    activation_base = (batch_index * GROUPS + group_index) * GROUP_HIDDEN
    activation_offsets = activation_base + offsets
    gate = tl.load(gate_ptr + activation_offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + activation_offsets, mask=mask, other=0.0).to(tl.float32)
    z0 = gate * tl.sigmoid(gate) * up
    delta = tl.zeros((BLOCK_HIDDEN,), dtype=tl.float32)
    factor_base = group_index * GROUP_HIDDEN * STATE + offsets * STATE
    for state_index in tl.static_range(0, STATE):
        compress = tl.load(
            compress_ptr + factor_base + state_index, mask=mask, other=0.0
        ).to(tl.float32)
        state_value = tl.sum(z0 * compress, axis=0)
        bounded_state = 2.0 * tl.sigmoid(2.0 * state_value) - 1.0
        expand = tl.load(
            expand_ptr + factor_base + state_index, mask=mask, other=0.0
        ).to(tl.float32)
        delta += bounded_state * expand
    refined_gate = gate + delta
    output = refined_gate * tl.sigmoid(refined_gate) * up
    tl.store(out_ptr + activation_offsets, output, mask=mask)


def baseline_activation(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(gate)
    elements = gate.numel()
    baseline_activation_kernel[(triton.cdiv(elements, 256),)](
        gate, up, output, elements=elements, BLOCK=256, num_warps=4
    )
    return output


def feedback_activation(
    gate: torch.Tensor,
    up: torch.Tensor,
    compress: torch.Tensor,
    expand: torch.Tensor,
) -> torch.Tensor:
    batch, hidden = gate.shape
    groups, group_hidden, state = compress.shape
    if hidden != groups * group_hidden or expand.shape != compress.shape:
        raise ValueError("inconsistent grouped feedback shapes")
    output = torch.empty_like(gate)
    block_hidden = triton.next_power_of_2(group_hidden)
    feedback_activation_kernel[(batch, groups)](
        gate,
        up,
        compress,
        expand,
        output,
        GROUP_HIDDEN=group_hidden,
        STATE=state,
        GROUPS=groups,
        BLOCK_HIDDEN=block_hidden,
        num_warps=8,
        num_stages=2,
    )
    return output


def baseline_path(
    x: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    g = x @ gate.T
    u = x @ up.T
    return baseline_activation(g, u) @ down.T


def feedback_path(
    x: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    compress: torch.Tensor,
    expand: torch.Tensor,
) -> torch.Tensor:
    g = x @ gate.T
    u = x @ up.T
    return feedback_activation(g, u, compress, expand) @ down.T


def capture(fn: Callable[[], torch.Tensor]) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    torch.cuda.synchronize()
    return graph, output


def timed_replay(graph: torch.cuda.CUDAGraph) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
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
    if hidden % groups:
        raise ValueError("hidden must be divisible by groups")
    group_hidden = hidden // groups
    gate = torch.randn(hidden, dimension, device=device, dtype=dtype) / dimension**0.5
    up = torch.randn(hidden, dimension, device=device, dtype=dtype) / dimension**0.5
    down = torch.randn(dimension, hidden, device=device, dtype=dtype) / hidden**0.5
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
    numerator = torch.linalg.vector_norm(actual.float() - expected.float(), dim=-1)
    denominator = torch.linalg.vector_norm(expected.float(), dim=-1).clamp_min(1e-12)
    return float((numerator / denominator).max())


def driver_version() -> str:
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).splitlines()[0]


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batches = [int(value) for value in args.batches.split(",")]
    device_name = torch.cuda.get_device_name(0)

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

    cells: list[dict[str, object]] = []
    zero_errors: list[float] = []
    for batch in batches:
        print(f"fused benchmark B={batch}", flush=True)
        x = torch.randn(batch, args.dimension, device=device, dtype=dtype)
        gate, up, down, compress, expand = base_weights
        eq_gate, eq_up, eq_down, eq_compress, eq_expand = equal_weights
        zero_expand = torch.zeros_like(expand)

        base_fn = lambda: baseline_path(x, gate, up, down)
        same_fn = lambda: feedback_path(x, gate, up, down, compress, expand)
        equal_fn = lambda: feedback_path(
            x, eq_gate, eq_up, eq_down, eq_compress, eq_expand
        )
        zero_fn = lambda: feedback_path(x, gate, up, down, compress, zero_expand)

        with torch.inference_mode():
            baseline_check = base_fn().clone()
            zero_check = zero_fn().clone()
            torch.cuda.synchronize()
            zero_errors.append(max_row_relative(zero_check, baseline_check))
            base_graph, _ = capture(base_fn)
            same_graph, _ = capture(same_fn)
            equal_graph, _ = capture(equal_fn)
            for _ in range(args.warmup):
                base_graph.replay()
                same_graph.replay()
                equal_graph.replay()
            torch.cuda.synchronize()

            timings = {"baseline": [], "same_width": [], "equal_parameter": []}
            graphs = {
                "baseline": base_graph,
                "same_width": same_graph,
                "equal_parameter": equal_graph,
            }
            for _ in range(args.samples):
                order = list(graphs)
                random.shuffle(order)
                for name in order:
                    timings[name].append(timed_replay(graphs[name]))

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
                "zero_feedback_max_row_relative_error": zero_errors[-1],
            }
        )

    protocol_valid = (
        "H100" in device_name
        and args.dimension == 4096
        and args.hidden == 14336
        and args.state == 8
        and args.groups == 8
        and set(batches) == {1, 8, 32, 128}
        and args.warmup >= 40
        and args.samples >= 200
        and equal_hidden == 14272
    )
    key_cells = [cell for cell in cells if cell["batch"] in {1, 8}]
    gates = {
        "protocol_valid": protocol_valid,
        "zero_feedback_inclusion_below_0p2_percent": max(zero_errors) <= 0.002,
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
    prereg = Path("results/feedback-swiglu-fused-h100-preregistration.md")
    return {
        "candidate": "group-local Feedback-SwiGLU",
        "scope": "fused single-H100 serving gate only",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "preregistration_sha256": hashlib.sha256(prereg.read_bytes()).hexdigest(),
        "args": {**vars(args), "output": str(args.output)},
        "device": device_name,
        "device_properties": str(torch.cuda.get_device_properties(0)),
        "driver_version": driver_version(),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "cuda_version": torch.version.cuda,
        "kernel": {
            "baseline_block": 256,
            "feedback_block_hidden_same_width":
                triton.next_power_of_2(args.hidden // args.groups),
            "feedback_block_hidden_equal_parameter":
                triton.next_power_of_2(equal_hidden // args.groups),
            "feedback_num_warps": 8,
            "feedback_num_stages": 2,
            "cuda_graph_full_path": True,
        },
        "ledger": {
            "baseline_hidden": args.hidden,
            "equal_parameter_hidden": equal_hidden,
            "feedback_groups": args.groups,
            "states_per_token": args.groups * args.state,
            "same_width_matrix_mac_overhead_fraction":
                2 * args.state / (3 * args.dimension),
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
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/feedback-swiglu-fused-h100.json"),
    )
    args = parser.parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": payload["gates"], "all_gates_pass": payload["all_gates_pass"]}, indent=2))
    raise SystemExit(0 if payload["all_gates_pass"] else 1)


if __name__ == "__main__":
    main()

