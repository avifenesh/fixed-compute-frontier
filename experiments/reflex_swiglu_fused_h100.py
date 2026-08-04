#!/usr/bin/env python3
"""Fused H100 serving gate for Reflex-SwiGLU."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Callable

import torch
import triton
import triton.language as tl


@triton.jit
def baseline_kernel(gate_ptr, up_ptr, out_ptr, elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, gate * tl.sigmoid(gate) * up, mask=mask)


@triton.jit
def reflex_kernel(
    gate_ptr,
    up_ptr,
    alpha_ptr,
    out_ptr,
    elements: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    alpha_offsets = offsets % hidden
    alpha = tl.load(alpha_ptr + alpha_offsets, mask=mask, other=0.0).to(tl.float32)
    z0 = gate * tl.sigmoid(gate) * up
    bounded = tl.maximum(-1.0, tl.minimum(1.0, z0))
    refined_gate = gate + alpha * bounded
    tl.store(
        out_ptr + offsets,
        refined_gate * tl.sigmoid(refined_gate) * up,
        mask=mask,
    )


def baseline_activation(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(gate)
    baseline_kernel[(triton.cdiv(gate.numel(), 256),)](
        gate, up, output, elements=gate.numel(), BLOCK=256, num_warps=4
    )
    return output


def reflex_activation(
    gate: torch.Tensor, up: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    output = torch.empty_like(gate)
    reflex_kernel[(triton.cdiv(gate.numel(), 256),)](
        gate,
        up,
        alpha,
        output,
        elements=gate.numel(),
        hidden=gate.shape[-1],
        BLOCK=256,
        num_warps=4,
    )
    return output


def baseline_path(x, gate, up, down):
    return baseline_activation(x @ gate.T, x @ up.T) @ down.T


def reflex_path(x, gate, up, down, alpha):
    return reflex_activation(x @ gate.T, x @ up.T, alpha) @ down.T


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


def make_weights(dimension, hidden, dtype, device):
    gate = torch.randn(hidden, dimension, device=device, dtype=dtype) / dimension**0.5
    up = torch.randn(hidden, dimension, device=device, dtype=dtype) / dimension**0.5
    down = torch.randn(dimension, hidden, device=device, dtype=dtype) / hidden**0.5
    alpha = torch.empty(hidden, device=device, dtype=dtype).uniform_(-0.5, 0.5)
    return gate, up, down, alpha


def max_row_relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(actual.float() - expected.float(), dim=-1)
    denominator = torch.linalg.vector_norm(expected.float(), dim=-1).clamp_min(1e-12)
    return float((numerator / denominator).max())


def nonzero_alpha_reference_check(device: torch.device) -> dict[str, float | int]:
    base = torch.linspace(-4.0, 4.0, 256, device=device, dtype=torch.float32)
    gate = torch.stack((base, base, base))
    up = torch.stack(
        (
            torch.full_like(base, 4.0),
            torch.full_like(base, 0.1),
            torch.full_like(base, -4.0),
        )
    )
    alpha = torch.linspace(-1.75, 1.75, 256, device=device, dtype=torch.float32)
    z0 = torch.nn.functional.silu(gate) * up
    expected = torch.nn.functional.silu(gate + alpha * z0.clamp(-1.0, 1.0)) * up
    actual = reflex_activation(
        gate.to(torch.bfloat16), up.to(torch.bfloat16), alpha.to(torch.bfloat16)
    ).float()
    torch.cuda.synchronize()
    return {
        "max_row_relative_error": max_row_relative(actual, expected),
        "negative_clipped_coordinates": int((z0 < -1.0).sum()),
        "unclipped_coordinates": int(((z0 >= -1.0) & (z0 <= 1.0)).sum()),
        "positive_clipped_coordinates": int((z0 > 1.0).sum()),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batches = [int(value) for value in args.batches.split(",")]
    exact_equal_hidden = (3 * args.dimension * args.hidden) // (3 * args.dimension + 1)
    equal_hidden = (exact_equal_hidden // args.hidden_alignment) * args.hidden_alignment
    base_weights = make_weights(args.dimension, args.hidden, dtype, device)
    equal_weights = make_weights(args.dimension, equal_hidden, dtype, device)
    nonzero_reference = nonzero_alpha_reference_check(device)

    cells = []
    zero_errors = []
    for batch in batches:
        print(f"reflex benchmark B={batch}", flush=True)
        x = torch.randn(batch, args.dimension, device=device, dtype=dtype)
        gate, up, down, alpha = base_weights
        eq_gate, eq_up, eq_down, eq_alpha = equal_weights
        zero_alpha = torch.zeros_like(alpha)
        base_fn = lambda: baseline_path(x, gate, up, down)
        same_fn = lambda: reflex_path(x, gate, up, down, alpha)
        equal_fn = lambda: reflex_path(x, eq_gate, eq_up, eq_down, eq_alpha)
        zero_fn = lambda: reflex_path(x, gate, up, down, zero_alpha)
        with torch.inference_mode():
            base_check = base_fn().clone()
            zero_check = zero_fn().clone()
            torch.cuda.synchronize()
            zero_errors.append(max_row_relative(zero_check, base_check))
            base_graph, _ = capture(base_fn)
            same_graph, _ = capture(same_fn)
            equal_graph, _ = capture(equal_fn)
            for _ in range(args.warmup):
                base_graph.replay(); same_graph.replay(); equal_graph.replay()
            torch.cuda.synchronize()
            timings = {"baseline": [], "same_width": [], "equal_parameter": []}
            graphs = {"baseline": base_graph, "same_width": same_graph, "equal_parameter": equal_graph}
            for _ in range(args.samples):
                order = list(graphs); random.shuffle(order)
                for name in order:
                    timings[name].append(timed_replay(graphs[name]))
        summaries = {name: timing_summary(values) for name, values in timings.items()}
        baseline_median = float(summaries["baseline"]["median_us"])
        cells.append({
            "batch": batch,
            "timings": summaries,
            "same_width_over_baseline_median": float(summaries["same_width"]["median_us"]) / baseline_median,
            "equal_parameter_over_baseline_median": float(summaries["equal_parameter"]["median_us"]) / baseline_median,
            "zero_reflex_max_row_relative_error": zero_errors[-1],
        })

    device_name = torch.cuda.get_device_name(0)
    protocol_valid = (
        "H100" in device_name and args.dimension == 4096 and args.hidden == 14336
        and set(batches) == {1, 8, 32, 128} and args.warmup >= 40 and args.samples >= 200
        and equal_hidden == 14328
    )
    key_cells = [cell for cell in cells if cell["batch"] in {1, 8}]
    gates = {
        "protocol_valid": protocol_valid,
        "zero_reflex_inclusion_below_0p2_percent": max(zero_errors) <= 0.002,
        "nonzero_reflex_reference_below_1_percent":
            nonzero_reference["max_row_relative_error"] <= 0.01
            and nonzero_reference["negative_clipped_coordinates"] > 0
            and nonzero_reference["unclipped_coordinates"] > 0
            and nonzero_reference["positive_clipped_coordinates"] > 0,
        "same_width_within_1p02_on_key_cells": all(cell["same_width_over_baseline_median"] <= 1.02 for cell in key_cells),
        "same_width_within_1p05_on_full_grid": all(cell["same_width_over_baseline_median"] <= 1.05 for cell in cells),
        "equal_parameter_within_1p02_on_key_cells": all(cell["equal_parameter_over_baseline_median"] <= 1.02 for cell in key_cells),
        "equal_parameter_within_1p05_on_full_grid": all(cell["equal_parameter_over_baseline_median"] <= 1.05 for cell in cells),
    }
    source = Path(__file__)
    prereg = Path("results/reflex-swiglu-fused-h100-preregistration.md")
    return {
        "candidate": "Reflex-SwiGLU",
        "scope": "fused single-H100 serving gate only",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "preregistration_sha256": hashlib.sha256(prereg.read_bytes()).hexdigest(),
        "args": {**vars(args), "output": str(args.output)},
        "device": device_name,
        "device_properties": str(torch.cuda.get_device_properties(0)),
        "driver_version": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).splitlines()[0],
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "cuda_version": torch.version.cuda,
        "kernel": {"block": 256, "num_warps": 4, "cuda_graph_full_path": True},
        "nonzero_alpha_reference": nonzero_reference,
        "ledger": {
            "baseline_hidden": args.hidden,
            "exact_equal_parameter_hidden": exact_equal_hidden,
            "aligned_equal_parameter_hidden": equal_hidden,
            "same_width_parameter_overhead_fraction": 1.0 / (3.0 * args.dimension),
        },
        "cells": cells,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=14336)
    parser.add_argument("--hidden-alignment", type=int, default=8)
    parser.add_argument("--batches", default="1,8,32,128")
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--output", type=Path, default=Path("results/reflex-swiglu-fused-h100.json"))
    args = parser.parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": payload["gates"], "all_gates_pass": payload["all_gates_pass"]}, indent=2))
    raise SystemExit(0 if payload["all_gates_pass"] else 1)


if __name__ == "__main__":
    main()
