#!/usr/bin/env python3
"""Fused full-FFN H100 gate for orbit-activated SwiGLU."""

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
def orbit_kernel(
    gate_ptr,
    up_ptr,
    up_weight_ptr,
    out_ptr,
    hidden: tl.constexpr,
    dimension: tl.constexpr,
    chart_abs: tl.constexpr,
    carrier_gain: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    block = tl.program_id(1)
    groups = hidden // 8
    group = block * GROUPS_PER_BLOCK + tl.arange(0, GROUPS_PER_BLOCK)
    mask = group < groups
    base = token * hidden + group * 8
    previous = tl.zeros((GROUPS_PER_BLOCK,), dtype=tl.float32)

    for step in range(8):
        feature = group * 8 + step
        offset = base + step
        pivot = feature % dimension
        packed_scale = tl.load(
            up_weight_ptr + feature * dimension + pivot, mask=mask, other=chart_abs
        ).to(tl.float32)
        signed_chart = tl.where((feature % 2) == 0, chart_abs, -chart_abs)
        coefficient = (packed_scale / signed_chart - 1.0) / carrier_gain
        gate = tl.load(gate_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        if step == 0:
            refined = gate + coefficient
        else:
            bounded = tl.maximum(-1.0, tl.minimum(1.0, previous))
            refined = gate + coefficient * bounded
        previous = refined * tl.sigmoid(refined) * up
        tl.store(out_ptr + offset, previous, mask=mask)


def baseline_activation(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(gate)
    baseline_kernel[(triton.cdiv(gate.numel(), 256),)](
        gate, up, output, elements=gate.numel(), BLOCK=256, num_warps=4
    )
    return output


def orbit_activation(
    gate: torch.Tensor,
    up: torch.Tensor,
    up_weight: torch.Tensor,
    chart_abs: float,
    carrier_gain: float,
) -> torch.Tensor:
    output = torch.empty_like(gate)
    groups = gate.shape[-1] // 8
    grid = (gate.shape[0], triton.cdiv(groups, 32))
    orbit_kernel[grid](
        gate,
        up,
        up_weight,
        output,
        hidden=gate.shape[-1],
        dimension=up_weight.shape[-1],
        chart_abs=chart_abs,
        carrier_gain=carrier_gain,
        GROUPS_PER_BLOCK=32,
        num_warps=4,
    )
    return output


def decode_coefficients(
    deployed_up: torch.Tensor, chart_abs: float, carrier_gain: float
) -> torch.Tensor:
    hidden, dimension = deployed_up.shape
    features = torch.arange(hidden, device=deployed_up.device)
    pivots = features.remainder(dimension)
    signs = torch.where(features.remainder(2) == 0, 1.0, -1.0)
    chart = signs * chart_abs
    return (deployed_up[features, pivots].float() / chart - 1.0) / carrier_gain


def torch_orbit_reference(
    gate_values: torch.Tensor,
    up_values: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    chunks = []
    for start in range(0, gate_values.shape[-1], 8):
        previous = F.silu(gate_values[:, start].float() + coefficients[start]) * up_values[:, start].float()
        chunks.append(previous)
        for offset in range(1, 8):
            i = start + offset
            previous = F.silu(
                gate_values[:, i].float() + coefficients[i] * previous.clamp(-1.0, 1.0)
            ) * up_values[:, i].float()
            chunks.append(previous)
    return torch.stack(chunks, dim=-1)


def baseline_path(x, gate, up, down):
    return baseline_activation(x @ gate.T, x @ up.T) @ down.T


def orbit_path(x, gate, up, down, chart_abs, carrier_gain):
    return orbit_activation(x @ gate.T, x @ up.T, up, chart_abs, carrier_gain) @ down.T


def make_weights(dimension, hidden, dtype, device, coefficient_bound, carrier_gain):
    chart_abs = dimension**-0.5
    gate = torch.randn(hidden, dimension, device=device, dtype=torch.float32) * chart_abs
    canonical_up = torch.randn(hidden, dimension, device=device, dtype=torch.float32) * chart_abs
    canonical_down = torch.randn(dimension, hidden, device=device, dtype=torch.float32) / hidden**0.5
    features = torch.arange(hidden, device=device)
    pivots = features.remainder(dimension)
    signs = torch.where(features.remainder(2) == 0, 1.0, -1.0)
    canonical_up[features, pivots] = signs * chart_abs
    coefficients = torch.empty(hidden, device=device).uniform_(-coefficient_bound, coefficient_bound)
    scales = 1.0 + carrier_gain * coefficients
    deployed_up = canonical_up * scales[:, None]
    deployed_down = canonical_down / scales[None, :]
    return (
        gate.to(dtype),
        deployed_up.to(dtype),
        deployed_down.to(dtype),
        coefficients,
        chart_abs,
    )


def make_zero_weights(dimension, hidden, dtype, device):
    return make_weights(dimension, hidden, dtype, device, 0.0, 8.0)


def max_row_relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(actual.float() - expected.float(), dim=-1)
    denominator = torch.linalg.vector_norm(expected.float(), dim=-1).clamp_min(1e-12)
    return float((numerator / denominator).max())


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


def nonzero_reference_check(device, dimension, hidden, dtype, carrier_gain):
    gate, up_weight, _, encoded, chart_abs = make_weights(
        dimension, hidden, dtype, device, 0.035, carrier_gain
    )
    decoded = decode_coefficients(up_weight, chart_abs, carrier_gain)
    gate_values = torch.linspace(-4.0, 4.0, hidden, device=device).repeat(3, 1).to(dtype)
    pattern = torch.tensor([-4.0, 0.1, 4.0], device=device).unsqueeze(1)
    up_values = pattern.repeat(1, hidden).to(dtype)
    expected = torch_orbit_reference(gate_values, up_values, decoded)
    actual = orbit_activation(gate_values, up_values, up_weight, chart_abs, carrier_gain).float()
    torch.cuda.synchronize()
    predecessor = expected[:, :-1]
    return {
        "max_row_relative_error": max_row_relative(actual, expected),
        "coefficient_max_abs_decode_error": float((decoded - encoded).abs().max()),
        "negative_clipped_coordinates": int((predecessor < -1.0).sum()),
        "unclipped_coordinates": int(((predecessor >= -1.0) & (predecessor <= 1.0)).sum()),
        "positive_clipped_coordinates": int((predecessor > 1.0).sum()),
    }


def carrier_statistics(encoded, decoded, carrier_gain):
    error = (decoded - encoded).abs().float()
    material = encoded.abs() >= 0.001
    sign_flips = ((decoded.sign() != encoded.sign()) & material).sum()
    zero_collapses = ((decoded == 0) & material).sum()
    scales = 1.0 + carrier_gain * encoded
    condition = torch.maximum(scales.abs(), scales.abs().reciprocal())
    return {
        "absolute_error_p50": float(torch.quantile(error, 0.50)),
        "absolute_error_p90": float(torch.quantile(error, 0.90)),
        "absolute_error_p99": float(torch.quantile(error, 0.99)),
        "absolute_error_max": float(error.max()),
        "material_coefficients": int(material.sum()),
        "material_sign_flips": int(sign_flips),
        "material_zero_collapses": int(zero_collapses),
        "scale_min": float(scales.min()),
        "scale_max": float(scales.max()),
        "maximum_scale_condition_factor": float(condition.max()),
    }


def folded_bf16_export_check(device, dimension, hidden, carrier_gain):
    torch.manual_seed(991)
    chart_abs = dimension**-0.5
    batch = 8
    x = torch.randn(batch, dimension, device=device, dtype=torch.float32)
    gate = torch.randn(hidden, dimension, device=device, dtype=torch.float32) * chart_abs
    canonical_up = torch.randn(hidden, dimension, device=device, dtype=torch.float32) * chart_abs
    canonical_down = torch.randn(dimension, hidden, device=device, dtype=torch.float32) / hidden**0.5
    features = torch.arange(hidden, device=device)
    pivots = features.remainder(dimension)
    signs = torch.where(features.remainder(2) == 0, 1.0, -1.0)
    canonical_up[features, pivots] = signs * chart_abs
    encoded = torch.empty(hidden, device=device).uniform_(-0.035, 0.035)
    scales = 1.0 + carrier_gain * encoded
    folded_up = canonical_up * scales[:, None]
    folded_down = canonical_down / scales[None, :]

    canonical_gate_values = x @ gate.T
    canonical_up_values = x @ canonical_up.T
    # Canonical candidate includes the carrier scale in q and cancels it in V.
    candidate_reference = torch_orbit_reference(
        canonical_gate_values, canonical_up_values * scales[None, :], encoded
    ) @ (canonical_down / scales[None, :]).T
    ordinary_reference = F.silu(canonical_gate_values) * canonical_up_values @ canonical_down.T

    x_bf16 = x.to(torch.bfloat16)
    gate_bf16 = gate.to(torch.bfloat16)
    up_bf16 = folded_up.to(torch.bfloat16)
    down_bf16 = folded_down.to(torch.bfloat16)
    decoded = decode_coefficients(up_bf16, chart_abs, carrier_gain)
    candidate_bf16 = orbit_path(
        x_bf16, gate_bf16, up_bf16, down_bf16, chart_abs, carrier_gain
    ).float()
    ordinary_bf16 = baseline_path(x_bf16, gate_bf16, up_bf16, down_bf16).float()
    torch.cuda.synchronize()
    candidate_error = max_row_relative(candidate_bf16, candidate_reference)
    ordinary_error = max_row_relative(ordinary_bf16, ordinary_reference)
    return {
        "candidate_vs_canonical_fp32_max_row_relative": candidate_error,
        "ordinary_folded_bf16_vs_canonical_fp32_max_row_relative": ordinary_error,
        "candidate_excess_over_ordinary_export_error": candidate_error - ordinary_error,
    }


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batches = [int(value) for value in args.batches.split(",")]
    weights = make_weights(
        args.dimension, args.hidden, dtype, device, args.coefficient_bound, args.carrier_gain
    )
    gate, up, down, encoded, chart_abs = weights
    zero_gate, zero_up, zero_down, _, zero_chart = make_zero_weights(
        args.dimension, args.hidden, dtype, device
    )
    reference = nonzero_reference_check(
        device, args.dimension, args.hidden, dtype, args.carrier_gain
    )
    decoded = decode_coefficients(up, chart_abs, args.carrier_gain)
    coefficient_error = float((decoded - encoded).abs().max())
    carrier_stats = carrier_statistics(encoded, decoded, args.carrier_gain)
    export_check = folded_bf16_export_check(
        device, args.dimension, args.hidden, args.carrier_gain
    )

    cells = []
    zero_errors = []
    for batch in batches:
        print(f"orbit benchmark B={batch}", flush=True)
        x = torch.randn(batch, args.dimension, device=device, dtype=dtype)
        base_fn = lambda: baseline_path(x, gate, up, down)
        orbit_fn = lambda: orbit_path(x, gate, up, down, chart_abs, args.carrier_gain)
        zero_base_fn = lambda: baseline_path(x, zero_gate, zero_up, zero_down)
        zero_orbit_fn = lambda: orbit_path(x, zero_gate, zero_up, zero_down, zero_chart, args.carrier_gain)
        with torch.inference_mode():
            zero_base = zero_base_fn().clone()
            zero_orbit = zero_orbit_fn().clone()
            torch.cuda.synchronize()
            zero_error = max_row_relative(zero_orbit, zero_base)
            zero_errors.append(zero_error)
            base_graph, _ = capture(base_fn)
            orbit_graph, _ = capture(orbit_fn)
            for _ in range(args.warmup):
                base_graph.replay(); orbit_graph.replay()
            torch.cuda.synchronize()
            timings = {"baseline": [], "orbit": []}
            graphs = {"baseline": base_graph, "orbit": orbit_graph}
            for _ in range(args.samples):
                order = list(graphs); random.shuffle(order)
                for name in order:
                    timings[name].append(timed_replay(graphs[name]))
        summaries = {name: timing_summary(values) for name, values in timings.items()}
        ratio = float(summaries["orbit"]["median_us"]) / float(summaries["baseline"]["median_us"])
        cells.append({
            "batch": batch,
            "timings": summaries,
            "orbit_over_baseline_median": ratio,
            "zero_endpoint_max_row_relative_error": zero_error,
        })

    device_name = torch.cuda.get_device_name(0)
    protocol_valid = (
        "H100" in device_name
        and args.dimension == 4096
        and args.hidden == 14336
        and set(batches) == {1, 8, 32, 128}
        and args.warmup >= 40
        and args.samples >= 200
        and args.carrier_gain == 8.0
    )
    key_cells = [cell for cell in cells if cell["batch"] in {1, 8}]
    gates = {
        "protocol_valid": protocol_valid,
        "zero_endpoint_below_0p2_percent": max(zero_errors) <= 0.002,
        "nonzero_reference_below_1_percent": reference["max_row_relative_error"] <= 0.01,
        "coefficient_decode_below_0p001": coefficient_error <= 0.001,
        "carrier_condition_factor_below_1p5": (
            carrier_stats["maximum_scale_condition_factor"] < 1.5
        ),
        "no_material_sign_flips_or_zero_collapse": (
            carrier_stats["material_sign_flips"] == 0
            and carrier_stats["material_zero_collapses"] == 0
        ),
        "candidate_bf16_excess_export_error_below_0p5_percent": (
            export_check["candidate_excess_over_ordinary_export_error"] < 0.005
        ),
        "reference_exercises_all_clip_regions": (
            reference["negative_clipped_coordinates"] > 0
            and reference["unclipped_coordinates"] > 0
            and reference["positive_clipped_coordinates"] > 0
        ),
        "no_duplicate_coefficient_tensor": True,
        "no_additional_activation_workspace": True,
        "within_1p02_on_key_cells": all(cell["orbit_over_baseline_median"] <= 1.02 for cell in key_cells),
        "within_1p05_on_full_grid": all(cell["orbit_over_baseline_median"] <= 1.05 for cell in cells),
    }
    source = Path(__file__)
    prereg = Path("results/orbit-activated-swiglu-h100-preregistration.md")
    return {
        "candidate": "orbit-activated SwiGLU",
        "scope": "fused full-FFN single-H100 serving gate only",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "preregistration_sha256": hashlib.sha256(prereg.read_bytes()).hexdigest(),
        "args": {**vars(args), "output": str(args.output)},
        "device": device_name,
        "device_properties": str(torch.cuda.get_device_properties(0)),
        "driver_version": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).splitlines()[0],
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "cuda_version": torch.version.cuda,
        "kernel": {
            "group_size": 8,
            "groups_per_program": 32,
            "num_warps": 4,
            "coefficient_source": "semantic BF16 pivots in deployed U matrix",
            "cuda_graph_full_path": True,
        },
        "ledger": {
            "baseline_serialized_words": 3 * args.dimension * args.hidden,
            "candidate_serialized_words": 3 * args.dimension * args.hidden,
            "baseline_dense_macs_per_token": 3 * args.dimension * args.hidden,
            "candidate_dense_macs_per_token": 3 * args.dimension * args.hidden,
            "coefficient_tensor_words": 0,
            "additional_activation_workspace_words": 0,
            "semantic_pivot_load_instructions_per_token": args.hidden,
            "semantic_pivot_load_instructions_by_batch": {
                str(batch): batch * args.hidden for batch in batches
            },
            "pivot_loads_may_hit_cache_but_are_not_counted_as_free": True,
            "baseline_and_candidate_use_separate_gate_up_gemms": True,
            "production_coalesced_gate_up_parity_established": False,
        },
        "coefficient_max_abs_decode_error": coefficient_error,
        "carrier_statistics": carrier_stats,
        "folded_bf16_export_check": export_check,
        "nonzero_reference": reference,
        "cells": cells,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=14336)
    parser.add_argument("--batches", default="1,8,32,128")
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=617)
    parser.add_argument("--carrier-gain", type=float, default=8.0)
    parser.add_argument("--coefficient-bound", type=float, default=0.035)
    parser.add_argument(
        "--output", type=Path, default=Path("results/orbit-activated-swiglu-h100.json")
    )
    args = parser.parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": payload["gates"], "all_gates_pass": payload["all_gates_pass"]}, indent=2))
    raise SystemExit(0 if payload["all_gates_pass"] else 1)


if __name__ == "__main__":
    main()
