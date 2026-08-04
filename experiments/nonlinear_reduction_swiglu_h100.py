#!/usr/bin/env python3
"""Fused H100 fatal gate for nonlinear-reduction SwiGLU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import median
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

from experiments.self_product_ffn_scale_fused_h100 import (
    swiglu_inplace,
    swiglu_packed_inplace,
)


OUTPUT = Path("results/nonlinear-reduction-swiglu-h100.json")
PREREGISTRATION = Path("results/nonlinear-reduction-swiglu-h100-preregistration.md")
SOURCE = Path(__file__)
ROWS = (1, 8, 32, 128, 512, 2048)
D = 4096
M = 14336
BRANCHES = 4
ALPHA = 0.05
SEED = 431


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_R": 16, "BLOCK_M": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_R": 32, "BLOCK_M": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_R": 32, "BLOCK_M": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    ],
    key=["rows", "hidden", "input_width", "ASSOCIATIVE"],
)
@triton.jit
def nonlinear_reduction_kernel(
    x_ptr, gate_ptr, up_ptr, out_ptr,
    rows, hidden: tl.constexpr, input_width: tl.constexpr,
    alpha,
    ASSOCIATIVE: tl.constexpr,
    BLOCK_R: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    program = tl.program_id(0)
    tiles_m = tl.cdiv(hidden, BLOCK_M)
    tile_r = program // tiles_m
    tile_m = program - tile_r * tiles_m
    offsets_r = tile_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offsets_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    branch_width: tl.constexpr = input_width // 4
    accumulator_gate = tl.zeros((BLOCK_R, BLOCK_M), tl.float32)
    accumulator_up = tl.zeros((BLOCK_R, BLOCK_M), tl.float32)

    if ASSOCIATIVE:
        for branch in tl.static_range(0, 4):
            block_gate = tl.zeros((BLOCK_R, BLOCK_M), tl.float32)
            block_up = tl.zeros((BLOCK_R, BLOCK_M), tl.float32)
            for inner in range(0, branch_width, BLOCK_K):
                offsets_k = branch * branch_width + inner + tl.arange(0, BLOCK_K)
                x = tl.load(
                    x_ptr + offsets_r[:, None] * input_width + offsets_k[None, :],
                    mask=(offsets_r[:, None] < rows) & (offsets_k[None, :] < input_width),
                    other=0.0,
                )
                weight_offsets = offsets_m[None, :] * input_width + offsets_k[:, None]
                gate = tl.load(
                    gate_ptr + weight_offsets,
                    mask=(offsets_m[None, :] < hidden) & (offsets_k[:, None] < input_width),
                    other=0.0,
                )
                up = tl.load(
                    up_ptr + weight_offsets,
                    mask=(offsets_m[None, :] < hidden) & (offsets_k[:, None] < input_width),
                    other=0.0,
                )
                block_gate += tl.dot(x, gate)
                block_up += tl.dot(x, up)
            previous_gate = accumulator_gate
            previous_up = accumulator_up
            accumulator_gate = previous_gate + block_gate + alpha * previous_gate * block_gate
            accumulator_up = previous_up + block_up + alpha * (
                previous_gate * block_up + previous_up * block_gate
            )
    else:
        for branch in tl.static_range(0, 4):
            for inner in range(0, branch_width, BLOCK_K):
                offsets_k = branch * branch_width + inner + tl.arange(0, BLOCK_K)
                x = tl.load(
                    x_ptr + offsets_r[:, None] * input_width + offsets_k[None, :],
                    mask=(offsets_r[:, None] < rows) & (offsets_k[None, :] < input_width),
                    other=0.0,
                )
                weight_offsets = offsets_m[None, :] * input_width + offsets_k[:, None]
                gate = tl.load(
                    gate_ptr + weight_offsets,
                    mask=(offsets_m[None, :] < hidden) & (offsets_k[:, None] < input_width),
                    other=0.0,
                )
                up = tl.load(
                    up_ptr + weight_offsets,
                    mask=(offsets_m[None, :] < hidden) & (offsets_k[:, None] < input_width),
                    other=0.0,
                )
                accumulator_gate += tl.dot(x, gate)
                accumulator_up += tl.dot(x, up)

    sigmoid = 1.0 / (1.0 + libdevice.exp(-accumulator_gate))
    output = (accumulator_gate * sigmoid) * accumulator_up
    tl.store(
        out_ptr + offsets_r[:, None] * hidden + offsets_m[None, :],
        output,
        mask=(offsets_r[:, None] < rows) & (offsets_m[None, :] < hidden),
    )


def nonlinear_reduction(
    inputs: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    alpha: float,
    associative_mode: bool,
) -> torch.Tensor:
    if not inputs.is_contiguous() or not gate.is_contiguous() or not up.is_contiguous():
        raise ValueError("contiguous tensors required")
    rows, input_width = inputs.shape
    hidden = gate.shape[0]
    if gate.shape != up.shape or gate.shape[1] != input_width:
        raise ValueError("shape mismatch")
    if input_width % BRANCHES or (input_width // BRANCHES) % 64:
        raise ValueError("four tensor-core-aligned K blocks required")
    output = torch.empty((rows, hidden), device=inputs.device, dtype=inputs.dtype)
    grid = lambda meta: (
        triton.cdiv(rows, meta["BLOCK_R"]) * triton.cdiv(hidden, meta["BLOCK_M"]),
    )
    nonlinear_reduction_kernel[grid](
        inputs, gate, up, output,
        rows=rows, hidden=hidden, input_width=input_width,
        alpha=alpha, ASSOCIATIVE=associative_mode,
    )
    return output


def reference(
    inputs: torch.Tensor, gate: torch.Tensor, up: torch.Tensor,
    alpha: float, associative_mode: bool,
) -> torch.Tensor:
    source = inputs.float()
    gate_float = gate.float()
    up_float = up.float()
    p = torch.zeros((inputs.shape[0], gate.shape[0]), device=inputs.device)
    q = torch.zeros_like(p)
    width = inputs.shape[1] // BRANCHES
    for branch in range(BRANCHES):
        start, stop = branch * width, (branch + 1) * width
        block_gate = source[:, start:stop] @ gate_float[:, start:stop].T
        block_up = source[:, start:stop] @ up_float[:, start:stop].T
        if associative_mode:
            previous_p, previous_q = p, q
            p = previous_p + block_gate + alpha * previous_p * block_gate
            q = previous_q + block_up + alpha * (
                previous_p * block_up + previous_q * block_gate
            )
        else:
            p += block_gate
            q += block_up
    return (F.silu(p) * q).to(inputs.dtype)


def error_record(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    difference = actual.float() - expected.float()
    return {
        "all_finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
        "relative_l2": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(expected.float()).clamp_min(1e-30)
        ),
        "max_absolute": float(difference.abs().max()),
    }


def capture(function: Callable[[], torch.Tensor]) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = function()
    return graph, output


def event_time(graph: torch.cuda.CUDAGraph) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop))


def bootstrap_ratio(
    numerator: list[float], denominator: list[float], seed: int,
) -> dict[str, float]:
    numerator_values = np.asarray(numerator, dtype=np.float64)
    denominator_values = np.asarray(denominator, dtype=np.float64)
    generator = np.random.default_rng(seed)
    ratios = np.empty(5000, dtype=np.float64)
    for index in range(ratios.size):
        sample = generator.integers(0, numerator_values.size, numerator_values.size)
        ratios[index] = np.median(numerator_values[sample]) / np.median(denominator_values[sample])
    return {
        "median_ratio": float(np.median(numerator_values) / np.median(denominator_values)),
        "lower_95": float(np.quantile(ratios, 0.025)),
        "upper_95": float(np.quantile(ratios, 0.975)),
    }


def peak_bytes(function: Callable[[], torch.Tensor]) -> int:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    output = function()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del output
    return int(peak - before)


def run() -> dict[str, Any]:
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    if torch.__version__ != "2.5.1+cu124" or torch.version.cuda != "12.4":
        raise RuntimeError("frozen PyTorch/CUDA runtime required")
    if triton.__version__ != "3.6.0":
        raise RuntimeError("isolated Triton 3.6.0 required")
    triton_path = str(Path(triton.__file__).resolve())
    if not triton_path.startswith("/workspace/triton36/"):
        raise RuntimeError(f"wrong Triton installation: {triton_path}")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")

    small_x = (torch.randn(16, 256, device=device) / math.sqrt(256)).to(torch.bfloat16)
    small_gate = torch.randn(256, 256, device=device, dtype=torch.bfloat16)
    small_up = torch.randn(256, 256, device=device, dtype=torch.bfloat16)
    small_errors = {}
    for name, associative_mode, alpha in (
        ("disabled", False, 0.0),
        ("associative", True, ALPHA),
    ):
        actual = nonlinear_reduction(small_x, small_gate, small_up, alpha, associative_mode)
        expected = reference(small_x, small_gate, small_up, alpha, associative_mode)
        small_errors[name] = error_record(actual, expected)

    packed = (torch.randn(2 * M, D, device=device) / math.sqrt(D)).to(torch.bfloat16)
    gate, up = packed[:M], packed[M:]
    down = (torch.randn(D, M, device=device) / math.sqrt(M)).to(torch.bfloat16)
    static_weight_scalars = packed.numel() + down.numel()
    expected_weight_scalars = 3 * D * M
    cells = {}
    all_times: dict[str, dict[str, list[float]]] = {}
    memory = {}
    cell_output_dtypes = {}
    cell_input_activation_dtypes = {}
    target_error = None
    target_candidate_error = None

    for row_count in ROWS:
        inputs = torch.randn(row_count, D, device=device, dtype=torch.bfloat16)

        def split_path() -> torch.Tensor:
            gate_values = F.linear(inputs, gate)
            up_values = F.linear(inputs, up)
            activated = swiglu_inplace(gate_values, up_values)
            return F.linear(activated, down)

        def packed_path() -> torch.Tensor:
            values = F.linear(inputs, packed)
            activated = swiglu_packed_inplace(values)
            return F.linear(activated, down)

        def custom_path() -> torch.Tensor:
            activated = nonlinear_reduction(inputs, gate, up, 0.0, False)
            return F.linear(activated, down)

        def candidate_path() -> torch.Tensor:
            activated = nonlinear_reduction(inputs, gate, up, ALPHA, True)
            return F.linear(activated, down)

        functions = {
            "ordinary_split": split_path,
            "ordinary_packed": packed_path,
            "custom_disabled": custom_path,
            "candidate": candidate_path,
        }
        probe_gate = F.linear(inputs, gate)
        probe_up = F.linear(inputs, up)
        probe_split = swiglu_inplace(probe_gate, probe_up)
        probe_packed_values = F.linear(inputs, packed)
        probe_packed = swiglu_packed_inplace(probe_packed_values)
        probe_custom = nonlinear_reduction(inputs, gate, up, 0.0, False)
        probe_candidate = nonlinear_reduction(inputs, gate, up, ALPHA, True)
        cell_input_activation_dtypes[str(row_count)] = {
            "input": str(inputs.dtype),
            "ordinary_split_activation": str(probe_split.dtype),
            "ordinary_packed_activation": str(probe_packed.dtype),
            "custom_disabled_activation": str(probe_custom.dtype),
            "candidate_activation": str(probe_candidate.dtype),
        }
        del probe_gate, probe_up, probe_split, probe_packed_values, probe_packed
        del probe_custom, probe_candidate
        warm_outputs = [function() for function in functions.values()]
        torch.cuda.synchronize()
        output_dtypes = {name: str(output.dtype) for name, output in zip(functions, warm_outputs)}
        cell_output_dtypes[str(row_count)] = output_dtypes
        del warm_outputs
        if row_count == 32:
            custom_activation = nonlinear_reduction(inputs, gate, up, 0.0, False)
            packed_reference = (F.silu(F.linear(inputs, gate).float())
                                * F.linear(inputs, up).float()).to(torch.bfloat16)
            target_error = error_record(custom_activation, packed_reference)
            target_candidate = nonlinear_reduction(inputs, gate, up, ALPHA, True)
            target_candidate_reference = reference(inputs, gate, up, ALPHA, True)
            target_candidate_error = error_record(target_candidate, target_candidate_reference)
            del target_candidate, target_candidate_reference

        memory[str(row_count)] = {name: peak_bytes(function) for name, function in functions.items()}
        graphs = {name: capture(function)[0] for name, function in functions.items()}
        for _ in range(30):
            for name in graphs:
                graphs[name].replay()
        torch.cuda.synchronize()
        times = {name: [] for name in graphs}
        randomizer = random.Random(SEED + row_count)
        for _ in range(100):
            order = list(graphs)
            randomizer.shuffle(order)
            for name in order:
                times[name].append(event_time(graphs[name]))
        medians = {name: median(values) for name, values in times.items()}
        fastest_name = min(("ordinary_split", "ordinary_packed"), key=medians.get)
        cells[str(row_count)] = {
            "median_ms": medians,
            "fastest_ordinary": fastest_name,
            "candidate_vs_custom": bootstrap_ratio(
                times["candidate"], times["custom_disabled"], SEED + row_count + 1,
            ),
            "candidate_vs_fastest_ordinary": bootstrap_ratio(
                times["candidate"], times[fastest_name], SEED + row_count + 2,
            ),
        }
        all_times[str(row_count)] = times
        del graphs, inputs
        torch.cuda.empty_cache()

    correctness_gate = all(
        record["all_finite"] and record["relative_l2"] <= 0.01
        for record in small_errors.values()
    )
    target_gate = bool(
        target_error and target_error["all_finite"] and target_error["relative_l2"] <= 0.01
    )
    target_candidate_gate = bool(
        target_candidate_error and target_candidate_error["all_finite"]
        and target_candidate_error["relative_l2"] <= 0.01
    )
    timing_vectors_valid = all(
        len(values) == 100
        and all(math.isfinite(value) and value > 0.0 for value in values)
        for times in all_times.values() for values in times.values()
    )
    dtype_ledger = {
        "packed_weight": str(packed.dtype),
        "down_weight": str(down.dtype),
        "cell_inputs_and_activations": cell_input_activation_dtypes,
        "cell_outputs": cell_output_dtypes,
    }
    gates = {
        "small_reference_correct": correctness_gate,
        "target_disabled_matches_ordinary": target_gate,
        "target_associative_matches_reference": target_candidate_gate,
        "static_weight_and_state_ledger_exact": (
            static_weight_scalars == expected_weight_scalars
            and static_weight_scalars * 2 == 3 * D * M * 2
            and dtype_ledger["packed_weight"] == "torch.bfloat16"
            and dtype_ledger["down_weight"] == "torch.bfloat16"
            and all(dtype_name == "torch.bfloat16"
                    for cell in cell_output_dtypes.values() for dtype_name in cell.values())
            and all(dtype_name == "torch.bfloat16"
                    for cell in cell_input_activation_dtypes.values()
                    for dtype_name in cell.values())
            and all(
                memory[str(rows)]["candidate"] <= max(
                    memory[str(rows)]["ordinary_split"],
                    memory[str(rows)]["ordinary_packed"],
                )
                for rows in ROWS
            )
        ),
        "raw_timing_vectors_valid": timing_vectors_valid,
        "candidate_within_custom_latency_envelope": all(
            cell["candidate_vs_custom"]["median_ratio"] <= 1.01
            and cell["candidate_vs_custom"]["upper_95"] <= 1.02
            for cell in cells.values()
        ),
        "candidate_within_fastest_ordinary_latency_envelope": all(
            cell["candidate_vs_fastest_ordinary"]["median_ratio"] <= 1.00
            and cell["candidate_vs_fastest_ordinary"]["upper_95"] <= 1.02
            for cell in cells.values()
        ),
        "candidate_peak_no_more_than_both_ordinary": all(
            memory[str(rows)]["candidate"] <= memory[str(rows)]["ordinary_split"]
            and memory[str(rows)]["candidate"] <= memory[str(rows)]["ordinary_packed"]
            for rows in ROWS
        ),
    }
    finite = all(
        math.isfinite(number)
        for cell in cells.values()
        for record in (cell["median_ms"], cell["candidate_vs_custom"],
                       cell["candidate_vs_fastest_ordinary"])
        for number in record.values()
        if isinstance(number, (int, float))
    )
    gates["all_timing_evidence_finite"] = finite
    return {
        "schema": "nonlinear-reduction-swiglu-h100-v1",
        "source_sha256": sha256_file(SOURCE),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "device": torch.cuda.get_device_name(0),
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "triton": triton.__version__, "triton_file": triton_path,
                    "numpy": np.__version__},
        "shape": {"input": D, "hidden": M, "branches": BRANCHES, "rows": list(ROWS)},
        "alpha": ALPHA,
        "small_correctness": small_errors,
        "target_correctness": target_error,
        "target_candidate_correctness": target_candidate_error,
        "static_weight_scalars": static_weight_scalars,
        "expected_weight_scalars": expected_weight_scalars,
        "static_weight_bytes": static_weight_scalars * 2,
        "dtype_ledger": dtype_ledger,
        "memory_incremental_bytes": memory,
        "raw_timing_ms": all_times,
        "cells": cells,
        "gates": gates,
        "h100_pass": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": result["gates"], "cells": result["cells"],
                      "h100_pass": result["h100_pass"]}, indent=2, sort_keys=True))
    if not result["h100_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
