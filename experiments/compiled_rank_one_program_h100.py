#!/usr/bin/env python3
"""H100 prototype gate for compiled ordered rank-one programs.

This intentionally benchmarks already-gathered per-token program tensors.  It
isolates the executor question before a later gate adds router and bank-gather
traffic.  Eager and torch.compile results are kept separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F


def make_functions(length: int):
    def additive(x, u, v, bias, scale, packed_coupling):
        del packed_coupling
        base = torch.sum(u * x[:, None, :], dim=-1) + bias
        coefficients = scale * F.silu(base)
        return x + torch.sum(coefficients[:, :, None] * v, dim=1)

    def direct(x, u, v, bias, scale, packed_coupling):
        del packed_coupling
        y = x
        for i in range(length):
            preactivation = torch.sum(y * u[:, i, :], dim=-1) + bias[:, i]
            coefficient = scale[:, i] * F.silu(preactivation)
            y = y + coefficient[:, None] * v[:, i, :]
        return y

    def compiled(x, u, v, bias, scale, packed_coupling):
        base = torch.sum(u * x[:, None, :], dim=-1) + bias
        coefficients = []
        for i in range(length):
            preactivation = base[:, i]
            if i:
                previous = torch.stack(coefficients, dim=1)
                start = i * (i - 1) // 2
                end = i * (i + 1) // 2
                preactivation = preactivation + torch.sum(
                    packed_coupling[:, start:end] * previous, dim=1
                )
            coefficients.append(scale[:, i] * F.silu(preactivation))
        coefficient_matrix = torch.stack(coefficients, dim=1)
        return x + torch.sum(coefficient_matrix[:, :, None] * v, dim=1)

    return {"additive": additive, "direct": direct, "compiled": compiled}


def pack_strict_lower(full_coupling: torch.Tensor) -> torch.Tensor:
    length = full_coupling.shape[1]
    if length <= 1:
        return full_coupling.new_empty((full_coupling.shape[0], 0))
    return torch.cat([full_coupling[:, i, :i] for i in range(1, length)], dim=1)


def make_inputs(batch: int, length: int, dimension: int, dtype, seed: int):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    x = torch.randn(batch, dimension, device="cuda", dtype=dtype, generator=generator)
    u = torch.randn(
        batch, length, dimension, device="cuda", dtype=dtype, generator=generator
    ) / (dimension**0.5)
    v = torch.randn(
        batch, length, dimension, device="cuda", dtype=dtype, generator=generator
    ) / (dimension**0.5)
    bias = 0.2 * torch.randn(
        batch, length, device="cuda", dtype=dtype, generator=generator
    )
    scale = 0.2 + 0.6 * torch.rand(
        batch, length, device="cuda", dtype=dtype, generator=generator
    )
    full_coupling = torch.einsum("bld,bmd->blm", u, v)
    packed_coupling = pack_strict_lower(full_coupling)
    return x, u, v, bias, scale, packed_coupling


def numerical_errors(
    actual: torch.Tensor, expected: torch.Tensor, residual_input: torch.Tensor
) -> dict[str, float]:
    numerator = torch.linalg.vector_norm(actual.float() - expected.float())
    denominator = torch.clamp(torch.linalg.vector_norm(expected.float()), min=1e-12)
    row_numerator = torch.linalg.vector_norm(
        actual.float() - expected.float(), dim=-1
    )
    row_denominator = torch.clamp(
        torch.linalg.vector_norm(expected.float(), dim=-1), min=1e-12
    )
    actual_update = actual.float() - residual_input.float()
    expected_update = expected.float() - residual_input.float()
    update_row_numerator = torch.linalg.vector_norm(
        actual_update - expected_update, dim=-1
    )
    update_row_denominator = torch.clamp(
        torch.linalg.vector_norm(expected_update, dim=-1), min=1e-6
    )
    return {
        "global_relative": float((numerator / denominator).item()),
        "max_row_relative": float(torch.max(row_numerator / row_denominator).item()),
        "max_update_row_relative": float(
            torch.max(update_row_numerator / update_row_denominator).item()
        ),
        "max_update_row_absolute": float(torch.max(update_row_numerator).item()),
    }


def timed_interleaved(functions, inputs, samples: int, warmup: int, seed: int):
    for fn in functions.values():
        for _ in range(warmup):
            fn(*inputs)
    torch.cuda.synchronize()

    schedule = [name for name in functions for _ in range(samples)]
    random.Random(seed).shuffle(schedule)
    values = {name: [] for name in functions}
    for name in schedule:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        functions[name](*inputs)
        end.record()
        end.synchronize()
        values[name].append(float(start.elapsed_time(end) * 1000.0))
    return values


def summarize(values):
    ordered = sorted(values)
    return {
        "samples": len(values),
        "median_us": statistics.median(values),
        "mean_us": statistics.fmean(values),
        "p10_us": ordered[int(0.10 * (len(ordered) - 1))],
        "p90_us": ordered[int(0.90 * (len(ordered) - 1))],
        "min_us": ordered[0],
        "max_us": ordered[-1],
    }


def benchmark_cell(
    *,
    batch: int,
    length: int,
    dimension: int,
    dtype,
    samples: int,
    warmup: int,
    seed: int,
):
    # Each frozen cell intentionally gets its own specialization.  Without a
    # reset, Torch 2.5's default eight-entry Dynamo cache is exhausted when the
    # dtype grid begins, even though every individual graph compiles.
    torch._dynamo.reset()
    inputs = make_inputs(batch, length, dimension, dtype, seed)
    eager = make_functions(length)

    # FP32 direct execution is the numerical reference.  The cached Gram table
    # is recomputed in FP32 from the already-quantized expert tensors.
    fp32_inputs = tuple(t.float() for t in inputs[:-1])
    fp32_full_coupling = torch.einsum(
        "bld,bmd->blm", fp32_inputs[1], fp32_inputs[2]
    )
    fp32_packed_coupling = pack_strict_lower(fp32_full_coupling)
    fp32_full_inputs = (*fp32_inputs, fp32_packed_coupling)
    reference = eager["direct"](*fp32_full_inputs)

    modes = {"eager": eager}
    compiled_functions = {
        name: torch.compile(fn, fullgraph=True, mode="reduce-overhead")
        for name, fn in eager.items()
    }
    # Trigger compilation outside timing.
    for fn in compiled_functions.values():
        fn(*inputs)
    torch.cuda.synchronize()
    modes["compiled"] = compiled_functions

    result = {
        "batch": batch,
        "length": length,
        "dimension": dimension,
        "dtype": str(dtype).removeprefix("torch."),
        "modes": {},
    }
    for mode_index, (mode_name, functions) in enumerate(modes.items()):
        errors = {
            name: numerical_errors(function(*inputs), reference, inputs[0])
            for name, function in functions.items()
        }
        raw_timings = timed_interleaved(
            functions,
            inputs,
            samples=samples,
            warmup=warmup,
            seed=seed + mode_index,
        )
        timings = {name: summarize(values) for name, values in raw_timings.items()}
        timings["compiled_over_additive_median"] = (
            timings["compiled"]["median_us"] / timings["additive"]["median_us"]
        )
        timings["compiled_over_direct_median"] = (
            timings["compiled"]["median_us"] / timings["direct"]["median_us"]
        )
        result["modes"][mode_name] = {"timings": timings, "relative_errors": errors}
    return result


def parse_int_list(value: str):
    return [int(item) for item in value.split(",") if item]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="1,8,32,128")
    parser.add_argument("--lengths", default="4,8,16,32")
    parser.add_argument("--dimension", type=int, default=4096)
    parser.add_argument("--dtypes", default="float16,bfloat16")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    cells = []
    for dtype_name in args.dtypes.split(","):
        for length in parse_int_list(args.lengths):
            for batch in parse_int_list(args.batches):
                print(
                    f"benchmark dtype={dtype_name} L={length} B={batch}",
                    flush=True,
                )
                cells.append(
                    benchmark_cell(
                        batch=batch,
                        length=length,
                        dimension=args.dimension,
                        dtype=dtype_map[dtype_name],
                        samples=args.samples,
                        warmup=args.warmup,
                        seed=args.seed + batch * 101 + length * 1009,
                    )
                )

    compiled_key_cells = [
        cell
        for cell in cells
        if cell["batch"] <= 8 and cell["length"] <= 16
    ]
    requested_batches = set(parse_int_list(args.batches))
    requested_lengths = set(parse_int_list(args.lengths))
    requested_dtypes = set(args.dtypes.split(","))
    full_frozen_grid = (
        requested_batches == {1, 8, 32, 128}
        and requested_lengths == {4, 8, 16, 32}
        and requested_dtypes == {"float16", "bfloat16"}
    )
    gates = {
        "full_frozen_grid_present": full_frozen_grid,
        "compiled_faster_than_direct_for_all_l_ge_8": all(
            cell["modes"]["compiled"]["timings"]["compiled_over_direct_median"]
            < 1.0
            for cell in cells
            if cell["length"] >= 8
        ),
        "compiled_within_1p15_additive_on_key_cells": all(
            cell["modes"]["compiled"]["timings"]["compiled_over_additive_median"]
            <= 1.15
            for cell in compiled_key_cells
        ),
        "float16_compiled_error_below_2e_3": any(
            cell["dtype"] == "float16" for cell in cells
        )
        and all(
            cell["modes"]["compiled"]["relative_errors"]["compiled"][
                "max_update_row_relative"
            ]
            <= 2e-3
            for cell in cells
            if cell["dtype"] == "float16"
        ),
        "bfloat16_compiled_error_below_2e_2": any(
            cell["dtype"] == "bfloat16" for cell in cells
        )
        and all(
            cell["modes"]["compiled"]["relative_errors"]["compiled"][
                "max_update_row_relative"
            ]
            <= 2e-2
            for cell in cells
            if cell["dtype"] == "bfloat16"
        ),
    }

    payload = {
        "candidate": "compiled-ordered-rank-one-program",
        "gate": "G1-unoptimized-H100-executor",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "device_properties": str(torch.cuda.get_device_properties(0)),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "args": vars(args) | {"output": str(args.output)},
        "cells": cells,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "scope": "executor only; selected program tensors already resident",
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": gates, "all_gates_pass": all(gates.values())}, indent=2))
    raise SystemExit(0 if payload["all_gates_pass"] else 1)


if __name__ == "__main__":
    main()
