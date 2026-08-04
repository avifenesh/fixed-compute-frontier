#!/usr/bin/env python3
"""H100 kill test for balanced-radix projection multiplexing.

The candidate decodes every K=32 Tensor Core partial before accumulating over
the full reduction.  The baseline is a fused dual binary-weight INT8 GEMM that
shares activation loads but issues two dots.  This is a development kernel, not
an optimized serving implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl

from experiments.balanced_radix_projection import pack_two_binary_weights


OUTPUT = Path("results/balanced-radix-projection-h100-development.json")


@triton.jit
def _packed_radix64_kernel(
    activation,
    packed_weight,
    output_low,
    output_high,
    rows: tl.constexpr,
    columns: tl.constexpr,
    reduction: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator_low = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    accumulator_high = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for start_k in range(0, reduction, BLOCK_K):
        offsets_k = start_k + tl.arange(0, BLOCK_K)
        values = tl.load(
            activation + offsets_m[:, None] * reduction + offsets_k[None, :],
            mask=offsets_m[:, None] < rows,
            other=0,
        )
        weights = tl.load(
            packed_weight + offsets_k[:, None] * columns + offsets_n[None, :],
            mask=offsets_n[None, :] < columns,
            other=0,
        )
        packed_partial = tl.dot(values, weights, out_dtype=tl.int32)
        magnitude = tl.abs(packed_partial)
        quotient = magnitude >> 6
        remainder = magnitude & 63
        round_up = (remainder > 32) | ((remainder == 32) & ((quotient & 1) == 1))
        rounded_magnitude = quotient + round_up.to(tl.int32)
        high = tl.where(packed_partial < 0, -rounded_magnitude, rounded_magnitude)
        low = packed_partial - (high << 6)
        accumulator_low += low
        accumulator_high += high
    output_offsets = offsets_m[:, None] * columns + offsets_n[None, :]
    mask = (offsets_m[:, None] < rows) & (offsets_n[None, :] < columns)
    tl.store(output_low + output_offsets, accumulator_low, mask=mask)
    tl.store(output_high + output_offsets, accumulator_high, mask=mask)


@triton.jit
def _fused_dual_int8_kernel(
    activation,
    weight_low,
    weight_high,
    output_low,
    output_high,
    rows: tl.constexpr,
    columns: tl.constexpr,
    reduction: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator_low = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    accumulator_high = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for start_k in range(0, reduction, BLOCK_K):
        offsets_k = start_k + tl.arange(0, BLOCK_K)
        values = tl.load(
            activation + offsets_m[:, None] * reduction + offsets_k[None, :],
            mask=offsets_m[:, None] < rows,
            other=0,
        )
        weights_low = tl.load(
            weight_low + offsets_k[:, None] * columns + offsets_n[None, :],
            mask=offsets_n[None, :] < columns,
            other=0,
        )
        weights_high = tl.load(
            weight_high + offsets_k[:, None] * columns + offsets_n[None, :],
            mask=offsets_n[None, :] < columns,
            other=0,
        )
        accumulator_low += tl.dot(values, weights_low, out_dtype=tl.int32)
        accumulator_high += tl.dot(values, weights_high, out_dtype=tl.int32)
    output_offsets = offsets_m[:, None] * columns + offsets_n[None, :]
    mask = (offsets_m[:, None] < rows) & (offsets_n[None, :] < columns)
    tl.store(output_low + output_offsets, accumulator_low, mask=mask)
    tl.store(output_high + output_offsets, accumulator_high, mask=mask)


def launch_packed(
    activation: torch.Tensor,
    packed_weight: torch.Tensor,
    output_low: torch.Tensor,
    output_high: torch.Tensor,
) -> None:
    rows, reduction = activation.shape
    weight_reduction, columns = packed_weight.shape
    if reduction != weight_reduction:
        raise ValueError("K mismatch")
    grid = (triton.cdiv(rows, 16), triton.cdiv(columns, 32))
    _packed_radix64_kernel[grid](
        activation,
        packed_weight,
        output_low,
        output_high,
        rows,
        columns,
        reduction,
        BLOCK_M=16,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
    )


def launch_dual(
    activation: torch.Tensor,
    weight_low: torch.Tensor,
    weight_high: torch.Tensor,
    output_low: torch.Tensor,
    output_high: torch.Tensor,
) -> None:
    rows, reduction = activation.shape
    weight_reduction, columns = weight_low.shape
    if reduction != weight_reduction or weight_high.shape != weight_low.shape:
        raise ValueError("shape mismatch")
    grid = (triton.cdiv(rows, 16), triton.cdiv(columns, 32))
    _fused_dual_int8_kernel[grid](
        activation,
        weight_low,
        weight_high,
        output_low,
        output_high,
        rows,
        columns,
        reduction,
        BLOCK_M=16,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
    )


def benchmark_cell(rows: int, columns: int, reduction: int, seed: int) -> dict[str, Any]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    activation = torch.randint(
        -1, 2, (rows, reduction), dtype=torch.int8, device="cuda", generator=generator
    ).contiguous()
    weight_low_nk = (
        2
        * torch.randint(
            0, 2, (columns, reduction), dtype=torch.int8, device="cuda", generator=generator
        )
        - 1
    ).contiguous()
    weight_high_nk = (
        2
        * torch.randint(
            0, 2, (columns, reduction), dtype=torch.int8, device="cuda", generator=generator
        )
        - 1
    ).contiguous()
    packed_nk = pack_two_binary_weights(weight_low_nk, weight_high_nk)
    weight_low = weight_low_nk.T.contiguous()
    weight_high = weight_high_nk.T.contiguous()
    packed_weight = packed_nk.T.contiguous()
    packed_low = torch.empty((rows, columns), dtype=torch.int32, device="cuda")
    packed_high = torch.empty_like(packed_low)
    dual_low = torch.empty_like(packed_low)
    dual_high = torch.empty_like(packed_low)
    launch_packed(activation, packed_weight, packed_low, packed_high)
    launch_dual(activation, weight_low, weight_high, dual_low, dual_high)
    torch.cuda.synchronize()
    error_low = int((packed_low - dual_low).abs().max())
    error_high = int((packed_high - dual_high).abs().max())
    if error_low or error_high:
        raise AssertionError((error_low, error_high))
    packed_ms = triton.testing.do_bench(
        lambda: launch_packed(activation, packed_weight, packed_low, packed_high),
        warmup=100,
        rep=300,
        quantiles=[0.5, 0.2, 0.8],
    )
    dual_ms = triton.testing.do_bench(
        lambda: launch_dual(
            activation, weight_low, weight_high, dual_low, dual_high
        ),
        warmup=100,
        rep=300,
        quantiles=[0.5, 0.2, 0.8],
    )
    return {
        "rows": rows,
        "columns": columns,
        "reduction": reduction,
        "packed_ms": packed_ms,
        "fused_dual_ms": dual_ms,
        "packed_speedup_over_fused_dual": dual_ms[0] / packed_ms[0],
        "max_abs_error_low": error_low,
        "max_abs_error_high": error_high,
        "packed_resident_weight_bytes": packed_weight.numel(),
        "dual_resident_weight_bytes": weight_low.numel() + weight_high.numel(),
        "packed_int8_scalar_products": rows * columns * reduction,
        "dual_int8_scalar_products": 2 * rows * columns * reduction,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    if args.reduction % 32:
        raise ValueError("reduction must be divisible by 32")
    cells = [
        benchmark_cell(rows, args.columns, args.reduction, args.seed + rows)
        for rows in args.rows
    ]
    payload = {
        "schema": "balanced-radix-projection-h100-development-v1",
        "device": torch.cuda.get_device_name(0),
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": triton.__version__,
        },
        "candidate": "radix-64 two binary projections from one INT8 MMA stream",
        "mandatory_constraint": "decode every K=32 partial before accumulation",
        "cells": cells,
        "decision": {
            "exact_all_cells": all(
                cell["max_abs_error_low"] == 0 and cell["max_abs_error_high"] == 0
                for cell in cells
            ),
            "beats_fused_dual_all_cells": all(
                cell["packed_speedup_over_fused_dual"] > 1.0 for cell in cells
            ),
            "advance_to_learning": all(
                cell["packed_speedup_over_fused_dual"] >= 1.25 for cell in cells
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--rows", type=int, nargs="+", default=[16, 128, 512])
    parser.add_argument("--columns", type=int, default=1536)
    parser.add_argument("--reduction", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2207)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
