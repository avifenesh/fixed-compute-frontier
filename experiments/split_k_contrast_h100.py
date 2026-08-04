#!/usr/bin/env python3
"""H100 register/occupancy gate for contrast-preserving W8 split-K."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import torch
import triton
import triton.language as tl


OUTPUT = Path("results/split-k-contrast-h100-development.json")


@triton.jit
def _dense_int8_kernel(
    activation,
    weight,
    output,
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
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for start_k in range(0, reduction, BLOCK_K):
        offsets_k = start_k + tl.arange(0, BLOCK_K)
        values = tl.load(
            activation + offsets_m[:, None] * reduction + offsets_k[None, :],
            mask=offsets_m[:, None] < rows,
            other=0,
        )
        weights = tl.load(
            weight + offsets_k[:, None] * columns + offsets_n[None, :],
            mask=offsets_n[None, :] < columns,
            other=0,
        )
        accumulator += tl.dot(values, weights, out_dtype=tl.int32)

    output_offsets = offsets_m[:, None] * columns + offsets_n[None, :]
    mask = (offsets_m[:, None] < rows) & (offsets_n[None, :] < columns)
    tl.store(output + output_offsets, accumulator, mask=mask)


@triton.jit
def _split_int8_kernel(
    activation,
    weight,
    alpha,
    output,
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
    first = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    second = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for start_k in range(0, reduction // 2, BLOCK_K):
        offsets_k = start_k + tl.arange(0, BLOCK_K)
        values = tl.load(
            activation + offsets_m[:, None] * reduction + offsets_k[None, :],
            mask=offsets_m[:, None] < rows,
            other=0,
        )
        weights = tl.load(
            weight + offsets_k[:, None] * columns + offsets_n[None, :],
            mask=offsets_n[None, :] < columns,
            other=0,
        )
        first += tl.dot(values, weights, out_dtype=tl.int32)
    for start_k in range(reduction // 2, reduction, BLOCK_K):
        offsets_k = start_k + tl.arange(0, BLOCK_K)
        values = tl.load(
            activation + offsets_m[:, None] * reduction + offsets_k[None, :],
            mask=offsets_m[:, None] < rows,
            other=0,
        )
        weights = tl.load(
            weight + offsets_k[:, None] * columns + offsets_n[None, :],
            mask=offsets_n[None, :] < columns,
            other=0,
        )
        second += tl.dot(values, weights, out_dtype=tl.int32)

    coefficient = tl.load(alpha)
    summed = first + second
    contrast = first - second
    result = summed + coefficient * contrast
    output_offsets = offsets_m[:, None] * columns + offsets_n[None, :]
    mask = (offsets_m[:, None] < rows) & (offsets_n[None, :] < columns)
    tl.store(output + output_offsets, result, mask=mask)


def _grid(rows: int, columns: int, block_m: int, block_n: int) -> tuple[int, int]:
    return triton.cdiv(rows, block_m), triton.cdiv(columns, block_n)


def _metadata(compiled: Any) -> dict[str, Any]:
    metadata = getattr(compiled, "metadata", None)
    assembly = getattr(compiled, "asm", {}) or {}
    ptx = assembly.get("ptx", "")
    return {
        "name": getattr(compiled, "name", None),
        "registers": getattr(compiled, "n_regs", getattr(metadata, "num_regs", None)),
        "spills": getattr(compiled, "n_spills", getattr(metadata, "num_spills", None)),
        "shared_bytes": getattr(compiled, "shared", getattr(metadata, "shared", None)),
        "has_wgmma": "wgmma.mma_async" in ptx,
        "has_mma_sync": "mma.sync" in ptx,
        "ptx_length": len(ptx),
    }


def _launch_dense(
    activation: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    configuration: dict[str, int],
) -> Any:
    rows, reduction = activation.shape
    columns = weight.shape[1]
    grid = _grid(rows, columns, configuration["block_m"], configuration["block_n"])
    return _dense_int8_kernel[grid](
        activation,
        weight,
        output,
        rows,
        columns,
        reduction,
        BLOCK_M=configuration["block_m"],
        BLOCK_N=configuration["block_n"],
        BLOCK_K=configuration["block_k"],
        num_warps=configuration["num_warps"],
        num_stages=configuration["num_stages"],
    )


def _launch_split(
    activation: torch.Tensor,
    weight: torch.Tensor,
    alpha: torch.Tensor,
    output: torch.Tensor,
    configuration: dict[str, int],
) -> Any:
    rows, reduction = activation.shape
    columns = weight.shape[1]
    grid = _grid(rows, columns, configuration["block_m"], configuration["block_n"])
    return _split_int8_kernel[grid](
        activation,
        weight,
        alpha,
        output,
        rows,
        columns,
        reduction,
        BLOCK_M=configuration["block_m"],
        BLOCK_N=configuration["block_n"],
        BLOCK_K=configuration["block_k"],
        num_warps=configuration["num_warps"],
        num_stages=configuration["num_stages"],
    )


def _time(function: Callable[[], Any]) -> list[float]:
    measurement = triton.testing.do_bench(
        function,
        warmup=100,
        rep=300,
        quantiles=[0.5, 0.2, 0.8],
    )
    return [float(value) for value in measurement]


def benchmark_configuration(
    activation: torch.Tensor,
    weight: torch.Tensor,
    alpha: torch.Tensor,
    dense_output: torch.Tensor,
    split_output: torch.Tensor,
    configuration: dict[str, int],
) -> dict[str, Any]:
    record: dict[str, Any] = {"configuration": configuration}
    try:
        dense_compiled = _launch_dense(activation, weight, dense_output, configuration)
        split_compiled = _launch_split(
            activation, weight, alpha, split_output, configuration
        )
        torch.cuda.synchronize()
        maximum_error = int((dense_output - split_output).abs().max().item())
        dense_ms = _time(
            lambda: _launch_dense(activation, weight, dense_output, configuration)
        )
        split_ms = _time(
            lambda: _launch_split(
                activation, weight, alpha, split_output, configuration
            )
        )
        record.update(
            {
                "valid": True,
                "maximum_absolute_error": maximum_error,
                "dense_ms": dense_ms,
                "split_ms": split_ms,
                "split_over_dense": split_ms[0] / dense_ms[0],
                "dense_compiler": _metadata(dense_compiled),
                "split_compiler": _metadata(split_compiled),
            }
        )
    except Exception as error:  # pragma: no cover - records compiler gate failures
        record.update({"valid": False, "error": repr(error)})
    return record


def configurations() -> list[dict[str, int]]:
    return [
        {
            "block_m": 64,
            "block_n": block_n,
            "block_k": 32,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
        for block_n in (64, 128)
        for num_warps in (4, 8)
        for num_stages in (3, 4)
    ]


def benchmark_cell(
    rows: int,
    columns: int,
    reduction: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    activation = torch.randint(
        -8,
        9,
        (rows, reduction),
        dtype=torch.int8,
        device="cuda",
        generator=generator,
    ).contiguous()
    weight = torch.randint(
        -8,
        9,
        (reduction, columns),
        dtype=torch.int8,
        device="cuda",
        generator=generator,
    ).contiguous()
    alpha = torch.zeros((), dtype=torch.int32, device="cuda")
    dense_output = torch.empty((rows, columns), dtype=torch.int32, device="cuda")
    split_output = torch.empty_like(dense_output)

    records = [
        benchmark_configuration(
            activation,
            weight,
            alpha,
            dense_output,
            split_output,
            configuration,
        )
        for configuration in configurations()
    ]
    valid = [record for record in records if record["valid"]]
    exact = [record for record in valid if record["maximum_absolute_error"] == 0]
    if not exact:
        return {"rows": rows, "records": records, "valid": False}
    best_dense = min(exact, key=lambda record: record["dense_ms"][0])
    best_split = min(exact, key=lambda record: record["split_ms"][0])
    return {
        "rows": rows,
        "valid": True,
        "records": records,
        "best_dense": {
            "configuration": best_dense["configuration"],
            "median_ms": best_dense["dense_ms"][0],
            "compiler": best_dense["dense_compiler"],
        },
        "best_split": {
            "configuration": best_split["configuration"],
            "median_ms": best_split["split_ms"][0],
            "compiler": best_split["split_compiler"],
        },
        "best_split_over_best_dense": best_split["split_ms"][0]
        / best_dense["dense_ms"][0],
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    cells = [
        benchmark_cell(
            rows,
            arguments.columns,
            arguments.reduction,
            arguments.seed + rows,
        )
        for rows in arguments.rows
    ]
    key_cells = [cell for cell in cells if cell["rows"] in (64, 256)]
    all_valid = all(cell["valid"] for cell in cells)
    compiler_visible = all(
        cell["best_dense"]["compiler"]["ptx_length"] > 0
        and cell["best_split"]["compiler"]["ptx_length"] > 0
        for cell in cells
        if cell["valid"]
    )
    tensor_core_all = all(
        (
            cell["best_dense"]["compiler"]["has_wgmma"]
            or cell["best_dense"]["compiler"]["has_mma_sync"]
        )
        and (
            cell["best_split"]["compiler"]["has_wgmma"]
            or cell["best_split"]["compiler"]["has_mma_sync"]
        )
        for cell in cells
        if cell["valid"]
    )
    payload = {
        "schema": "split-k-contrast-h100-development-v1",
        "device": torch.cuda.get_device_name(0),
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": triton.__version__,
        },
        "shape": {
            "rows": arguments.rows,
            "columns": arguments.columns,
            "reduction": arguments.reduction,
        },
        "runtime_alpha": 0,
        "cells": cells,
        "decision": {
            "all_valid": all_valid,
            "bit_exact_all": all_valid
            and all(
                all(
                    not record["valid"] or record["maximum_absolute_error"] == 0
                    for record in cell["records"]
                )
                for cell in cells
            ),
            "compiler_metadata_visible": compiler_visible,
            "tensor_core_all_best": tensor_core_all,
            "key_cells_at_most_1_02": all_valid
            and all(cell["best_split_over_best_dense"] <= 1.02 for cell in key_cells),
            "all_cells_at_most_1_05": all_valid
            and all(cell["best_split_over_best_dense"] <= 1.05 for cell in cells),
        },
    }
    payload["decision"]["advance_to_fused_ffn"] = all(
        payload["decision"].values()
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 64, 256, 1024])
    parser.add_argument("--columns", type=int, default=4096)
    parser.add_argument("--reduction", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260728)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

