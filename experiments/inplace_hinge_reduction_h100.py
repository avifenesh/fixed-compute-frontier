#!/usr/bin/env python3
"""H100 gate for one baseline-containing in-place accumulator hinge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl

from mid_reduction_sign_h100 import (
    _dense_int8_kernel,
    _grid,
    _metadata,
    _time,
    configurations,
)


OUTPUT = Path("results/inplace-hinge-reduction-h100-development.json")


@triton.jit
def _inplace_hinge_int8_kernel(
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
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

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
        accumulator += tl.dot(values, weights, out_dtype=tl.int32)

    coefficient = tl.load(alpha)
    accumulator += coefficient * tl.abs(accumulator)

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
        accumulator += tl.dot(values, weights, out_dtype=tl.int32)

    output_offsets = offsets_m[:, None] * columns + offsets_n[None, :]
    mask = (offsets_m[:, None] < rows) & (offsets_n[None, :] < columns)
    tl.store(output + output_offsets, accumulator, mask=mask)


def _launch_dense(
    activation: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    configuration: dict[str, int],
) -> Any:
    rows, reduction = activation.shape
    columns = weight.shape[1]
    return _dense_int8_kernel[_grid(rows, columns, configuration)](
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


def _launch_hinge(
    activation: torch.Tensor,
    weight: torch.Tensor,
    alpha: torch.Tensor,
    output: torch.Tensor,
    configuration: dict[str, int],
) -> Any:
    rows, reduction = activation.shape
    columns = weight.shape[1]
    return _inplace_hinge_int8_kernel[_grid(rows, columns, configuration)](
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


def benchmark_configuration(
    activation: torch.Tensor,
    weight: torch.Tensor,
    alpha: torch.Tensor,
    dense_output: torch.Tensor,
    hinge_output: torch.Tensor,
    configuration: dict[str, int],
) -> dict[str, Any]:
    record: dict[str, Any] = {"configuration": configuration}
    try:
        dense_compiled = _launch_dense(activation, weight, dense_output, configuration)
        hinge_compiled = _launch_hinge(
            activation, weight, alpha, hinge_output, configuration
        )
        torch.cuda.synchronize()
        maximum_error = int((dense_output - hinge_output).abs().max().item())
        dense_ms = _time(
            lambda: _launch_dense(activation, weight, dense_output, configuration)
        )
        hinge_ms = _time(
            lambda: _launch_hinge(
                activation, weight, alpha, hinge_output, configuration
            )
        )
        record.update(
            {
                "valid": True,
                "maximum_absolute_error": maximum_error,
                "dense_ms": dense_ms,
                "hinge_ms": hinge_ms,
                "dense_compiler": _metadata(dense_compiled),
                "hinge_compiler": _metadata(hinge_compiled),
            }
        )
    except Exception as error:  # pragma: no cover
        record.update({"valid": False, "error": repr(error)})
    return record


def benchmark_cell(rows: int, columns: int, reduction: int, seed: int) -> dict[str, Any]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    activation = torch.randint(
        -8, 9, (rows, reduction), dtype=torch.int8, device="cuda", generator=generator
    ).contiguous()
    weight = torch.randint(
        -8, 9, (reduction, columns), dtype=torch.int8, device="cuda", generator=generator
    ).contiguous()
    alpha = torch.zeros((), dtype=torch.int32, device="cuda")
    dense_output = torch.empty((rows, columns), dtype=torch.int32, device="cuda")
    hinge_output = torch.empty_like(dense_output)
    records = [
        benchmark_configuration(
            activation, weight, alpha, dense_output, hinge_output, configuration
        )
        for configuration in configurations()
    ]
    exact = [
        record
        for record in records
        if record["valid"] and record["maximum_absolute_error"] == 0
    ]
    if not exact:
        return {"rows": rows, "records": records, "valid": False}
    best_dense = min(exact, key=lambda record: record["dense_ms"][0])
    best_hinge = min(exact, key=lambda record: record["hinge_ms"][0])
    return {
        "rows": rows,
        "valid": True,
        "records": records,
        "best_dense": {
            "configuration": best_dense["configuration"],
            "median_ms": best_dense["dense_ms"][0],
            "compiler": best_dense["dense_compiler"],
        },
        "best_hinge": {
            "configuration": best_hinge["configuration"],
            "median_ms": best_hinge["hinge_ms"][0],
            "compiler": best_hinge["hinge_compiler"],
            "same_configuration_dense_compiler": best_hinge["dense_compiler"],
        },
        "best_hinge_over_best_dense": best_hinge["hinge_ms"][0]
        / best_dense["dense_ms"][0],
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    cells = [
        benchmark_cell(rows, arguments.columns, arguments.reduction, arguments.seed + rows)
        for rows in arguments.rows
    ]
    all_valid = all(cell["valid"] for cell in cells)
    key_cells = [cell for cell in cells if cell["rows"] in (64, 256)]
    register_deltas = [
        cell["best_hinge"]["compiler"]["registers"]
        - cell["best_hinge"]["same_configuration_dense_compiler"]["registers"]
        for cell in cells
        if cell["valid"]
    ]
    decision = {
        "all_valid": all_valid,
        "bit_exact_all": all_valid
        and all(
            all(
                not record["valid"] or record["maximum_absolute_error"] == 0
                for record in cell["records"]
            )
            for cell in cells
        ),
        "compiler_metadata_visible": all(
            cell[arm]["compiler"]["ptx_length"] > 0
            for cell in cells
            if cell["valid"]
            for arm in ("best_dense", "best_hinge")
        ),
        "tensor_core_all_best": all(
            cell[arm]["compiler"]["has_wgmma"]
            or cell[arm]["compiler"]["has_mma_sync"]
            for cell in cells
            if cell["valid"]
            for arm in ("best_dense", "best_hinge")
        ),
        "key_cells_at_most_1_01": all_valid
        and all(cell["best_hinge_over_best_dense"] <= 1.01 for cell in key_cells),
        "all_cells_at_most_1_02": all_valid
        and all(cell["best_hinge_over_best_dense"] <= 1.02 for cell in cells),
        "selected_hinge_no_spills": all(
            cell["best_hinge"]["compiler"]["spills"] == 0
            for cell in cells
            if cell["valid"]
        ),
        "selected_register_delta_at_most_4": all(delta <= 4 for delta in register_deltas),
    }
    decision["advance_to_algebra_screen"] = all(decision.values())
    payload = {
        "schema": "inplace-hinge-reduction-h100-development-v1",
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
        "selected_register_deltas": register_deltas,
        "decision": decision,
    }
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
