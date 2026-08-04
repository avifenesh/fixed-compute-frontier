#!/usr/bin/env python3
"""Materially different fused H100 gate for ordered rank-one programs.

The earlier executor used a sequence of generated PyTorch kernels.  This gate
uses one CUDA block and one launch per token program for all three controls:

* additive: all rank-one coefficients see the original input;
* compiled: parallel base projections plus the exact scalar triangular program;
* direct: sequential rank-one updates held in on-chip shared memory.

Selected expert tensors are intentionally already resident.  Routing, gathers,
and the program table remain outside this executor-only gate.
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
from torch.utils.cpp_extension import load


HERE = Path(__file__).resolve().parent
CSRC = HERE / "csrc"
MODE = {"additive": 0, "compiled": 1, "direct": 2}


def load_extension():
    return load(
        name="fused_rank_one_ext_v1",
        sources=[
            str(CSRC / "fused_rank_one.cpp"),
            str(CSRC / "fused_rank_one_cuda.cu"),
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        extra_cflags=["-O3"],
        verbose=False,
    )


def pack_strict_lower(full: torch.Tensor) -> torch.Tensor:
    length = full.shape[1]
    if length <= 1:
        return full.new_empty((full.shape[0], 0))
    return torch.cat([full[:, i, :i] for i in range(1, length)], dim=1).contiguous()


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
    coupling = pack_strict_lower(torch.einsum("bld,bmd->blm", u, v))
    return tuple(t.contiguous() for t in (x, u, v, bias, scale, coupling))


def reference(inputs, mode: str) -> torch.Tensor:
    x, u, v, bias, scale, coupling = (tensor.float() for tensor in inputs)
    if mode == "direct":
        y = x.clone()
        for i in range(u.shape[1]):
            c = scale[:, i] * F.silu(torch.sum(y * u[:, i], dim=-1) + bias[:, i])
            y = y + c[:, None] * v[:, i]
        return y
    base = torch.sum(u * x[:, None, :], dim=-1) + bias
    if mode == "additive":
        coefficients = scale * F.silu(base)
    else:
        cs = []
        for i in range(u.shape[1]):
            value = base[:, i]
            if i:
                start = i * (i - 1) // 2
                value = value + torch.sum(
                    coupling[:, start : start + i] * torch.stack(cs, dim=1), dim=1
                )
            cs.append(scale[:, i] * F.silu(value))
        coefficients = torch.stack(cs, dim=1)
    return x + torch.sum(coefficients[:, :, None] * v, dim=1)


def summarize(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "samples": len(values),
        "median_us": statistics.median(values),
        "mean_us": statistics.fmean(values),
        "p10_us": ordered[int(0.10 * (len(ordered) - 1))],
        "p90_us": ordered[int(0.90 * (len(ordered) - 1))],
    }


def benchmark_cell(ext, *, batch, length, dimension, dtype, samples, warmup, seed):
    inputs = make_inputs(batch, length, dimension, dtype, seed)
    outputs = {name: ext.forward(*inputs, mode) for name, mode in MODE.items()}
    torch.cuda.synchronize()
    errors = {}
    for name, output in outputs.items():
        expected = reference(inputs, name)
        difference = output.float() - expected
        errors[name] = {
            "global_relative": float(
                torch.linalg.vector_norm(difference)
                / torch.clamp(torch.linalg.vector_norm(expected), min=1e-12)
            ),
            "max_row_relative": float(
                torch.max(
                    torch.linalg.vector_norm(difference, dim=-1)
                    / torch.clamp(torch.linalg.vector_norm(expected, dim=-1), min=1e-12)
                )
            ),
            "max_absolute": float(torch.max(torch.abs(difference))),
        }

    for mode in MODE.values():
        for _ in range(warmup):
            ext.forward(*inputs, mode)
    torch.cuda.synchronize()
    schedule = [name for name in MODE for _ in range(samples)]
    random.Random(seed).shuffle(schedule)
    timings = {name: [] for name in MODE}
    for name in schedule:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        ext.forward(*inputs, MODE[name])
        end.record()
        end.synchronize()
        timings[name].append(float(start.elapsed_time(end) * 1000.0))
    summaries = {name: summarize(values) for name, values in timings.items()}
    summaries["compiled_over_additive_median"] = (
        summaries["compiled"]["median_us"] / summaries["additive"]["median_us"]
    )
    summaries["compiled_over_direct_median"] = (
        summaries["compiled"]["median_us"] / summaries["direct"]["median_us"]
    )
    return {
        "batch": batch,
        "length": length,
        "dimension": dimension,
        "dtype": str(dtype).removeprefix("torch."),
        "errors": errors,
        "timings": summaries,
    }


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="1,8,32,128")
    parser.add_argument("--lengths", default="4,8,16,32")
    parser.add_argument("--dimension", type=int, default=4096)
    parser.add_argument("--dtypes", default="float16,bfloat16")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ext = load_extension()
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    cells = []
    for dtype_name in args.dtypes.split(","):
        for length in parse_ints(args.lengths):
            for batch in parse_ints(args.batches):
                print(f"benchmark dtype={dtype_name} L={length} B={batch}", flush=True)
                cells.append(
                    benchmark_cell(
                        ext,
                        batch=batch,
                        length=length,
                        dimension=args.dimension,
                        dtype=dtype_map[dtype_name],
                        samples=args.samples,
                        warmup=args.warmup,
                        seed=args.seed + 101 * batch + 1009 * length,
                    )
                )
    requested_batches = set(parse_ints(args.batches))
    requested_lengths = set(parse_ints(args.lengths))
    full_grid = requested_batches == {1, 8, 32, 128} and requested_lengths == {
        4,
        8,
        16,
        32,
    }
    key_cells = [cell for cell in cells if cell["batch"] <= 8 and cell["length"] <= 16]
    gates = {
        "full_frozen_grid_present": full_grid,
        "compiled_within_1p15_additive_on_key_cells": all(
            cell["timings"]["compiled_over_additive_median"] <= 1.15
            for cell in key_cells
        ),
        "compiled_faster_than_fused_direct_for_l_ge_8": all(
            cell["timings"]["compiled_over_direct_median"] < 1.0
            for cell in cells
            if cell["length"] >= 8
        ),
        "max_row_relative_below_0p005": all(
            cell["errors"]["compiled"]["max_row_relative"] <= 0.005
            for cell in cells
        ),
    }
    source_paths = [Path(__file__), CSRC / "fused_rank_one.cpp", CSRC / "fused_rank_one_cuda.cu"]
    source_digest = hashlib.sha256(
        b"".join(path.read_bytes() for path in source_paths)
    ).hexdigest()
    payload = {
        "candidate": "compiled-ordered-rank-one-program",
        "gate": "G1-single-launch-fused-H100-executor",
        "source_sha256": source_digest,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
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
