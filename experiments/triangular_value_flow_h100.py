#!/usr/bin/env python3
"""H100 systems screen for block-16 triangular value flow."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics

import numpy as np
import torch
import triton
import triton.language as tl


HIDDEN = 4096
QUERY_HEADS = 32
KV_HEADS = 8
GROUPS = QUERY_HEADS // KV_HEADS
HEAD_DIM = 128
BLOCK = 16
BLOCKS = HEAD_DIM // BLOCK
TAU = 0.125
OUTPUT = Path("results/triangular-value-flow-h100-development.json")


@triton.jit
def value_flow_kernel(
    input_ptr,
    output_ptr,
    output_weight_ptr,
    WIDTH: tl.constexpr,
    HEAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    TAU_VALUE: tl.constexpr,
    TVF: tl.constexpr,
):
    token = tl.program_id(0)
    query_head = tl.program_id(1)
    block_index = tl.program_id(2)
    coordinate = tl.arange(0, BLOCK_SIZE)
    block_start = block_index * BLOCK_SIZE
    column_start = query_head * HEAD + block_start
    values = tl.load(
        input_ptr + token * WIDTH + column_start + coordinate
    ).to(tl.float32)
    result = values
    if TVF:
        source = tl.arange(0, BLOCK_SIZE)
        target = tl.arange(0, BLOCK_SIZE)[:, None]
        kv_head = query_head // GROUP_SIZE
        representative_head = kv_head * GROUP_SIZE
        coefficient_row = block_start + target
        coefficient_column = representative_head * HEAD + block_start + source[None, :]
        coefficient = tl.load(
            output_weight_ptr + coefficient_row * WIDTH + coefficient_column,
            mask=source[None, :] < target,
            other=0.0,
        ).to(tl.float32) / TAU_VALUE
        feature = values * tl.abs(values)
        result += tl.sum(coefficient * feature[None, :], axis=1)
    tl.store(output_ptr + token * WIDTH + column_start + coordinate, result)


def flow_into(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    output_weight: torch.Tensor,
    *,
    candidate: bool,
) -> None:
    value_flow_kernel[(inputs.shape[0], QUERY_HEADS, BLOCKS)](
        inputs,
        outputs,
        output_weight,
        WIDTH=HIDDEN,
        HEAD=HEAD_DIM,
        BLOCK_SIZE=BLOCK,
        GROUP_SIZE=GROUPS,
        TAU_VALUE=TAU,
        TVF=candidate,
        num_warps=1,
    )


def make_output_weight(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    weight = torch.randn(
        HIDDEN,
        HIDDEN,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ) / math.sqrt(HIDDEN)
    for kv_head in range(KV_HEADS):
        representative_head = kv_head * GROUPS
        for block_start in range(0, HEAD_DIM, BLOCK):
            coefficient = 0.03 * torch.randn(
                BLOCK,
                BLOCK,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            coefficient = torch.tril(coefficient, diagonal=-1) * TAU
            rows = slice(block_start, block_start + BLOCK)
            columns = slice(
                representative_head * HEAD_DIM + block_start,
                representative_head * HEAD_DIM + block_start + BLOCK,
            )
            existing = weight[rows, columns]
            weight[rows, columns] = torch.triu(existing) + coefficient
    return weight.contiguous()


def reference_flow(
    inputs: torch.Tensor,
    output_weight: torch.Tensor,
) -> torch.Tensor:
    grouped = inputs.view(-1, QUERY_HEADS, HEAD_DIM)
    pieces = []
    for block_start in range(0, HEAD_DIM, BLOCK):
        values = grouped[..., block_start:block_start + BLOCK]
        coefficients = []
        for kv_head in range(KV_HEADS):
            representative_head = kv_head * GROUPS
            physical = output_weight[
                block_start:block_start + BLOCK,
                representative_head * HEAD_DIM + block_start:
                representative_head * HEAD_DIM + block_start + BLOCK,
            ]
            coefficients.append(torch.tril(physical, diagonal=-1))
        coefficient = torch.stack(coefficients).to(values.dtype).float() / TAU
        values_by_group = values.unflatten(1, (KV_HEADS, GROUPS))
        feature = values_by_group.float() * values_by_group.float().abs()
        delta = torch.matmul(
            coefficient[None, :, None], feature.unsqueeze(-1)
        ).squeeze(-1)
        pieces.append((values_by_group.float() + delta).to(inputs.dtype))
    return torch.cat(pieces, dim=-1).flatten(1)


def bootstrap_ratio(left: list[float], right: list[float], seed: int) -> dict[str, float]:
    candidate = np.asarray(left)
    control = np.asarray(right)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(left), size=(5000, len(left)))
    ratios = np.median(candidate[indices], axis=1) / np.median(control[indices], axis=1)
    return {
        "median_ratio": float(np.median(candidate) / np.median(control)),
        "lower_95": float(np.quantile(ratios, 0.025)),
        "upper_95": float(np.quantile(ratios, 0.975)),
    }


def benchmark(rows: int, trials: int, warmup: int, seed: int) -> dict[str, object]:
    generator = torch.Generator(device="cuda").manual_seed(seed + rows)
    inputs = torch.randn(
        rows, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    weight = make_output_weight(seed)
    flowed = {
        arm: torch.empty_like(inputs) for arm in ("control", "candidate")
    }
    projected = {
        arm: torch.empty(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
        for arm in ("control", "candidate")
    }
    l2_flush = torch.zeros(64 * 1024 * 1024, device="cuda", dtype=torch.bfloat16)

    def run_proxy(arm: str) -> None:
        flow_into(inputs, flowed[arm], weight, candidate=(arm == "candidate"))
        torch.mm(flowed[arm], weight.T, out=projected[arm])

    def run_dropin(arm: str) -> None:
        if arm == "candidate":
            flow_into(inputs, flowed[arm], weight, candidate=True)
            torch.mm(flowed[arm], weight.T, out=projected[arm])
        else:
            torch.mm(inputs, weight.T, out=projected[arm])

    for _ in range(warmup):
        run_proxy("control")
        run_proxy("candidate")
    torch.cuda.synchronize()
    flow_into(inputs, flowed["control"], weight, candidate=False)
    flow_into(inputs, flowed["candidate"], weight, candidate=True)
    torch.cuda.synchronize()
    reference = reference_flow(inputs, weight)
    correctness = {
        "control_bit_exact": bool(torch.equal(flowed["control"], inputs)),
        "candidate_bit_exact": bool(torch.equal(flowed["candidate"], reference)),
        "candidate_max_abs": float((flowed["candidate"].float() - reference.float()).abs().max()),
    }

    def collect(run) -> dict[str, object]:
        timings = {arm: [] for arm in ("control", "candidate")}
        order_generator = random.Random(seed + rows)
        for _ in range(trials):
            order = ["control", "candidate"]
            order_generator.shuffle(order)
            for arm in order:
                l2_flush.zero_()
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                run(arm)
                end.record()
                end.synchronize()
                timings[arm].append(start.elapsed_time(end))
        return {
            "control_median_us": statistics.median(timings["control"]) * 1000.0,
            "candidate_median_us": statistics.median(timings["candidate"]) * 1000.0,
            "candidate_over_control": bootstrap_ratio(
                timings["candidate"], timings["control"], seed + 991
            ),
        }

    return {
        "rows": rows,
        "trials": trials,
        "correctness": correctness,
        "fused_epilogue_proxy": collect(run_proxy),
        "unfused_dropin": collect(run_dropin),
        "ledger": {
            "learned_weight_bytes_each": weight.numel() * weight.element_size(),
            "kv_bytes_added": 0,
            "metadata_bits": 0,
            "coefficient_slots_per_kv_group": BLOCKS * BLOCK * (BLOCK - 1) // 2,
            "triangular_mac_per_token_layer": QUERY_HEADS * BLOCKS * BLOCK * (BLOCK - 1) // 2,
            "signed_square_per_token_layer": QUERY_HEADS * HEAD_DIM,
            "dense_masked_multiply_slots_per_token_layer": QUERY_HEADS * BLOCKS * BLOCK * BLOCK,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument(
        "--output", type=Path, default=OUTPUT
    )
    args = parser.parse_args()
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    cells = [
        benchmark(rows, args.trials, args.warmup, args.seed + index * 1009)
        for index, rows in enumerate((1, 8, 32, 128, 512, 2048))
    ]
    payload = {
        "schema": "triangular-value-flow-h100-development-v1",
        "device": torch.cuda.get_device_name(0),
        "runtime": {
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda": torch.version.cuda,
        },
        "cells": cells,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        str(cell["rows"]): {
            "correctness": cell["correctness"],
            "proxy_ratio": cell["fused_epilogue_proxy"]["candidate_over_control"],
            "dropin_ratio": cell["unfused_dropin"]["candidate_over_control"],
        }
        for cell in cells
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
