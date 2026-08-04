#!/usr/bin/env python3
"""H100 serving-cost screen for pre-cache triangular value encoding."""

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
HALF = HEAD_DIM // 2
Q_DIM = QUERY_HEADS * HEAD_DIM
K_DIM = KV_HEADS * HEAD_DIM
V_DIM = K_DIM
QKV_DIM = Q_DIM + K_DIM + V_DIM
BLOCK = 16
BLOCKS = HEAD_DIM // BLOCK
TOKEN_BLOCK = 16
VALUE_TOKEN_BLOCK = 64
TAU = 0.125
BOOTSTRAP_REPLICATES = 5000
OUTPUT = Path("results/triangular-value-encoding-h100-development.json")


@triton.jit
def qkv_epilogue_kernel(
    qkv_ptr,
    output_weight_ptr,
    cosine_ptr,
    sine_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    Q_HEADS: tl.constexpr,
    K_HEADS: tl.constexpr,
    HEAD: tl.constexpr,
    HALF_HEAD: tl.constexpr,
    Q_WIDTH: tl.constexpr,
    K_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TAU_VALUE: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    VALUE_TOKEN_BLOCK_SIZE: tl.constexpr,
    TVE: tl.constexpr,
):
    task = tl.program_id(0)
    token_tiles = (M + TOKEN_BLOCK_SIZE - 1) // TOKEN_BLOCK_SIZE
    qk_heads = Q_HEADS + K_HEADS
    qk_tasks = token_tiles * qk_heads
    if task < qk_tasks:
        token_tile = task // qk_heads
        head_program = task % qk_heads
        qk_token = token_tile * TOKEN_BLOCK_SIZE + tl.arange(0, TOKEN_BLOCK_SIZE)
        qk_token_mask = qk_token < M
        pair = tl.arange(0, HALF_HEAD)
        is_query = head_program < Q_HEADS
        projected_head = tl.where(is_query, head_program, head_program - Q_HEADS)
        base = tl.where(is_query, 0, Q_WIDTH)
        row = base + projected_head * HEAD
        even_ptr = qkv_ptr + qk_token[:, None] * N + row + pair[None, :]
        odd_ptr = even_ptr + HALF_HEAD
        even = tl.load(even_ptr, mask=qk_token_mask[:, None]).to(tl.float32)
        odd = tl.load(odd_ptr, mask=qk_token_mask[:, None]).to(tl.float32)
        cosine = tl.load(
            cosine_ptr + qk_token[:, None] * HALF_HEAD + pair[None, :],
            mask=qk_token_mask[:, None],
        ).to(tl.float32)
        sine = tl.load(
            sine_ptr + qk_token[:, None] * HALF_HEAD + pair[None, :],
            mask=qk_token_mask[:, None],
        ).to(tl.float32)
        tl.store(even_ptr, cosine * even - sine * odd, mask=qk_token_mask[:, None])
        tl.store(odd_ptr, sine * even + cosine * odd, mask=qk_token_mask[:, None])
    else:
        # Candidate-only tasks tile tokens and one 16x16 V block. This keeps
        # Q/K programs unchanged while mapping the small products to tensor cores.
        value_task = task - qk_tasks
        value_tasks_per_tile = K_HEADS * (HEAD // BLOCK_SIZE)
        token_tile = value_task // value_tasks_per_tile
        value_in_tile = value_task % value_tasks_per_tile
        kv_head = value_in_tile // (HEAD // BLOCK_SIZE)
        block_index = value_in_tile % (HEAD // BLOCK_SIZE)
        value_token = token_tile * VALUE_TOKEN_BLOCK_SIZE + tl.arange(0, VALUE_TOKEN_BLOCK_SIZE)
        value_token_mask = value_token < M
        target = tl.arange(0, BLOCK_SIZE)
        source = tl.arange(0, BLOCK_SIZE)
        block_start = block_index * BLOCK_SIZE
        value_base = Q_WIDTH + K_WIDTH + kv_head * HEAD + block_start
        representative_head = kv_head * GROUP_SIZE
        value_ptr = qkv_ptr + value_token[:, None] * N + value_base + source[None, :]
        value = tl.load(value_ptr, mask=value_token_mask[:, None]).to(tl.float32)
        coefficient = (tl.load(
            output_weight_ptr
            + (block_start + target[:, None]) * OUTPUT_WIDTH
            + representative_head * HEAD
            + block_start
            + source[None, :],
            mask=source[None, :] < target[:, None],
            other=0.0,
        ).to(tl.float32) / TAU_VALUE).to(tl.bfloat16)
        feature = (value * tl.abs(value)).to(tl.bfloat16)
        delta = tl.dot(
            feature, tl.trans(coefficient), out_dtype=tl.float32
        ).to(tl.bfloat16)
        encoded = value + delta.to(tl.float32)
        tl.store(value_ptr, encoded, mask=value_token_mask[:, None])


def qkv_epilogue(
    qkv: torch.Tensor,
    output_weight: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    *,
    candidate: bool,
) -> None:
    token_tiles = triton.cdiv(qkv.shape[0], TOKEN_BLOCK)
    qk_tasks = token_tiles * (QUERY_HEADS + KV_HEADS)
    value_token_tiles = triton.cdiv(qkv.shape[0], VALUE_TOKEN_BLOCK)
    value_tasks = value_token_tiles * KV_HEADS * BLOCKS if candidate else 0
    qkv_epilogue_kernel[(qk_tasks + value_tasks,)](
        qkv,
        output_weight,
        cosine,
        sine,
        M=qkv.shape[0],
        N=QKV_DIM,
        OUTPUT_WIDTH=HIDDEN,
        Q_HEADS=QUERY_HEADS,
        K_HEADS=KV_HEADS,
        HEAD=HEAD_DIM,
        HALF_HEAD=HALF,
        Q_WIDTH=Q_DIM,
        K_WIDTH=K_DIM,
        GROUP_SIZE=GROUPS,
        BLOCK_SIZE=BLOCK,
        TAU_VALUE=TAU,
        TOKEN_BLOCK_SIZE=TOKEN_BLOCK,
        VALUE_TOKEN_BLOCK_SIZE=VALUE_TOKEN_BLOCK,
        TVE=candidate,
        num_warps=4,
    )


def make_weights(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    qkv_weight = torch.randn(
        QKV_DIM,
        HIDDEN,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ) / math.sqrt(HIDDEN)
    output_weight = torch.randn(
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
            existing = output_weight[rows, columns]
            output_weight[rows, columns] = torch.triu(existing) + coefficient
    return qkv_weight.contiguous(), output_weight.contiguous()


def reference_epilogue(
    qkv: torch.Tensor,
    output_weight: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    *,
    candidate: bool,
) -> torch.Tensor:
    result = qkv.clone()
    query = result[:, :Q_DIM].view(-1, QUERY_HEADS, HEAD_DIM)
    key = result[:, Q_DIM:Q_DIM + K_DIM].view(-1, KV_HEADS, HEAD_DIM)
    cos = cosine[:, None].float()
    sin_ = sine[:, None].float()
    for tensor in (query, key):
        even = tensor[..., :HALF].float()
        odd = tensor[..., HALF:].float()
        tensor[..., :HALF] = (cos * even - sin_ * odd).to(tensor.dtype)
        tensor[..., HALF:] = (sin_ * even + cos * odd).to(tensor.dtype)
    if not candidate:
        return result
    values = result[:, Q_DIM + K_DIM:].view(-1, KV_HEADS, HEAD_DIM)
    encoded_blocks = []
    for block_start in range(0, HEAD_DIM, BLOCK):
        block = values[..., block_start:block_start + BLOCK]
        coefficients = []
        for kv_head in range(KV_HEADS):
            representative_head = kv_head * GROUPS
            physical = output_weight[
                block_start:block_start + BLOCK,
                representative_head * HEAD_DIM + block_start:
                representative_head * HEAD_DIM + block_start + BLOCK,
            ]
            coefficients.append(torch.tril(physical, diagonal=-1))
        coefficient = (
            torch.stack(coefficients).to(block.dtype).float() / TAU
        ).to(block.dtype)
        feature = (block.float() * block.float().abs()).to(block.dtype)
        delta = torch.matmul(
            coefficient[None], feature.unsqueeze(-1)
        ).squeeze(-1)
        encoded_blocks.append((block.float() + delta.float()).to(block.dtype))
    values.copy_(torch.cat(encoded_blocks, dim=-1))
    return result


def bootstrap_ratio(candidate: list[float], control: list[float], seed: int) -> dict[str, float]:
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(control, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, left.size, size=(BOOTSTRAP_REPLICATES, left.size))
    ratios = np.median(left[indices], axis=1) / np.median(right[indices], axis=1)
    return {
        "median_ratio": float(np.median(left) / np.median(right)),
        "lower_95": float(np.quantile(ratios, 0.025)),
        "upper_95": float(np.quantile(ratios, 0.975)),
    }


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(samples) * 1000.0,
        "p05_us": ordered[int(0.05 * (len(ordered) - 1))] * 1000.0,
        "p95_us": ordered[int(0.95 * (len(ordered) - 1))] * 1000.0,
    }


def benchmark(rows: int, trials: int, warmup: int, seed: int) -> dict[str, object]:
    generator = torch.Generator(device="cuda").manual_seed(seed + rows)
    inputs = torch.randn(
        rows, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    qkv_weight, output_weight = make_weights(seed)
    angles = torch.randn(rows, HALF, device="cuda", dtype=torch.float32, generator=generator)
    cosine = angles.cos().to(torch.bfloat16)
    sine = angles.sin().to(torch.bfloat16)
    outputs = {
        arm: torch.empty(rows, QKV_DIM, device="cuda", dtype=torch.bfloat16)
        for arm in ("canonical_control", "triangular_value_encoding")
    }
    l2_flush = torch.zeros(64 * 1024 * 1024, device="cuda", dtype=torch.bfloat16)

    def run(arm: str) -> None:
        torch.mm(inputs, qkv_weight.T, out=outputs[arm])
        qkv_epilogue(
            outputs[arm],
            output_weight,
            cosine,
            sine,
            candidate=(arm == "triangular_value_encoding"),
        )

    for _ in range(warmup):
        run("canonical_control")
        run("triangular_value_encoding")
    torch.cuda.synchronize()

    raw = inputs @ qkv_weight.T
    references = {
        arm: reference_epilogue(
            raw,
            output_weight,
            cosine,
            sine,
            candidate=(arm == "triangular_value_encoding"),
        )
        for arm in outputs
    }
    for arm in outputs:
        run(arm)
    torch.cuda.synchronize()
    correctness = {}
    for arm in outputs:
        difference = (outputs[arm].float() - references[arm].float()).abs()
        correctness[f"{arm}_bit_exact"] = bool(torch.equal(outputs[arm], references[arm]))
        correctness[f"{arm}_max_abs"] = float(difference.max())

    timings = {arm: [] for arm in outputs}
    order_generator = random.Random(seed + rows)
    for _ in range(trials):
        order = list(outputs)
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
        "rows": rows,
        "trials": trials,
        "correctness": correctness,
        "timing": {
            arm: summarize(samples) for arm, samples in timings.items()
        },
        "candidate_over_control": bootstrap_ratio(
            timings["triangular_value_encoding"],
            timings["canonical_control"],
            seed + 991,
        ),
        "ledger": {
            "qkv_weight_bytes_each": qkv_weight.numel() * qkv_weight.element_size(),
            "output_weight_bytes_each": output_weight.numel() * output_weight.element_size(),
            "learned_weight_bytes_added": 0,
            "kv_bytes_added": 0,
            "metadata_bits": 0,
            "epilogue_token_tile": TOKEN_BLOCK,
            "value_token_tile": VALUE_TOKEN_BLOCK,
            "control_epilogue_programs_per_tile": QUERY_HEADS + KV_HEADS,
            "candidate_epilogue_programs_per_tile": QUERY_HEADS + KV_HEADS + KV_HEADS * BLOCKS,
            "coefficient_slots_per_kv_group": BLOCKS * BLOCK * (BLOCK - 1) // 2,
            "coefficient_slots_per_layer": KV_HEADS * BLOCKS * BLOCK * (BLOCK - 1) // 2,
            "triangular_mac_per_token_layer": KV_HEADS * BLOCKS * BLOCK * (BLOCK - 1) // 2,
            "signed_square_per_token_layer": KV_HEADS * HEAD_DIM,
            "dense_masked_multiply_slots_per_token_layer": KV_HEADS * BLOCKS * BLOCK * BLOCK,
            "tensor_core_multiply_slots_per_token_layer": KV_HEADS * BLOCKS * BLOCK * BLOCK,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="*", default=[1, 8, 32, 128, 512, 2048])
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=503)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    cells = [
        benchmark(rows, args.trials, args.warmup, args.seed + index * 1009)
        for index, rows in enumerate(args.rows)
    ]
    payload = {
        "schema": "triangular-value-encoding-h100-development-v1",
        "device": torch.cuda.get_device_name(0),
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": triton.__version__,
        },
        "shape": {
            "hidden": HIDDEN,
            "query_heads": QUERY_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "block": BLOCK,
            "qkv_width": QKV_DIM,
        },
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
