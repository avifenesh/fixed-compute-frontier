#!/usr/bin/env python3
"""H100 development gate for packed gauge-slot QKV projection plus RoPE."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics

import torch
import triton
import triton.language as tl


HIDDEN = 4096
QUERY_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
PAIRS = HEAD_DIM // 2
Q_DIM = QUERY_HEADS * HEAD_DIM
K_DIM = KV_HEADS * HEAD_DIM
V_DIM = K_DIM
QKV_DIM = Q_DIM + K_DIM + V_DIM
TAU = 1.0 / math.sqrt(HEAD_DIM)
OUTPUT = Path("results/gauge-slot-qkv-h100-development.json")


MATMUL_CONFIGS = [
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=8),
]


@triton.autotune(configs=MATMUL_CONFIGS, key=["M", "DECODE_PACKED", "G2"])
@triton.jit
def qkv_matmul_kernel(
    a_ptr,
    b_ptr,
    atlas_ptr,
    cosine_ptr,
    sine_ptr,
    output_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    Q_START: tl.constexpr,
    K_START: tl.constexpr,
    K_END: tl.constexpr,
    HEAD: tl.constexpr,
    HALF: tl.constexpr,
    TAU_VALUE: tl.constexpr,
    DECODE_PACKED: tl.constexpr,
    FUSE_ROPE: tl.constexpr,
    G2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]

    if DECODE_PACKED:
        key_mask = (offs_n >= K_START) & (offs_n < K_END)
        key_coordinate = offs_n - K_START
        kv_head = key_coordinate // HEAD
        local = key_coordinate % HEAD
        pair = tl.where(local < HALF, local, local - HALF)
        atlas_index = kv_head * HALF + pair
        pivot = tl.load(atlas_ptr + atlas_index, mask=key_mask, other=0)
        odd = local >= HALF

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for block_start in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K - block_start * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        if DECODE_PACKED:
            absolute_k = block_start * BLOCK_K + offs_k
            is_pivot = key_mask[None, :] & (absolute_k[:, None] == pivot[None, :])
            canonical = tl.where(odd[None, :], TAU_VALUE, 0.0).to(tl.bfloat16)
            b = tl.where(is_pivot, canonical, b).to(tl.bfloat16)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    if FUSE_ROPE:
        block_start_n = pid_n * BLOCK_N
        block_is_query = block_start_n < K_START
        block_is_key = (block_start_n >= K_START) & (block_start_n < K_END)
        block_uses_rope = block_is_query | block_is_key
        pairwise = tl.reshape(accumulator, (BLOCK_M, 2, HALF))
        pairwise = tl.permute(pairwise, (0, 2, 1))
        even, odd = tl.split(pairwise)
        original_even = even
        original_odd = odd
        if G2:
            coefficient = tl.load(
                b_ptr + offs_n * K + pivot,
                mask=key_mask & (offs_n < N),
                other=0.0,
            ).to(tl.float32)
            coefficient_pair = tl.reshape(coefficient, (2, HALF))
            coefficient_pair = tl.permute(coefficient_pair, (1, 0))
            alpha, beta = tl.split(coefficient_pair)
            even = tl.where(
                block_is_key,
                original_even + alpha[None, :] * original_odd * original_odd,
                original_even,
            )
            odd = tl.where(
                block_is_key,
                original_odd + beta[None, :] * original_even * original_even,
                original_odd,
            )
        cosine = tl.load(
            cosine_ptr + offs_m[:, None] * HALF + tl.arange(0, HALF)[None, :],
            mask=offs_m[:, None] < M,
            other=1.0,
        ).to(tl.float32)
        sine = tl.load(
            sine_ptr + offs_m[:, None] * HALF + tl.arange(0, HALF)[None, :],
            mask=offs_m[:, None] < M,
            other=0.0,
        ).to(tl.float32)
        rotated_even = cosine * even - sine * odd
        rotated_odd = sine * even + cosine * odd
        rotated = tl.join(rotated_even, rotated_odd)
        rotated = tl.permute(rotated, (0, 2, 1))
        rotated = tl.reshape(rotated, (BLOCK_M, BLOCK_N))
        accumulator = tl.where(block_uses_rope, rotated, accumulator)

    output = accumulator.to(tl.bfloat16)
    output_ptrs = output_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(output_ptrs, output, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def rope_inplace_kernel(
    qkv_ptr,
    packed_weight_ptr,
    atlas_ptr,
    cosine_ptr,
    sine_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    WEIGHT_K: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS_VALUE: tl.constexpr,
    HEAD: tl.constexpr,
    HALF: tl.constexpr,
    Q_WIDTH: tl.constexpr,
    K_WIDTH: tl.constexpr,
    G2: tl.constexpr,
):
    token = tl.program_id(0)
    head_program = tl.program_id(1)
    pair = tl.arange(0, HALF)
    is_query = head_program < Q_HEADS
    projected_head = tl.where(is_query, head_program, head_program - Q_HEADS)
    base = tl.where(is_query, 0, Q_WIDTH)
    row = base + projected_head * HEAD
    even_ptr = qkv_ptr + token * N + row + pair
    odd_ptr = even_ptr + HALF
    even = tl.load(even_ptr)
    odd = tl.load(odd_ptr)

    if G2:
        is_key = ~is_query
        atlas_index = projected_head * HALF + pair
        pivot = tl.load(atlas_ptr + atlas_index, mask=is_key, other=0)
        alpha_row = Q_WIDTH + projected_head * HEAD + pair
        beta_row = alpha_row + HALF
        alpha = tl.load(
            packed_weight_ptr + alpha_row * WEIGHT_K + pivot,
            mask=is_key,
            other=0.0,
        ).to(tl.float32)
        beta = tl.load(
            packed_weight_ptr + beta_row * WEIGHT_K + pivot,
            mask=is_key,
            other=0.0,
        ).to(tl.float32)
        original_even = even.to(tl.float32)
        original_odd = odd.to(tl.float32)
        even = tl.where(
            is_key,
            original_even + alpha * original_odd * original_odd,
            original_even,
        )
        odd = tl.where(
            is_key,
            original_odd + beta * original_even * original_even,
            original_odd,
        )

    cosine = tl.load(cosine_ptr + token * HALF + pair).to(tl.float32)
    sine = tl.load(sine_ptr + token * HALF + pair).to(tl.float32)
    even_float = even.to(tl.float32)
    odd_float = odd.to(tl.float32)
    tl.store(even_ptr, cosine * even_float - sine * odd_float)
    tl.store(odd_ptr, sine * even_float + cosine * odd_float)


def qkv_matmul(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    atlas: torch.Tensor,
    output: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    *,
    decode_packed: bool,
    fuse_rope: bool,
    g2: bool,
) -> None:
    rows = inputs.shape[0]
    grid = lambda meta: (
        triton.cdiv(rows, meta["BLOCK_M"]),
        triton.cdiv(QKV_DIM, meta["BLOCK_N"]),
    )
    qkv_matmul_kernel[grid](
        inputs,
        weight,
        atlas,
        cosine,
        sine,
        output,
        M=rows,
        K=HIDDEN,
        N=QKV_DIM,
        Q_START=0,
        K_START=Q_DIM,
        K_END=Q_DIM + K_DIM,
        HEAD=HEAD_DIM,
        HALF=PAIRS,
        TAU_VALUE=TAU,
        DECODE_PACKED=decode_packed,
        FUSE_ROPE=fuse_rope,
        G2=g2,
    )


def rope_inplace(
    output: torch.Tensor,
    packed_weight: torch.Tensor,
    atlas: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    *,
    g2: bool,
) -> None:
    rope_inplace_kernel[(output.shape[0], QUERY_HEADS + KV_HEADS)](
        output,
        packed_weight,
        atlas,
        cosine,
        sine,
        M=output.shape[0],
        N=QKV_DIM,
        WEIGHT_K=HIDDEN,
        Q_HEADS=QUERY_HEADS,
        KV_HEADS_VALUE=KV_HEADS,
        HEAD=HEAD_DIM,
        HALF=PAIRS,
        Q_WIDTH=Q_DIM,
        K_WIDTH=K_DIM,
        G2=g2,
        num_warps=2,
    )


def make_weights(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    dense = torch.randn(
        QKV_DIM, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=generator
    ) / math.sqrt(HIDDEN)
    packed = dense.clone()
    atlas = torch.randint(
        0,
        HIDDEN,
        (KV_HEADS, PAIRS),
        device="cuda",
        dtype=torch.int32,
        generator=generator,
    )
    coefficients = 0.02 * torch.randn(
        KV_HEADS, PAIRS, 2, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    for kv_head in range(KV_HEADS):
        for pair in range(PAIRS):
            column = int(atlas[kv_head, pair])
            even_row = Q_DIM + kv_head * HEAD_DIM + pair
            odd_row = even_row + PAIRS
            dense[even_row, column] = 0.0
            dense[odd_row, column] = TAU
            packed[even_row, column] = coefficients[kv_head, pair, 0]
            packed[odd_row, column] = coefficients[kv_head, pair, 1]
    return dense.contiguous(), packed.contiguous(), atlas.contiguous()


def reference_candidate(
    inputs: torch.Tensor,
    dense: torch.Tensor,
    packed: torch.Tensor,
    atlas: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    output = inputs @ dense.T
    query = output[:, :Q_DIM].view(-1, QUERY_HEADS, HEAD_DIM)
    key = output[:, Q_DIM:Q_DIM + K_DIM].view(-1, KV_HEADS, HEAD_DIM)
    q_even, q_odd = query[..., :PAIRS], query[..., PAIRS:]
    k_even, k_odd = key[..., :PAIRS], key[..., PAIRS:]
    original_q_even = q_even.float()
    original_q_odd = q_odd.float()
    rows = torch.arange(KV_HEADS, device="cuda")[:, None] * HEAD_DIM
    alpha_rows = Q_DIM + rows + torch.arange(PAIRS, device="cuda")[None]
    beta_rows = alpha_rows + PAIRS
    alpha = packed[alpha_rows, atlas].float()[None]
    beta = packed[beta_rows, atlas].float()[None]
    original_even = k_even.float()
    original_odd = k_odd.float()
    k_even = original_even + alpha * original_odd.square()
    k_odd = original_odd + beta * original_even.square()
    cos_q = cosine[:, None].float()
    sin_q = sine[:, None].float()
    query[..., :PAIRS] = (cos_q * original_q_even - sin_q * original_q_odd).to(query.dtype)
    query[..., PAIRS:] = (sin_q * original_q_even + cos_q * original_q_odd).to(query.dtype)
    key[..., :PAIRS] = (cos_q * k_even - sin_q * k_odd).to(key.dtype)
    key[..., PAIRS:] = (sin_q * k_even + cos_q * k_odd).to(key.dtype)
    return output


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(samples) * 1000.0,
        "p05_us": ordered[int(0.05 * (len(ordered) - 1))] * 1000.0,
        "p95_us": ordered[int(0.95 * (len(ordered) - 1))] * 1000.0,
    }


def benchmark_cell(rows: int, trials: int, warmup: int, seed: int) -> dict[str, object]:
    generator = torch.Generator(device="cuda").manual_seed(seed + rows)
    inputs = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=generator)
    dense, packed, atlas = make_weights(seed)
    angles = torch.randn(rows, PAIRS, device="cuda", dtype=torch.float32, generator=generator)
    cosine = torch.cos(angles).to(torch.bfloat16)
    sine = torch.sin(angles).to(torch.bfloat16)
    outputs = {
        name: torch.empty(rows, QKV_DIM, device="cuda", dtype=torch.bfloat16)
        for name in ("torch_baseline", "triton_baseline", "decoded_control", "gauge_slot_g2")
    }
    # Standardize each sample to a cold-weight condition. The 128 MiB flush is
    # larger than H100's L2 and executes before, outside, the timed interval.
    l2_flush = torch.zeros(64 * 1024 * 1024, device="cuda", dtype=torch.bfloat16)

    def run_arm(name: str) -> None:
        if name == "torch_baseline":
            torch.mm(inputs, dense.T, out=outputs[name])
            rope_inplace(outputs[name], packed, atlas, cosine, sine, g2=False)
        elif name == "triton_baseline":
            qkv_matmul(
                inputs,
                dense,
                atlas,
                outputs[name],
                cosine,
                sine,
                decode_packed=False,
                fuse_rope=True,
                g2=False,
            )
        elif name == "decoded_control":
            qkv_matmul(
                inputs,
                packed,
                atlas,
                outputs[name],
                cosine,
                sine,
                decode_packed=True,
                fuse_rope=True,
                g2=False,
            )
        elif name == "gauge_slot_g2":
            qkv_matmul(
                inputs,
                packed,
                atlas,
                outputs[name],
                cosine,
                sine,
                decode_packed=True,
                fuse_rope=True,
                g2=True,
            )
        else:
            raise ValueError(name)

    arms = list(outputs)
    for _ in range(warmup):
        for arm in arms:
            run_arm(arm)
    torch.cuda.synchronize()
    reference = reference_candidate(inputs, dense, packed, atlas, cosine, sine)
    run_arm("gauge_slot_g2")
    torch.cuda.synchronize()
    difference = (outputs["gauge_slot_g2"].float() - reference.float()).abs()
    correctness = {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "allclose_atol_0_25_rtol_0_02": bool(torch.allclose(
            outputs["gauge_slot_g2"].float(), reference.float(), atol=0.25, rtol=0.02
        )),
    }

    rng = random.Random(seed + 10 * rows)
    records: dict[str, list[float]] = {arm: [] for arm in arms}
    events = []
    for _ in range(trials):
        order = arms.copy()
        rng.shuffle(order)
        for arm in order:
            l2_flush.add_(1)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_arm(arm)
            end.record()
            events.append((arm, start, end))
    torch.cuda.synchronize()
    for arm, start, end in events:
        records[arm].append(start.elapsed_time(end))
    timing = {arm: summarize(values) for arm, values in records.items()}
    baseline = timing["torch_baseline"]["median_us"]
    for arm in arms[1:]:
        timing[arm]["median_ratio_to_torch_baseline"] = (
            timing[arm]["median_us"] / baseline
        )
    return {
        "rows": rows,
        "correctness": correctness,
        "timing": timing,
        "logical": {
            "resident_weight_bytes_each": dense.numel() * dense.element_size(),
            "output_bytes_each": outputs["torch_baseline"].numel() * outputs["torch_baseline"].element_size(),
            "atlas_bits": atlas.numel() * math.ceil(math.log2(HIDDEN)),
            "qkv_values": dense.numel(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="1,8,32,128,512,2048")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3181)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 is required")
    payload = {
        "schema": "gauge-slot-qkv-h100-development-v1",
        "device": torch.cuda.get_device_name(0),
        "shape": {
            "hidden": HIDDEN,
            "query_heads": QUERY_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "qkv_dim": QKV_DIM,
        },
        "cells": [],
    }
    for rows in [int(value) for value in args.rows.split(",") if value]:
        print(json.dumps({"starting_rows": rows}), flush=True)
        payload["cells"].append(benchmark_cell(rows, args.trials, args.warmup, args.seed))
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload["cells"][-1]["timing"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
