#!/usr/bin/env python3
"""H100 gate for TVE fused into the required paged-KV cache write."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from experiments.triangular_value_encoding_h100 import (
    BLOCK,
    BLOCKS,
    BOOTSTRAP_REPLICATES,
    GROUPS,
    HALF,
    HEAD_DIM,
    HIDDEN,
    K_DIM,
    KV_HEADS,
    Q_DIM,
    QKV_DIM,
    QUERY_HEADS,
    TAU,
    V_DIM,
    make_weights,
)
from experiments.triangular_value_encoding_attention import serving_value_encoding


QK_TOKEN_BLOCK = 16
VALUE_TOKEN_BLOCK = 64
OUTPUT = Path("results/triangular-value-encoding-cache-h100-development.json")


@triton.jit
def cache_write_epilogue_kernel(
    qkv_ptr,
    output_weight_ptr,
    cosine_ptr,
    sine_ptr,
    key_cache_ptr,
    value_cache_ptr,
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
    QK_TOKEN_BLOCK_SIZE: tl.constexpr,
    VALUE_TOKEN_BLOCK_SIZE: tl.constexpr,
    TVE: tl.constexpr,
):
    task = tl.program_id(0)
    qk_tiles = (M + QK_TOKEN_BLOCK_SIZE - 1) // QK_TOKEN_BLOCK_SIZE
    qk_heads = Q_HEADS + K_HEADS
    qk_tasks = qk_tiles * qk_heads
    if task < qk_tasks:
        token_tile = task // qk_heads
        head_program = task % qk_heads
        qk_token = token_tile * QK_TOKEN_BLOCK_SIZE + tl.arange(0, QK_TOKEN_BLOCK_SIZE)
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
        rotated_even = cosine * even - sine * odd
        rotated_odd = sine * even + cosine * odd
        tl.store(even_ptr, rotated_even, mask=qk_token_mask[:, None])
        tl.store(odd_ptr, rotated_odd, mask=qk_token_mask[:, None])
        if head_program >= Q_HEADS:
            key_base = projected_head * HEAD
            key_even_ptr = key_cache_ptr + qk_token[:, None] * K_WIDTH + key_base + pair[None, :]
            tl.store(key_even_ptr, rotated_even, mask=qk_token_mask[:, None])
            tl.store(key_even_ptr + HALF_HEAD, rotated_odd, mask=qk_token_mask[:, None])
    else:
        value_task = task - qk_tasks
        tasks_per_tile = K_HEADS * (HEAD // BLOCK_SIZE)
        token_tile = value_task // tasks_per_tile
        in_tile = value_task % tasks_per_tile
        kv_head = in_tile // (HEAD // BLOCK_SIZE)
        block_index = in_tile % (HEAD // BLOCK_SIZE)
        value_token = token_tile * VALUE_TOKEN_BLOCK_SIZE + tl.arange(0, VALUE_TOKEN_BLOCK_SIZE)
        value_token_mask = value_token < M
        target = tl.arange(0, BLOCK_SIZE)
        source = tl.arange(0, BLOCK_SIZE)
        block_start = block_index * BLOCK_SIZE
        qkv_value_base = Q_WIDTH + K_WIDTH + kv_head * HEAD + block_start
        qkv_value_ptr = (
            qkv_ptr + value_token[:, None] * N + qkv_value_base + source[None, :]
        )
        value = tl.load(
            qkv_value_ptr,
            mask=value_token_mask[:, None],
        ).to(tl.float32)
        encoded = value
        if TVE:
            representative_head = kv_head * GROUP_SIZE
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
            encoded += delta.to(tl.float32)
            tl.store(qkv_value_ptr, encoded, mask=value_token_mask[:, None])
        cache_base = kv_head * HEAD + block_start
        tl.store(
            value_cache_ptr + value_token[:, None] * K_WIDTH + cache_base + target[None, :],
            encoded,
            mask=value_token_mask[:, None],
        )


def cache_write_epilogue(
    qkv: torch.Tensor,
    output_weight: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    candidate: bool,
) -> None:
    qk_tiles = triton.cdiv(qkv.shape[0], QK_TOKEN_BLOCK)
    value_tiles = triton.cdiv(qkv.shape[0], VALUE_TOKEN_BLOCK)
    tasks = (
        qk_tiles * (QUERY_HEADS + KV_HEADS)
        + value_tiles * KV_HEADS * BLOCKS
    )
    cache_write_epilogue_kernel[(tasks,)](
        qkv,
        output_weight,
        cosine,
        sine,
        key_cache,
        value_cache,
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
        QK_TOKEN_BLOCK_SIZE=QK_TOKEN_BLOCK,
        VALUE_TOKEN_BLOCK_SIZE=VALUE_TOKEN_BLOCK,
        TVE=candidate,
        num_warps=4,
    )


def reference(
    raw: torch.Tensor,
    output_weight: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    *,
    candidate: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    qkv = raw.clone()
    query = qkv[:, :Q_DIM].view(-1, QUERY_HEADS, HEAD_DIM)
    key = qkv[:, Q_DIM:Q_DIM + K_DIM].view(-1, KV_HEADS, HEAD_DIM)
    cos = cosine[:, None].float()
    sin_ = sine[:, None].float()
    for tensor in (query, key):
        even = tensor[..., :HALF].float()
        odd = tensor[..., HALF:].float()
        tensor[..., :HALF] = (cos * even - sin_ * odd).to(tensor.dtype)
        tensor[..., HALF:] = (sin_ * even + cos * odd).to(tensor.dtype)
    key_cache = key.reshape(-1, K_DIM).clone()
    values = qkv[:, Q_DIM + K_DIM:].view(-1, KV_HEADS, HEAD_DIM)
    if not candidate:
        raw_values = values.reshape(-1, V_DIM).clone()
        return qkv, key_cache, raw_values, raw_values
    pieces = []
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
        pieces.append((block.float() + delta.float()).to(block.dtype))
    algebra_value_cache = torch.cat(pieces, dim=-1).reshape(-1, V_DIM)
    value_cache = serving_value_encoding(
        values.contiguous(),
        output_weight,
        query_groups=GROUPS,
        block_size=BLOCK,
        tau=TAU,
    ).reshape(-1, V_DIM)
    qkv[:, Q_DIM + K_DIM:].copy_(value_cache)
    return qkv, key_cache, value_cache, algebra_value_cache


def bootstrap_ratio(candidate: list[float], control: list[float], seed: int) -> dict[str, float]:
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(control, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, left.size, size=(BOOTSTRAP_REPLICATES, left.size))
    ratio = np.median(left[indices], axis=1) / np.median(right[indices], axis=1)
    return {
        "median_ratio": float(np.median(left) / np.median(right)),
        "lower_95": float(np.quantile(ratio, 0.025)),
        "upper_95": float(np.quantile(ratio, 0.975)),
    }


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median_us": statistics.median(values) * 1000.0,
        "p05_us": ordered[int(0.05 * (len(ordered) - 1))] * 1000.0,
        "p95_us": ordered[int(0.95 * (len(ordered) - 1))] * 1000.0,
    }


@torch.no_grad()
def downstream_error(
    actual_values: torch.Tensor,
    expected_values: torch.Tensor,
    output_weight: torch.Tensor,
    seed: int,
) -> dict[str, float]:
    """Propagate executor drift through a fixed attention-mix/O/NLL proxy."""
    rows = actual_values.shape[0]
    generator = torch.Generator(device="cuda").manual_seed(seed)
    probabilities = torch.randn(
        QUERY_HEADS, 16, rows, device="cuda", dtype=torch.float32, generator=generator
    ).softmax(-1)
    actual = actual_values.view(rows, KV_HEADS, HEAD_DIM).float()
    expected = expected_values.view(rows, KV_HEADS, HEAD_DIM).float()

    def mix(values: torch.Tensor) -> torch.Tensor:
        heads = [
            probabilities[head] @ values[:, head // GROUPS]
            for head in range(QUERY_HEADS)
        ]
        return torch.cat(heads, dim=-1)

    actual_mix = mix(actual)
    expected_mix = mix(expected)
    actual_output = F.linear(actual_mix.to(torch.bfloat16), output_weight).float()
    expected_output = F.linear(expected_mix.to(torch.bfloat16), output_weight).float()
    readout = torch.randn(
        256, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=generator
    ) / HIDDEN ** 0.5
    actual_logits = F.linear(actual_output.to(torch.bfloat16), readout).float()
    expected_logits = F.linear(expected_output.to(torch.bfloat16), readout).float()
    labels = torch.arange(16, device="cuda") % 256
    actual_nll = F.cross_entropy(actual_logits, labels)
    expected_nll = F.cross_entropy(expected_logits, labels)
    mix_difference = (actual_mix - expected_mix).abs()
    output_difference = (actual_output - expected_output).abs()
    return {
        "attention_mix_max_abs": float(mix_difference.max()),
        "attention_mix_mean_abs": float(mix_difference.mean()),
        "output_projection_max_abs": float(output_difference.max()),
        "output_projection_mean_abs": float(output_difference.mean()),
        "proxy_nll_abs_difference": float((actual_nll - expected_nll).abs()),
    }


def benchmark(rows: int, trials: int, warmup: int, seed: int) -> dict[str, object]:
    generator = torch.Generator(device="cuda").manual_seed(seed + rows)
    inputs = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=generator)
    qkv_weight, output_weight = make_weights(seed)
    angles = torch.randn(rows, HALF, device="cuda", dtype=torch.float32, generator=generator)
    cosine = angles.cos().to(torch.bfloat16)
    sine = angles.sin().to(torch.bfloat16)
    arms = ("canonical_control", "triangular_value_encoding")
    qkv = {arm: torch.empty(rows, QKV_DIM, device="cuda", dtype=torch.bfloat16) for arm in arms}
    key_cache = {arm: torch.empty(rows, K_DIM, device="cuda", dtype=torch.bfloat16) for arm in arms}
    value_cache = {arm: torch.empty(rows, V_DIM, device="cuda", dtype=torch.bfloat16) for arm in arms}
    l2_touch = torch.zeros(64 * 1024 * 1024, device="cuda", dtype=torch.bfloat16)

    def run(arm: str) -> None:
        torch.mm(inputs, qkv_weight.T, out=qkv[arm])
        cache_write_epilogue(
            qkv[arm], output_weight, cosine, sine, key_cache[arm], value_cache[arm],
            candidate=(arm == "triangular_value_encoding"),
        )

    for _ in range(warmup):
        for arm in arms:
            run(arm)
    torch.cuda.synchronize()
    raw = inputs @ qkv_weight.T
    expected = {
        arm: reference(raw, output_weight, cosine, sine, candidate=(arm == "triangular_value_encoding"))
        for arm in arms
    }
    for arm in arms:
        run(arm)
    torch.cuda.synchronize()
    correctness = {}
    for arm in arms:
        actual = (qkv[arm], key_cache[arm], value_cache[arm])
        for name, actual_tensor, expected_tensor in zip(("qkv", "key_cache", "value_cache"), actual, expected[arm]):
            difference = (actual_tensor.float() - expected_tensor.float()).abs()
            correctness[f"{arm}_{name}_max_abs"] = float(difference.max())
            correctness[f"{arm}_{name}_mean_abs"] = float(difference.mean())
            correctness[f"{arm}_{name}_mismatch_fraction"] = float(
                (difference != 0).float().mean()
            )
            correctness[f"{arm}_{name}_bit_exact"] = bool(torch.equal(actual_tensor, expected_tensor))
    correctness["nonzero_candidate_value_delta"] = bool(
        torch.count_nonzero(value_cache[arms[1]] - value_cache[arms[0]])
    )
    algebra_difference = (
        value_cache[arms[1]].float() - expected[arms[1]][3].float()
    ).abs()
    correctness.update({
        "candidate_algebra_rounding_max_abs": float(algebra_difference.max()),
        "candidate_algebra_rounding_mean_abs": float(algebra_difference.mean()),
        "candidate_algebra_rounding_mismatch_fraction": float(
            (algebra_difference != 0).float().mean()
        ),
    })
    propagated_error = downstream_error(
        value_cache[arms[1]], expected[arms[1]][3], output_weight, seed + 1777
    )

    timings = {arm: [] for arm in arms}
    order_generator = random.Random(seed + rows)
    for _ in range(trials):
        order = list(arms)
        order_generator.shuffle(order)
        for arm in order:
            l2_touch.add_(1.0)
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
        "propagated_executor_error": propagated_error,
        "timing": {arm: summarize(timings[arm]) for arm in arms},
        "candidate_over_control": bootstrap_ratio(timings[arms[1]], timings[arms[0]], seed + 991),
        "ledger": {
            "learned_weight_bytes_added": 0,
            "kv_bytes_added": 0,
            "metadata_bits": 0,
            "control_and_candidate_cache_bytes_each": rows * (K_DIM + V_DIM) * 2,
            "coefficient_slots_per_layer": KV_HEADS * BLOCKS * BLOCK * (BLOCK - 1) // 2,
            "logical_triangular_mac_per_token_layer": KV_HEADS * BLOCKS * BLOCK * (BLOCK - 1) // 2,
            "executed_tensor_core_mac_per_token_layer": KV_HEADS * BLOCKS * BLOCK * BLOCK,
            "logical_and_executed_signed_square_per_token_layer": KV_HEADS * HEAD_DIM,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="*", default=[1, 8, 32, 128, 512, 2048])
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=607)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    cells = [benchmark(rows, args.trials, args.warmup, args.seed + i * 1009) for i, rows in enumerate(args.rows)]
    payload = {
        "schema": "triangular-value-encoding-cache-h100-development-v1",
        "device": torch.cuda.get_device_name(0),
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__},
        "scope": "BF16 QKV GEMM plus FP32-one-store Q/K RoPE plus required K/V cache write",
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
