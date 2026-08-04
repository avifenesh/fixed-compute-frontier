#!/usr/bin/env python3
"""Exploratory/frozen H100 gate for peak-lifted attention.

Run with --smoke while developing. Formal mode requires a preregistration file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


OUTPUT = Path("results/peak-lifted-attention-h100.json")
PREREGISTRATION = Path("results/peak-lifted-attention-h100-preregistration.md")
SOURCE = Path(__file__)
HEAD_DIM = 128
SEMANTIC_CHUNK = 64
SEED = 20260729


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@triton.jit
def attention_forward_kernel(
    query, key, value, coefficients, output, logsumexp,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_cb, stride_ch,
    stride_ob, stride_oh, stride_om, stride_od,
    batch, heads, kv_heads, query_length, key_length,
    softmax_scale,
    CANDIDATE: tl.constexpr, CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    tile_m = tl.program_id(0)
    head_batch = tl.program_id(1)
    head = head_batch % heads
    batch_index = head_batch // heads
    group_size = heads // kv_heads
    kv_head = head // group_size

    log2e: tl.constexpr = 1.4426950408889634
    ln2: tl.constexpr = 0.6931471805599453
    offsets_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n_base = tl.arange(0, BLOCK_N)
    offsets_d = tl.arange(0, BLOCK_D)

    q_ptrs = (
        query + batch_index * stride_qb + head * stride_qh
        + offsets_m[:, None] * stride_qm + offsets_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=offsets_m[:, None] < query_length, other=0.0)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    running_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    running_mass = tl.zeros((BLOCK_M,), dtype=tl.float32)

    if CANDIDATE:
        lift = tl.load(
            coefficients + batch_index * stride_cb + head * stride_ch
        ).to(tl.float32)
    else:
        lift = 0.0

    query_position = key_length - query_length + offsets_m

    for start_n in range(0, key_length, BLOCK_N):
        offsets_n = start_n + offsets_n_base
        k_ptrs = (
            key + batch_index * stride_kb + kv_head * stride_kh
            + offsets_d[:, None] * stride_kd + offsets_n[None, :] * stride_kn
        )
        v_ptrs = (
            value + batch_index * stride_vb + kv_head * stride_vh
            + offsets_n[:, None] * stride_vn + offsets_d[None, :] * stride_vd
        )
        k = tl.load(k_ptrs, mask=offsets_n[None, :] < key_length, other=0.0)
        v = tl.load(v_ptrs, mask=offsets_n[:, None] < key_length, other=0.0)
        if BLOCK_M == 1:
            scores = tl.sum((q.T * k).to(tl.float32), axis=0, keep_dims=True)
        else:
            scores = tl.dot(q, k, input_precision="ieee")
        scores *= softmax_scale * log2e

        valid = offsets_n[None, :] < key_length
        if CAUSAL:
            valid &= offsets_n[None, :] <= query_position[:, None]
        valid &= offsets_m[:, None] < query_length
        scores = tl.where(valid, scores, -float("inf"))

        if CANDIDATE:
            block_max = tl.max(scores, axis=1)
            has_valid = block_max != -float("inf")
            safe_block_max = tl.where(has_valid, block_max, 0.0)
            offset = lift * safe_block_max
            lifted_max = tl.where(has_valid, block_max + offset, -float("inf"))
            new_max = tl.maximum(running_max, lifted_max)
            old_scale = tl.math.exp2(running_max - new_max)
            block_center = new_max - offset
            probability = tl.math.exp2(scores - block_center[:, None])
        else:
            new_max = tl.maximum(running_max, tl.max(scores, axis=1))
            old_scale = tl.math.exp2(running_max - new_max)
            probability = tl.math.exp2(scores - new_max[:, None])

        accumulator *= old_scale[:, None]
        if BLOCK_M == 1:
            accumulator += tl.sum(
                probability.T * v, axis=0, keep_dims=True
            )
        else:
            accumulator += tl.dot(probability.to(q.dtype), v, input_precision="ieee")
        running_mass = running_mass * old_scale + tl.sum(probability, axis=1)
        running_max = new_max

    normalized = accumulator / running_mass[:, None]
    output_ptrs = (
        output + batch_index * stride_ob + head * stride_oh
        + offsets_m[:, None] * stride_om + offsets_d[None, :] * stride_od
    )
    tl.store(output_ptrs, normalized, mask=offsets_m[:, None] < query_length)
    lse_offsets = (batch_index * heads + head) * query_length + offsets_m
    tl.store(
        logsumexp + lse_offsets,
        running_max * ln2 + tl.log(running_mass),
        mask=offsets_m < query_length,
    )


def peak_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    candidate: bool,
    causal: bool,
    chunk_size: int = SEMANTIC_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query.dtype != torch.bfloat16 or key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("BF16 Q/K/V required")
    if not all(tensor.is_cuda and tensor.is_contiguous() for tensor in (query, key, value, coefficients)):
        raise ValueError("contiguous CUDA tensors required")
    batch, heads, query_length, head_dim = query.shape
    key_batch, kv_heads, key_length, key_dim = key.shape
    if value.shape != key.shape or key_batch != batch or key_dim != head_dim:
        raise ValueError("shape mismatch")
    if head_dim != HEAD_DIM or heads % kv_heads:
        raise ValueError("frozen head geometry required")
    if coefficients.shape != (batch, heads):
        raise ValueError("coefficient shape mismatch")
    if chunk_size not in (64, 128):
        raise ValueError("chunk size must be 64 or 128")
    if key_length % chunk_size:
        raise ValueError("key length must align to semantic chunks")
    if query_length == 1:
        block_m, warps, stages = 1, 4, 1
    else:
        block_m, warps, stages = 64, 4, 3
    output = torch.empty_like(query)
    lse = torch.empty((batch, heads, query_length), device=query.device, dtype=torch.float32)
    grid = (triton.cdiv(query_length, block_m), batch * heads)
    attention_forward_kernel[grid](
        query, key, value, coefficients, output, lse,
        *query.stride(), *key.stride(), *value.stride(), *coefficients.stride(),
        *output.stride(),
        batch, heads, kv_heads, query_length, key_length,
        softmax_scale=1.0 / math.sqrt(head_dim),
        CANDIDATE=candidate, CAUSAL=causal,
        BLOCK_M=block_m, BLOCK_N=chunk_size, BLOCK_D=head_dim,
        num_warps=warps, num_stages=stages,
    )
    return output, lse


def reference_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    coefficients: torch.Tensor, *, candidate: bool, causal: bool,
    chunk_size: int = SEMANTIC_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, query_length, head_dim = query.shape
    _, kv_heads, key_length, _ = key.shape
    group = heads // kv_heads
    key = key.repeat_interleave(group, dim=1).float()
    value = value.repeat_interleave(group, dim=1).float()
    scores = torch.matmul(query.float(), key.transpose(-1, -2)) / math.sqrt(head_dim)
    if causal:
        query_position = key_length - query_length + torch.arange(query_length, device=query.device)
        key_position = torch.arange(key_length, device=query.device)
        scores = scores.masked_fill(key_position[None, :] > query_position[:, None], -torch.inf)
    if candidate:
        chunks = key_length // chunk_size
        pieces = []
        for block in range(chunks):
            start, stop = block * chunk_size, (block + 1) * chunk_size
            local = scores[..., start:stop]
            maximum = local.max(dim=-1).values
            finite = torch.isfinite(maximum)
            safe_maximum = torch.where(finite, maximum, torch.zeros_like(maximum))
            offset = coefficients[..., None] * safe_maximum
            pieces.append(local + offset[..., None])
        scores = torch.cat(pieces, dim=-1)
    lse = torch.logsumexp(scores, dim=-1)
    probability = torch.softmax(scores, dim=-1)
    output = torch.matmul(probability, value)
    return output.to(query.dtype), lse


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


def smoke() -> dict[str, Any]:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    records = {}
    for name, batch, heads, kv_heads, query_length, key_length, causal in (
        ("decode", 2, 8, 2, 1, 256, False),
        ("prefill", 1, 8, 2, 128, 128, True),
    ):
        query = torch.randn(batch, heads, query_length, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        key = torch.randn(batch, kv_heads, key_length, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        value = torch.randn_like(key)
        coefficients = torch.full((batch, heads), -0.3, device="cuda", dtype=torch.float32)
        cell = {}
        for candidate in (False, True):
            actual, actual_lse = peak_attention(
                query, key, value, coefficients, candidate=candidate, causal=causal
            )
            expected, expected_lse = reference_attention(
                query, key, value, coefficients, candidate=candidate, causal=causal
            )
            cell["candidate" if candidate else "ordinary"] = {
                "output": error_record(actual, expected),
                "lse": error_record(actual_lse, expected_lse),
            }
        records[name] = cell
    return {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "triton": triton.__version__, "triton_file": str(Path(triton.__file__).resolve()),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.smoke:
        print(json.dumps(smoke(), indent=2, sort_keys=True))
        return
    if not PREREGISTRATION.exists():
        raise RuntimeError("formal mode is locked until preregistration exists")
    raise NotImplementedError("formal benchmark not frozen yet")


if __name__ == "__main__":
    main()
