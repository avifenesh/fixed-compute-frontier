#!/usr/bin/env python3
"""Normal-GEMM H100 latency gate for zero-pivot G1 attention."""

from __future__ import annotations

import argparse
import hashlib
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
HEAD_DIM = 128
PAIRS = HEAD_DIM // 2
Q_DIM = QUERY_HEADS * HEAD_DIM
K_DIM = KV_HEADS * HEAD_DIM
V_DIM = K_DIM
QKV_DIM = Q_DIM + K_DIM + V_DIM
TAU = 0.125
OUTPUT = Path("results/gauge-zero-g1-h100-development.json")
BOOTSTRAP_REPLICATES = 5000
PREREGISTRATION = Path("results/gauge-zero-g1-h100-preregistration.md")
TEST_SOURCE = Path("tests/test_gauge_zero_g1_h100.py")
ATTENTION_SOURCE = Path("experiments/gauge_zero_g1_attention.py")
ATTENTION_TEST = Path("tests/test_gauge_zero_g1_attention.py")
INTEGRITY_MANIFEST = Path("results/gauge-zero-g1-h100-integrity-manifest.json")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_integrity() -> dict[str, bool]:
    manifest = json.loads(INTEGRITY_MANIFEST.read_text())
    paths = {
        "source": Path(__file__),
        "test": TEST_SOURCE,
        "preregistration": PREREGISTRATION,
        "attention_source": ATTENTION_SOURCE,
        "attention_test": ATTENTION_TEST,
    }
    checks = {
        name: manifest.get(f"{name}_sha256") == sha256_file(path)
        for name, path in paths.items()
    }
    checks["torch_version"] = manifest.get("torch_version") == torch.__version__
    checks["triton_version"] = manifest.get("triton_version") == triton.__version__
    checks["cuda_version"] = manifest.get("cuda_version") == torch.version.cuda
    if not all(checks.values()):
        raise ValueError(f"invalid gauge-zero G1 integrity: {checks}")
    return checks


@triton.jit
def rope_zero_g1_kernel(
    qkv_ptr,
    weight_ptr,
    cosine_ptr,
    sine_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    WEIGHT_K: tl.constexpr,
    Q_HEADS: tl.constexpr,
    HEAD: tl.constexpr,
    HALF: tl.constexpr,
    Q_WIDTH: tl.constexpr,
    TAU_VALUE: tl.constexpr,
    G1: tl.constexpr,
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
    even = tl.load(even_ptr).to(tl.float32)
    odd = tl.load(odd_ptr).to(tl.float32)

    if G1:
        is_key = ~is_query
        pivot = pair
        coefficient_row = Q_WIDTH + projected_head * HEAD + HALF + pair
        coefficient = tl.load(
            weight_ptr + coefficient_row * WEIGHT_K + pivot,
            mask=is_key,
            other=0.0,
        ).to(tl.float32) / TAU_VALUE
        odd = tl.where(is_key, odd + coefficient * even * even, odd)

    cosine = tl.load(cosine_ptr + token * HALF + pair).to(tl.float32)
    sine = tl.load(sine_ptr + token * HALF + pair).to(tl.float32)
    tl.store(even_ptr, cosine * even - sine * odd)
    tl.store(odd_ptr, sine * even + cosine * odd)


def rope_inplace(
    output: torch.Tensor,
    weight: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    *,
    g1: bool,
    hidden: int = HIDDEN,
    query_heads: int = QUERY_HEADS,
    kv_heads: int = KV_HEADS,
    head_dim: int = HEAD_DIM,
    tau: float = TAU,
) -> None:
    pairs = head_dim // 2
    q_width = query_heads * head_dim
    rope_zero_g1_kernel[(output.shape[0], query_heads + kv_heads)](
        output,
        weight,
        cosine,
        sine,
        M=output.shape[0],
        N=output.shape[1],
        WEIGHT_K=hidden,
        Q_HEADS=query_heads,
        HEAD=head_dim,
        HALF=pairs,
        Q_WIDTH=q_width,
        TAU_VALUE=tau,
        G1=g1,
        num_warps=2,
    )


def make_weight(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    weight = torch.randn(
        QKV_DIM, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=generator
    ) / math.sqrt(HIDDEN)
    coefficients = 0.02 * torch.randn(
        KV_HEADS, PAIRS, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    for kv_head in range(KV_HEADS):
        for pair in range(PAIRS):
            column = pair
            even_row = Q_DIM + kv_head * HEAD_DIM + pair
            odd_row = even_row + PAIRS
            weight[even_row, column] = weight[even_row, column].abs()
            weight[odd_row, column] = coefficients[kv_head, pair]
    return weight.contiguous()


def reference_candidate(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    output = inputs @ weight.T
    query = output[:, :Q_DIM].view(-1, QUERY_HEADS, HEAD_DIM)
    key = output[:, Q_DIM:Q_DIM + K_DIM].view(-1, KV_HEADS, HEAD_DIM)
    query_even = query[..., :PAIRS].float()
    query_odd = query[..., PAIRS:].float()
    key_even = key[..., :PAIRS].float()
    key_odd = key[..., PAIRS:].float()
    row_base = Q_DIM + torch.arange(KV_HEADS, device="cuda")[:, None] * HEAD_DIM
    coefficient_rows = row_base + PAIRS + torch.arange(PAIRS, device="cuda")[None]
    columns = torch.arange(PAIRS, device="cuda")[None].expand(KV_HEADS, -1)
    coefficient = (weight[coefficient_rows, columns].float() / TAU)[None]
    key_odd = key_odd + coefficient * key_even.square()
    cos = cosine[:, None].float()
    sin = sine[:, None].float()
    query[..., :PAIRS] = (cos * query_even - sin * query_odd).to(query.dtype)
    query[..., PAIRS:] = (sin * query_even + cos * query_odd).to(query.dtype)
    key[..., :PAIRS] = (cos * key_even - sin * key_odd).to(key.dtype)
    key[..., PAIRS:] = (sin * key_even + cos * key_odd).to(key.dtype)
    return output


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(samples) * 1000.0,
        "p05_us": ordered[int(0.05 * (len(ordered) - 1))] * 1000.0,
        "p95_us": ordered[int(0.95 * (len(ordered) - 1))] * 1000.0,
    }


def bootstrap_median_ratio(
    candidate: list[float], control: list[float], seed: int
) -> dict[str, float]:
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(control, dtype=np.float64)
    if left.shape != right.shape or left.size < 2:
        raise ValueError("bootstrap requires matched sample counts")
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0, left.size, size=(BOOTSTRAP_REPLICATES, left.size)
    )
    ratio = np.median(left[indices], axis=1) / np.median(right[indices], axis=1)
    return {
        "median_ratio": float(np.median(left) / np.median(right)),
        "lower_95": float(np.quantile(ratio, 0.025)),
        "upper_95": float(np.quantile(ratio, 0.975)),
    }


def benchmark_cell(rows: int, trials: int, warmup: int, seed: int) -> dict[str, object]:
    generator = torch.Generator(device="cuda").manual_seed(seed + rows)
    inputs = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=generator)
    weight = make_weight(seed)
    angles = torch.randn(rows, PAIRS, device="cuda", dtype=torch.float32, generator=generator)
    cosine = angles.cos().to(torch.bfloat16)
    sine = angles.sin().to(torch.bfloat16)
    outputs = {
        arm: torch.empty(rows, QKV_DIM, device="cuda", dtype=torch.bfloat16)
        for arm in ("canonical_control", "gauge_zero_g1")
    }
    l2_flush = torch.zeros(64 * 1024 * 1024, device="cuda", dtype=torch.bfloat16)

    def run_arm(arm: str) -> None:
        torch.mm(inputs, weight.T, out=outputs[arm])
        rope_inplace(
            outputs[arm], weight, cosine, sine, g1=(arm == "gauge_zero_g1")
        )

    for _ in range(warmup):
        run_arm("canonical_control")
        run_arm("gauge_zero_g1")
    torch.cuda.synchronize()
    reference = reference_candidate(inputs, weight, cosine, sine)
    control_reference = inputs @ weight.T
    control_query = control_reference[:, :Q_DIM].view(-1, QUERY_HEADS, HEAD_DIM)
    control_key = control_reference[:, Q_DIM:Q_DIM + K_DIM].view(-1, KV_HEADS, HEAD_DIM)
    cos = cosine[:, None].float()
    sin = sine[:, None].float()
    q_even = control_query[..., :PAIRS].float()
    q_odd = control_query[..., PAIRS:].float()
    k_even = control_key[..., :PAIRS].float()
    k_odd = control_key[..., PAIRS:].float()
    control_query[..., :PAIRS] = (cos * q_even - sin * q_odd).to(control_query.dtype)
    control_query[..., PAIRS:] = (sin * q_even + cos * q_odd).to(control_query.dtype)
    control_key[..., :PAIRS] = (cos * k_even - sin * k_odd).to(control_key.dtype)
    control_key[..., PAIRS:] = (sin * k_even + cos * k_odd).to(control_key.dtype)
    run_arm("gauge_zero_g1")
    run_arm("canonical_control")
    torch.cuda.synchronize()
    difference = (outputs["gauge_zero_g1"].float() - reference.float()).abs()
    control_difference = (
        outputs["canonical_control"].float() - control_reference.float()
    ).abs()
    correctness = {
        "candidate_max_abs": float(difference.max()),
        "candidate_mean_abs": float(difference.mean()),
        "candidate_bit_exact_bf16": bool(torch.equal(outputs["gauge_zero_g1"], reference)),
        "control_max_abs": float(control_difference.max()),
        "control_mean_abs": float(control_difference.mean()),
        "control_bit_exact_bf16": bool(torch.equal(outputs["canonical_control"], control_reference)),
    }

    rng = random.Random(seed + 10 * rows)
    arms = list(outputs)
    samples: dict[str, list[float]] = {arm: [] for arm in arms}
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
        samples[arm].append(start.elapsed_time(end))
    timing = {
        arm: {**summarize(value), "samples_ms": value}
        for arm, value in samples.items()
    }
    ratio = bootstrap_median_ratio(
        samples["gauge_zero_g1"],
        samples["canonical_control"],
        seed + rows,
    )
    return {
        "rows": rows,
        "correctness": correctness,
        "timing": timing,
        "candidate_over_control": ratio,
        "logical": {
            "resident_weight_bytes_each": weight.numel() * weight.element_size(),
            "output_bytes_each": outputs["canonical_control"].numel() * outputs["canonical_control"].element_size(),
            "pivot_metadata_bits": 0,
            "pivot_rule": "column equals RoPE pair index",
            "dense_qkv_macs_each": rows * HIDDEN * QKV_DIM,
            "candidate_extra_per_key_pair_token": "one square, one coefficient multiply, one add, one coefficient weight load per key-pair lane",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="1,8,32,128,512,2048")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=4093)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    integrity = validate_integrity()
    payload = {
        "schema": "gauge-zero-g1-h100-v1",
        "device": torch.cuda.get_device_name(0),
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": triton.__version__,
        },
        "integrity": integrity,
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
        cells = payload["cells"]
        payload["gates"] = {
            "all_bf16_outputs_bit_exact": all(
                cell["correctness"]["candidate_bit_exact_bf16"]
                and cell["correctness"]["control_bit_exact_bf16"]
                for cell in cells
            ),
            "all_median_ratios_at_most_1_02": all(
                cell["candidate_over_control"]["median_ratio"] <= 1.02
                for cell in cells
            ),
            "all_upper_95_ratios_at_most_1_03": all(
                cell["candidate_over_control"]["upper_95"] <= 1.03
                for cell in cells
            ),
            "decode_1_and_8_upper_95_at_most_1_02": all(
                cell["candidate_over_control"]["upper_95"] <= 1.02
                for cell in cells
                if cell["rows"] in (1, 8)
            ),
            "resident_weight_bytes_equal_by_shared_dense_tensor": all(
                cell["logical"]["resident_weight_bytes_each"] == 50331648
                for cell in cells
            ),
            "deterministic_pivot_requires_zero_metadata_bits": all(
                cell["logical"]["pivot_metadata_bits"] == 0 for cell in cells
            ),
        }
        payload["all_gates_pass"] = (
            len(cells) == len([value for value in args.rows.split(",") if value])
            and all(payload["gates"].values())
        )
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps({
            "rows": rows,
            "correctness": payload["cells"][-1]["correctness"],
            "ratio": payload["cells"][-1]["candidate_over_control"],
        }), flush=True)


if __name__ == "__main__":
    main()
