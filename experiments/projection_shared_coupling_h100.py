#!/usr/bin/env python3
"""H100 stage-0 gate for a projection-shared nonlinear coupling chart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
from typing import Any

import torch
import triton
import triton.language as tl


HIDDEN = 4096
ROWS = (1, 8, 64, 256, 1024)
WIDTHS = (6144, 14336)
OUTPUT = Path("results/projection-shared-coupling-h100-development.json")


@triton.jit
def rms_gain_kernel(
    x_ptr,
    gain_ptr,
    output_ptr,
    stride: tl.constexpr,
    epsilon: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * stride + offsets).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(x * x, axis=0) / stride + epsilon)
    gain = tl.load(gain_ptr + offsets).to(tl.float32)
    tl.store(output_ptr + row * stride + offsets, x * inverse_rms * gain)


@triton.jit
def rms_coupling_kernel(
    x_ptr,
    alpha_ptr,
    beta_ptr,
    output_ptr,
    stride: tl.constexpr,
    half: tl.constexpr,
    epsilon: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    pair_offsets = tl.arange(0, BLOCK // 2)
    a_raw = tl.load(x_ptr + row * stride + pair_offsets).to(tl.float32)
    b_raw = tl.load(x_ptr + row * stride + pair_offsets + half).to(tl.float32)
    squared_sum = tl.sum(a_raw * a_raw, axis=0) + tl.sum(b_raw * b_raw, axis=0)
    inverse_rms = tl.rsqrt(squared_sum / stride + epsilon)
    a = a_raw * inverse_rms
    b = b_raw * inverse_rms
    alpha = tl.load(alpha_ptr + pair_offsets).to(tl.float32)
    beta = tl.load(beta_ptr + pair_offsets).to(tl.float32)
    b_prime = b + alpha * tl.abs(a)
    a_prime = a + beta * tl.abs(b_prime)
    tl.store(output_ptr + row * stride + pair_offsets, a_prime)
    tl.store(output_ptr + row * stride + pair_offsets + half, b_prime)


def metadata(compiled: Any) -> dict[str, Any]:
    details = getattr(compiled, "metadata", None)
    return {
        "registers": getattr(compiled, "n_regs", getattr(details, "num_regs", None)),
        "spills": getattr(compiled, "n_spills", getattr(details, "num_spills", None)),
        "shared_bytes": getattr(compiled, "shared", getattr(details, "shared", None)),
    }


def launch_baseline(x, gain, output):
    return rms_gain_kernel[(x.shape[0],)](
        x,
        gain,
        output,
        HIDDEN,
        1e-6,
        BLOCK=HIDDEN,
        num_warps=8,
    )


def launch_candidate(x, alpha, beta, output):
    return rms_coupling_kernel[(x.shape[0],)](
        x,
        alpha,
        beta,
        output,
        HIDDEN,
        HIDDEN // 2,
        1e-6,
        BLOCK=HIDDEN,
        num_warps=8,
    )


def capture_graph(callable_):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        callable_()
    return graph


def elapsed_ms(graph: torch.cuda.CUDAGraph) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def run(timed_iterations: int, warmup_iterations: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("This frozen screen requires SM90")

    torch.manual_seed(20260728)
    random.seed(20260728)
    device = torch.device("cuda")
    gain = (0.75 + 0.5 * torch.rand(HIDDEN, device=device)).to(torch.bfloat16)
    alpha = (0.25 * torch.randn(HIDDEN // 2, device=device)).to(torch.bfloat16)
    beta = (0.25 * torch.randn(HIDDEN // 2, device=device)).to(torch.bfloat16)

    records = []
    kernel_metadata = None
    correctness = []
    for rows in ROWS:
        x = torch.randn((rows, HIDDEN), device=device, dtype=torch.bfloat16)
        baseline_norm = torch.empty_like(x)
        candidate_norm = torch.empty_like(x)
        compiled_baseline = launch_baseline(x, gain, baseline_norm)
        compiled_candidate = launch_candidate(x, alpha, beta, candidate_norm)
        torch.cuda.synchronize()

        x_float = x.float()
        inverse_rms = torch.rsqrt(torch.mean(x_float * x_float, dim=1, keepdim=True) + 1e-6)
        normalized = x_float * inverse_rms
        expected_baseline = normalized * gain.float()
        a, b = normalized.chunk(2, dim=1)
        expected_b = b + alpha.float() * torch.abs(a)
        expected_a = a + beta.float() * torch.abs(expected_b)
        expected_candidate = torch.cat((expected_a, expected_b), dim=1)
        correctness.append(
            {
                "rows": rows,
                "baseline_max_abs_error": float(
                    (baseline_norm.float() - expected_baseline).abs().max().item()
                ),
                "candidate_max_abs_error": float(
                    (candidate_norm.float() - expected_candidate).abs().max().item()
                ),
            }
        )
        if kernel_metadata is None:
            kernel_metadata = {
                "baseline": metadata(compiled_baseline),
                "candidate": metadata(compiled_candidate),
            }

        for width in WIDTHS:
            weight = torch.randn(
                (HIDDEN, width), device=device, dtype=torch.bfloat16
            ) / HIDDEN**0.5
            baseline_projection = torch.empty(
                (rows, width), device=device, dtype=torch.bfloat16
            )
            candidate_projection = torch.empty_like(baseline_projection)

            # Initialize cuBLAS and its workspace before stream capture.
            torch.mm(baseline_norm, weight, out=baseline_projection)
            torch.mm(candidate_norm, weight, out=candidate_projection)
            torch.cuda.synchronize()

            baseline_graph = capture_graph(
                lambda: (
                    launch_baseline(x, gain, baseline_norm),
                    torch.mm(baseline_norm, weight, out=baseline_projection),
                )
            )
            candidate_graph = capture_graph(
                lambda: (
                    launch_candidate(x, alpha, beta, candidate_norm),
                    torch.mm(candidate_norm, weight, out=candidate_projection),
                )
            )
            for _ in range(warmup_iterations):
                baseline_graph.replay()
                candidate_graph.replay()
            torch.cuda.synchronize()

            samples = {"baseline": [], "candidate": []}
            for _ in range(timed_iterations):
                order = ["baseline", "candidate"]
                random.shuffle(order)
                for arm in order:
                    graph = baseline_graph if arm == "baseline" else candidate_graph
                    samples[arm].append(elapsed_ms(graph))

            baseline_median = statistics.median(samples["baseline"])
            candidate_median = statistics.median(samples["candidate"])
            records.append(
                {
                    "rows": rows,
                    "width": width,
                    "baseline_median_ms": baseline_median,
                    "candidate_median_ms": candidate_median,
                    "candidate_over_baseline": candidate_median / baseline_median,
                    "baseline_p10_ms": percentile(samples["baseline"], 0.10),
                    "baseline_p90_ms": percentile(samples["baseline"], 0.90),
                    "candidate_p10_ms": percentile(samples["candidate"], 0.10),
                    "candidate_p90_ms": percentile(samples["candidate"], 0.90),
                }
            )

    key_records = [record for record in records if record["rows"] in (256, 1024)]
    decision = {
        "correct": all(
            item["baseline_max_abs_error"] <= 0.02
            and item["candidate_max_abs_error"] <= 0.02
            for item in correctness
        ),
        "identical_resources": kernel_metadata["baseline"] == kernel_metadata["candidate"],
        "key_cells_at_most_1_01": all(
            record["candidate_over_baseline"] <= 1.01 for record in key_records
        ),
        "all_cells_at_most_1_03": all(
            record["candidate_over_baseline"] <= 1.03 for record in records
        ),
    }
    decision["pass"] = all(decision.values())
    return {
        "schema": "projection-shared-coupling-h100-v1",
        "device": torch.cuda.get_device_name(),
        "shape": {"hidden": HIDDEN, "rows": ROWS, "widths": WIDTHS},
        "iterations": {"warmup": warmup_iterations, "timed": timed_iterations},
        "parameter_slots": {
            "baseline_gain": HIDDEN,
            "candidate_alpha_beta": HIDDEN,
        },
        "kernel_metadata": kernel_metadata,
        "correctness": correctness,
        "records": records,
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timed-iterations", type=int, default=500)
    parser.add_argument("--warmup-iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    payload = run(args.timed_iterations, args.warmup_iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
