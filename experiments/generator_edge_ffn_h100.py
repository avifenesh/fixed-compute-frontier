#!/usr/bin/env python3
"""Unfused whole-model H100 gate for exact-budget generator-edge FFNs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from experiments.coalesced_attention_ffn_lm_screen import build_model
from experiments.generator_edge_ffn_gate import edge_indices
from experiments.triangular_microdepth_lm_screen import (
    BASELINE_INTERMEDIATE_SIZE,
    HIDDEN_SIZE,
)


ARMS = ("parallel_swiglu", "parallel_self_product", "parallel_duplicate_edge", "parallel_generator_edge")
CELLS = ((1, 1), (1, 512), (8, 512), (32, 512))
PREREGISTRATION = Path("results/generator-edge-ffn-h100-preregistration.md")
STAGE0_SOURCE = Path("experiments/generator_edge_ffn_gate.py")
STAGE0_RESULT = Path("results/generator-edge-ffn-stage0.json")
TEST_SOURCE = Path("tests/test_generator_edge_ffn_h100.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SelfProductMLP(nn.Module):
    WIDTH = 3 * BASELINE_INTERMEDIATE_SIZE // 2

    def __init__(self, initializer_range: float) -> None:
        super().__init__()
        self.generator_proj = nn.Linear(HIDDEN_SIZE, self.WIDTH, bias=False)
        self.down_proj = nn.Linear(self.WIDTH, HIDDEN_SIZE, bias=False)
        for projection in (self.generator_proj, self.down_proj):
            nn.init.normal_(projection.weight, mean=0.0, std=initializer_range)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        generated = self.generator_proj(hidden_states)
        return self.down_proj(F.silu(generated) * generated)


class GeneratorEdgeMLP(nn.Module):
    def __init__(self, initializer_range: float, *, duplicate: bool = False) -> None:
        super().__init__()
        self.duplicate = duplicate
        self.generator_proj = nn.Linear(HIDDEN_SIZE, BASELINE_INTERMEDIATE_SIZE, bias=False)
        self.down_proj = nn.Linear(2 * BASELINE_INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False)
        for projection in (self.generator_proj, self.down_proj):
            nn.init.normal_(projection.weight, mean=0.0, std=initializer_range)
        first, second = edge_indices(BASELINE_INTERMEDIATE_SIZE, duplicate=duplicate)
        self.register_buffer("first", first, persistent=False)
        self.register_buffer("second", second, persistent=False)

    def features(self, hidden_states: torch.Tensor) -> torch.Tensor:
        generated = self.generator_proj(hidden_states)
        activated = F.silu(generated)
        return torch.cat(
            (activated * generated[..., self.first], activated * generated[..., self.second]),
            dim=-1,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.features(hidden_states))


def build_arm(device: torch.device, arm: str) -> nn.Module:
    if arm not in ARMS:
        raise ValueError(f"invalid arm: {arm}")
    torch.manual_seed(1907)
    torch.cuda.manual_seed_all(1907)
    model, _ = build_model(device, "parallel_baseline")
    if arm == "parallel_swiglu":
        return model.eval()
    for layer in model.model.layers:
        if arm == "parallel_self_product":
            replacement = SelfProductMLP(model.config.initializer_range)
        else:
            replacement = GeneratorEdgeMLP(
                model.config.initializer_range,
                duplicate=arm == "parallel_duplicate_edge",
            )
        layer.mlp = replacement.to(device)
    return model.eval()


def percentile(samples: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(samples, dtype=np.float64), q))


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if args.arms != ",".join(ARMS) or args.cells != "1x1,1x512,8x512,32x512":
        raise ValueError("frozen arms/cells changed")
    if args.warmups != 5 or args.repetitions != 30 or args.seed != 1907:
        raise ValueError("frozen timing protocol changed")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    models = {arm: build_arm(device, arm) for arm in ARMS}
    counts = {arm: sum(parameter.numel() for parameter in model.parameters()) for arm, model in models.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"parameter mismatch: {counts}")
    results: dict[str, object] = {}
    peak_bytes: dict[str, int] = {}
    randomizer = random.Random(args.seed)
    for batch, tokens in CELLS:
        inputs = torch.randint(0, 49152, (batch, tokens), device=device)
        for arm, model in models.items():
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(args.warmups):
                    model(input_ids=inputs, use_cache=False)
            torch.cuda.synchronize()
        samples = {arm: [] for arm in ARMS}
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.repetitions):
            order = list(ARMS)
            randomizer.shuffle(order)
            for arm in order:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    started = time.perf_counter()
                    models[arm](input_ids=inputs, use_cache=False)
                    torch.cuda.synchronize()
                samples[arm].append((time.perf_counter() - started) * 1000.0)
        cell_key = f"{batch}x{tokens}"
        cell_results = {}
        baseline_median = float(np.median(samples["parallel_swiglu"]))
        for arm in ARMS:
            median = float(np.median(samples[arm]))
            cell_results[arm] = {
                "median_ms": median,
                "p10_ms": percentile(samples[arm], 0.10),
                "p90_ms": percentile(samples[arm], 0.90),
                "min_ms": min(samples[arm]),
                "ratio_to_parallel_swiglu": median / baseline_median,
                "samples_ms": samples[arm],
            }
        results[cell_key] = cell_results
        peak_bytes[cell_key] = torch.cuda.max_memory_allocated()
    candidate_passes = {
        cell: values["parallel_generator_edge"]["ratio_to_parallel_swiglu"] <= 1.05
        for cell, values in results.items()
    }
    return {
        "schema": "generator-edge-ffn-h100-v1",
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "parameter_counts": counts,
        "protocol": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "hashes": {
            "source": sha256_file(Path(__file__)),
            "preregistration": sha256_file(PREREGISTRATION),
            "stage0_source": sha256_file(STAGE0_SOURCE),
            "stage0_result": sha256_file(STAGE0_RESULT),
            "test": sha256_file(TEST_SOURCE),
        },
        "results": results,
        "peak_allocated_bytes": peak_bytes,
        "candidate_cell_passes": candidate_passes,
        "h100_feasibility_pass": all(candidate_passes.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--cells", default="1x1,1x512,8x512,32x512")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1907)
    parser.add_argument("--output", type=Path, default=Path("results/generator-edge-ffn-h100.json"))
    args = parser.parse_args()
    payload = benchmark(args)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "candidate_cell_passes": payload["candidate_cell_passes"],
        "h100_feasibility_pass": payload["h100_feasibility_pass"],
        "medians": {
            cell: {arm: values[arm]["median_ms"] for arm in ARMS}
            for cell, values in payload["results"].items()
        },
        "ratios": {
            cell: {arm: values[arm]["ratio_to_parallel_swiglu"] for arm in ARMS}
            for cell, values in payload["results"].items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
