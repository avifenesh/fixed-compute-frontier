#!/usr/bin/env python3
"""T18 H100 runtime gate for equal-MAC dense FFN and soft-record paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import budget_neutral_soft_record_t18 as t18
from experiments import raw_prose_equality_plane_t10 as t10


PREREGISTRATION = ROOT / "results/budget-neutral-soft-record-t18-runtime-preregistration.md"
STAGE0 = ROOT / "results/budget-neutral-soft-record-t18-stage0.json"
OUTPUT = ROOT / "results/budget-neutral-soft-record-t18-runtime.json"

STAGE0_SHA256 = "ba2ac04c370a4c000b513438ac30ce5b5fccd0d6856ba98861198ea83db47126"
STAGE0_SOURCE_SHA256 = "25c01d95905ffb6026cefa855d9a882f4deecd76a9ea63e2f9e9ef0e1908c628"
SEED = 10_109
TRIALS = 5
WARMUPS = 20
BLOCK_SURFACES = ((1, 1), (1, 128), (8, 128), (32, 128))
MODEL_SURFACES = ((1, 1), (1, 128), (8, 128))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BaselineFFNPair(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        small = t10.core.small
        self.norms = nn.ModuleList([small.RMSNorm(), small.RMSNorm()])
        self.gates = nn.ModuleList(
            [nn.Linear(t18.HIDDEN, t18.BASELINE_FFN, bias=False) for _ in range(2)]
        )
        self.ups = nn.ModuleList(
            [nn.Linear(t18.HIDDEN, t18.BASELINE_FFN, bias=False) for _ in range(2)]
        )
        self.downs = nn.ModuleList(
            [nn.Linear(t18.BASELINE_FFN, t18.HIDDEN, bias=False) for _ in range(2)]
        )

    def forward(self, hidden: Tensor) -> Tensor:
        for norm, gate, up, down in zip(
            self.norms, self.gates, self.ups, self.downs, strict=True
        ):
            normalized = norm(hidden)
            hidden = hidden + down(F.silu(gate(normalized)) * up(normalized))
        return hidden


class CandidateRecordPair(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        small = t10.core.small
        self.memory_norm = small.RMSNorm()
        self.memory = t18.DenseRecordMemory()
        self.small_norm = small.RMSNorm()
        self.gate = nn.Linear(t18.HIDDEN, t18.SMALL_FFN, bias=False)
        self.up = nn.Linear(t18.HIDDEN, t18.SMALL_FFN, bias=False)
        self.down = nn.Linear(t18.SMALL_FFN, t18.HIDDEN, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.memory(self.memory_norm(hidden))
        normalized = self.small_norm(hidden)
        return hidden + self.down(F.silu(self.gate(normalized)) * self.up(normalized))


def module_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


@torch.inference_mode()
def benchmark_trial(
    module: nn.Module,
    inputs: Tensor,
    iterations: int,
    device: torch.device,
) -> float:
    for _ in range(WARMUPS):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = module(inputs)
    del output
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = module(inputs)
    end.record()
    end.synchronize()
    elapsed = start.elapsed_time(end) / iterations
    del output
    return float(elapsed)


@torch.inference_mode()
def incremental_peak_bytes(
    module: nn.Module, inputs: Tensor, device: torch.device
) -> int:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = module(inputs)
    del output
    torch.cuda.synchronize(device)
    before = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = module(inputs)
    torch.cuda.synchronize(device)
    peak = int(torch.cuda.max_memory_allocated(device))
    del output
    torch.cuda.synchronize(device)
    return max(peak - before, 0)


def benchmark_surface(
    modules: dict[str, nn.Module],
    inputs: Tensor,
    iterations: int,
    device: torch.device,
) -> dict[str, object]:
    trials = {name: [] for name in modules}
    names = tuple(modules)
    for trial in range(TRIALS):
        order = names if trial % 2 == 0 else tuple(reversed(names))
        for name in order:
            trials[name].append(
                benchmark_trial(modules[name], inputs, iterations, device)
            )
    medians = {name: statistics.median(values) for name, values in trials.items()}
    peaks = {
        name: incremental_peak_bytes(module, inputs, device)
        for name, module in modules.items()
    }
    return {
        "iterations_per_trial": iterations,
        "trials_ms": trials,
        "median_ms": medians,
        "candidate_over_baseline": medians["candidate"] / medians["baseline"],
        "incremental_peak_allocated_bytes": peaks,
        "candidate_over_baseline_peak": (
            peaks["candidate"] / peaks["baseline"]
            if peaks["baseline"]
            else float("inf")
        ),
    }


def run(device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        raise RuntimeError("runtime gate requires CUDA")
    device_name = torch.cuda.get_device_name(device)
    if "H100" not in device_name and "H200" not in device_name:
        raise RuntimeError(f"runtime gate requires H100/H200, found {device_name}")
    if sha256_file(STAGE0) != STAGE0_SHA256:
        raise RuntimeError("T18 Stage-0 result hash mismatch")
    if sha256_file(ROOT / "experiments/budget_neutral_soft_record_t18.py") != STAGE0_SOURCE_SHA256:
        raise RuntimeError("T18 Stage-0 source hash mismatch")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    block_modules = {
        "baseline": BaselineFFNPair().to(device).eval(),
        "candidate": CandidateRecordPair().to(device).eval(),
    }
    model_modules = {
        "baseline": t10.core.build_model(SEED, device).eval(),
        "candidate": t18.build_candidate(SEED, device).eval(),
    }
    block_parameters = {
        name: module_parameters(module) for name, module in block_modules.items()
    }
    model_parameters = {
        name: module_parameters(module) for name, module in model_modules.items()
    }

    block_results: dict[str, object] = {}
    model_results: dict[str, object] = {}
    generator = torch.Generator(device=device).manual_seed(SEED + 1)
    for batch, tokens in BLOCK_SURFACES:
        inputs = torch.randn(
            batch, tokens, t18.HIDDEN, generator=generator, device=device
        )
        iterations = 200 if batch * tokens <= 128 else 50
        block_results[f"b{batch}_t{tokens}"] = benchmark_surface(
            block_modules, inputs, iterations, device
        )
    for batch, tokens in MODEL_SURFACES:
        inputs = torch.randint(
            0,
            t10.core.small.VOCAB,
            (batch, tokens),
            generator=generator,
            device=device,
        )
        iterations = 50 if batch * tokens <= 128 else 10
        model_results[f"b{batch}_t{tokens}"] = benchmark_surface(
            model_modules, inputs, iterations, device
        )

    all_results = (*block_results.values(), *model_results.values())
    finite = all(
        math.isfinite(result["candidate_over_baseline"])
        and math.isfinite(result["candidate_over_baseline_peak"])
        for result in all_results
    )
    gates = {
        "environment_h100_or_h200": True,
        "finite_measurements": finite,
        "block_parameter_count_exact": block_parameters["baseline"]
        == block_parameters["candidate"],
        "model_parameter_count_exact": model_parameters["baseline"]
        == model_parameters["candidate"]
        == t10.core.small.parameter_count(),
        "matrix_mac_ledger_exact": t18.BASELINE_REPLACED_SCALARS
        == t18.CANDIDATE_REPLACED_SCALARS,
        "block_latency_noninferior": all(
            result["candidate_over_baseline"] <= 1.05
            for result in block_results.values()
        ),
        "model_latency_noninferior": all(
            result["candidate_over_baseline"] <= 1.05
            for result in model_results.values()
        ),
        "block_peak_allocation_noninferior": all(
            result["candidate_over_baseline_peak"] <= 1.05
            for result in block_results.values()
        ),
        "model_peak_allocation_noninferior": all(
            result["candidate_over_baseline_peak"] <= 1.05
            for result in model_results.values()
        ),
    }
    return {
        "schema": "budget-neutral-soft-record-t18-runtime-v1",
        "status": "pass" if all(gates.values()) else "fail",
        "device": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "protocol": {
            "seed": SEED,
            "trials": TRIALS,
            "warmups_per_trial": WARMUPS,
            "precision": "bfloat16_autocast_fp32_parameters",
            "block_surfaces": BLOCK_SURFACES,
            "model_surfaces": MODEL_SURFACES,
            "cuda_graph": False,
            "torch_compile": False,
            "custom_kernel": False,
        },
        "parameter_counts": {
            "block_pair": block_parameters,
            "whole_model": model_parameters,
        },
        "matrix_macs_per_token": {
            "baseline": t18.BASELINE_REPLACED_SCALARS,
            "candidate": t18.CANDIDATE_REPLACED_SCALARS,
        },
        "block_pair": block_results,
        "whole_model": model_results,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "integrity": {
            "source_sha256": sha256_file(Path(__file__)),
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "stage0_result_sha256": sha256_file(STAGE0),
            "stage0_source_sha256": sha256_file(
                ROOT / "experiments/budget_neutral_soft_record_t18.py"
            ),
        },
        "claim_boundary": (
            "This is an unfused eager H100 runtime gate. Passing does not prove "
            "capability; failure requires a fused operator before training."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    result = run(torch.device(arguments.device))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "all_gates_pass": result["all_gates_pass"],
                "failed_gates": sorted(
                    name for name, passed in result["gates"].items() if not passed
                ),
                "block_ratios": {
                    name: value["candidate_over_baseline"]
                    for name, value in result["block_pair"].items()
                },
                "model_ratios": {
                    name: value["candidate_over_baseline"]
                    for name, value in result["whole_model"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
