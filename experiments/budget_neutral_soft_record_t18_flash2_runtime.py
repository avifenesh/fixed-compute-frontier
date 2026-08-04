#!/usr/bin/env python3
"""T18-Flash2 exact successor runtime gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import budget_neutral_soft_record_t18 as t18
from experiments import budget_neutral_soft_record_t18_flash2 as flash2
from experiments import budget_neutral_soft_record_t18_runtime as runtime
from experiments import raw_prose_equality_plane_t10 as t10


PREREGISTRATION = ROOT / "results/budget-neutral-soft-record-t18-flash2-preregistration.md"
RUNTIME_V1 = ROOT / "results/budget-neutral-soft-record-t18-runtime.json"
OUTPUT = ROOT / "results/budget-neutral-soft-record-t18-flash2-runtime.json"
RUNTIME_V1_SHA256 = "099e2742108f66fb5dfd54a53e95a9dd09ee9c13d287f66d4f13be6f723a9bc9"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Flash2CandidatePair(runtime.CandidateRecordPair):
    def __init__(self) -> None:
        super().__init__()
        self.memory = flash2.Flash2RecordMemory()


def flash_backend_contract(device: torch.device) -> dict[str, object]:
    query = torch.randn(2, flash2.HEADS, 8, flash2.HEAD_DIM, device=device, dtype=torch.bfloat16)
    key_base = torch.randn(
        1, flash2.HEADS, t18.RECORD_SLOTS, flash2.HEAD_DIM,
        device=device, dtype=torch.bfloat16
    )
    value_base = torch.randn_like(key_base)
    key = key_base.expand(2, -1, -1, -1)
    value = value_base.expand(2, -1, -1, -1)
    parameters = torch.nn.attention.SDPAParams(
        query, key, value, None, 0.0, False, False
    )
    eligible = torch.backends.cuda.can_use_flash_attention(parameters, debug=True)
    same_storage = (
        key.untyped_storage().data_ptr() == key_base.untyped_storage().data_ptr()
        and value.untyped_storage().data_ptr()
        == value_base.untyped_storage().data_ptr()
    )
    with torch.nn.attention.sdpa_kernel(
        torch.nn.attention.SDPBackend.FLASH_ATTENTION
    ):
        output = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    torch.cuda.synchronize(device)
    return {
        "eligible": bool(eligible),
        "forced_execution_finite": bool(torch.isfinite(output).all().item()),
        "batch_expand_is_storage_view": same_storage,
        "key_stride": list(key.stride()),
        "value_stride": list(value.stride()),
    }


def run(device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        raise RuntimeError("Flash2 runtime gate requires CUDA")
    device_name = torch.cuda.get_device_name(device)
    if "H100" not in device_name and "H200" not in device_name:
        raise RuntimeError(f"Flash2 requires H100/H200, found {device_name}")
    if sha256_file(RUNTIME_V1) != RUNTIME_V1_SHA256:
        raise RuntimeError("T18 runtime-v1 result hash mismatch")

    backend = flash_backend_contract(device)
    cpu_reference = flash2.bounded_reference(torch.device("cpu"))
    maximum_reference_error = max(cpu_reference["maximum_errors"].values())
    torch.manual_seed(runtime.SEED)
    torch.cuda.manual_seed_all(runtime.SEED)
    block_modules = {
        "baseline": runtime.BaselineFFNPair().to(device).eval(),
        "candidate": Flash2CandidatePair().to(device).eval(),
    }
    model_modules = {
        "baseline": t10.core.build_model(runtime.SEED, device).eval(),
        "candidate": flash2.build_candidate(runtime.SEED, device).eval(),
    }
    block_parameters = {
        name: runtime.module_parameters(module)
        for name, module in block_modules.items()
    }
    model_parameters = {
        name: runtime.module_parameters(module)
        for name, module in model_modules.items()
    }
    generator = torch.Generator(device=device).manual_seed(runtime.SEED + 1)
    block_results: dict[str, object] = {}
    model_results: dict[str, object] = {}
    for batch, tokens in runtime.BLOCK_SURFACES:
        inputs = torch.randn(
            batch, tokens, t18.HIDDEN, generator=generator, device=device
        )
        iterations = 200 if batch * tokens <= 128 else 50
        block_results[f"b{batch}_t{tokens}"] = runtime.benchmark_surface(
            block_modules, inputs, iterations, device
        )
    for batch, tokens in runtime.MODEL_SURFACES:
        inputs = torch.randint(
            0,
            t10.core.small.VOCAB,
            (batch, tokens),
            generator=generator,
            device=device,
        )
        iterations = 50 if batch * tokens <= 128 else 10
        model_results[f"b{batch}_t{tokens}"] = runtime.benchmark_surface(
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
        "flash_backend_eligible_and_forced": backend["eligible"]
        and backend["forced_execution_finite"],
        "expanded_kv_are_views": backend["batch_expand_is_storage_view"],
        "dense_reference_within_tolerance": maximum_reference_error <= 2e-6,
        "dominant_null_suppresses_output": cpu_reference["null_output_max_abs"]
        < 1e-5,
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
        "schema": "budget-neutral-soft-record-t18-flash2-runtime-v1",
        "status": "pass" if all(gates.values()) else "fail",
        "device": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "operator": {
            "heads": flash2.HEADS,
            "head_dimension": flash2.HEAD_DIM,
            "record_slots": t18.RECORD_SLOTS,
            "softmax_exponentials_per_token": flash2.HEADS * t18.RECORD_SLOTS,
            "matrix_macs_per_token": t18.CANDIDATE_REPLACED_SCALARS,
        },
        "backend_contract": backend,
        "cpu_bounded_reference": cpu_reference,
        "protocol": {
            "seed": runtime.SEED,
            "trials": runtime.TRIALS,
            "warmups_per_trial": runtime.WARMUPS,
            "block_surfaces": runtime.BLOCK_SURFACES,
            "model_surfaces": runtime.MODEL_SURFACES,
            "forced_backend": "FLASH_ATTENTION",
            "cuda_graph": False,
            "torch_compile": False,
            "custom_kernel": False,
        },
        "parameter_counts": {
            "block_pair": block_parameters,
            "whole_model": model_parameters,
        },
        "block_pair": block_results,
        "whole_model": model_results,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "integrity": {
            "source_sha256": sha256_file(Path(__file__)),
            "operator_source_sha256": sha256_file(
                ROOT / "experiments/budget_neutral_soft_record_t18_flash2.py"
            ),
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "runtime_v1_result_sha256": sha256_file(RUNTIME_V1),
            "stage0_result_sha256": sha256_file(runtime.STAGE0),
        },
        "claim_boundary": (
            "Passing admits training design only. It does not establish record "
            "usefulness, language preservation, QA gain, or a Pareto result."
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
