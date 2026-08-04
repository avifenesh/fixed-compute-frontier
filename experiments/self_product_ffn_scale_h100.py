#!/usr/bin/env python3
"""Scale-matched BF16 H100 serving and memory gate."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import transformers

from experiments.self_product_ffn_lm_screen import sha256_file
from experiments.self_product_ffn_scale import (
    PREREGISTRATION as LM_PREREGISTRATION,
    TEST_SOURCE as LM_TEST,
    VALID_ARMS,
    build_scale_model,
)


PREFILL_CELLS = ((1, 512), (8, 512), (32, 512))
DECODE_CELLS = ((1, 512), (8, 512))
BOOTSTRAP_REPETITIONS = 5000
PREREGISTRATION = Path("results/self-product-ffn-scale-h100-preregistration.md")
TEST_SOURCE = Path("tests/test_self_product_ffn_scale_h100.py")
LM_SOURCE = Path("experiments/self_product_ffn_scale.py")
BASE_LM_SOURCE = Path("experiments/self_product_ffn_lm_screen.py")
TRIANGULAR_SOURCE = Path("experiments/triangular_microdepth_lm_screen.py")
COALESCED_SOURCE = Path("experiments/coalesced_attention_ffn_lm_screen.py")
REFLEX_SOURCE = Path("experiments/reflex_swiglu_lm_screen.py")


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def bootstrap_median_ratio(
    candidate: list[float], baseline: list[float], seed: int
) -> dict[str, float]:
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(baseline, dtype=np.float64)
    if left.shape != right.shape or left.size < 2:
        raise ValueError("bootstrap needs matched sample counts")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, left.size, size=(BOOTSTRAP_REPETITIONS, left.size))
    ratios = np.median(left[indices], axis=1) / np.median(right[indices], axis=1)
    return {
        "median_ratio": float(np.median(left) / np.median(right)),
        "lower_95": float(np.quantile(ratios, 0.025)),
        "upper_95": float(np.quantile(ratios, 0.975)),
    }


def build_folded(device: torch.device, arm: str):
    torch.manual_seed(271828)
    torch.cuda.manual_seed_all(271828)
    model, modules = build_scale_model(device, arm)
    for module in modules:
        module.fold_for_deployment()
    model.to(dtype=torch.bfloat16)
    floating_dtypes = {
        tensor.dtype
        for tensor in (*model.parameters(), *model.buffers())
        if tensor.is_floating_point()
    }
    if floating_dtypes != {torch.bfloat16}:
        raise RuntimeError(f"deployment model is not wholly BF16-resident: {floating_dtypes}")
    return model.eval(), modules


def static_ledger(model: torch.nn.Module) -> dict[str, Any]:
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()),
        "buffer_bytes": sum(buffer.numel() * buffer.element_size() for buffer in model.buffers()),
        "parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()}),
        "floating_buffer_dtypes": sorted({str(buffer.dtype) for buffer in model.buffers() if buffer.is_floating_point()}),
    }


@torch.inference_mode()
def service_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    past_key_values: Any | None = None,
    use_cache: bool = False,
):
    output = model.model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        return_dict=True,
    )
    logits = model.lm_head(output.last_hidden_state[:, -1:, :])
    return logits, output.past_key_values


def timed_call(callable_) -> tuple[float, float, Any]:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    started = time.perf_counter()
    output = callable_()
    end_event.record()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0, float(start_event.elapsed_time(end_event)), output


def time_prefill(
    models: dict[str, torch.nn.Module], randomizer: random.Random, warmups: int, repetitions: int
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for cell_index, (batch, tokens) in enumerate(PREFILL_CELLS):
        inputs = torch.randint(0, 49152, (batch, tokens), device="cuda")
        for model in models.values():
            for _ in range(warmups):
                output = service_forward(model, inputs)
            del output
        torch.cuda.synchronize()
        samples = {clock: {arm: [] for arm in VALID_ARMS} for clock in ("wall_ms", "cuda_event_ms")}
        for _ in range(repetitions):
            order = list(VALID_ARMS)
            randomizer.shuffle(order)
            for arm in order:
                wall, event, output = timed_call(lambda arm=arm: service_forward(models[arm], inputs))
                samples["wall_ms"][arm].append(wall)
                samples["cuda_event_ms"][arm].append(event)
                del output
        cell = {}
        for clock, clock_samples in samples.items():
            cell[clock] = {
                "arms": {
                    arm: {
                        "median": float(np.median(clock_samples[arm])),
                        "p95": percentile(clock_samples[arm], 0.95),
                        "samples": clock_samples[arm],
                    }
                    for arm in VALID_ARMS
                },
                "candidate_bootstrap_ratio": bootstrap_median_ratio(
                    clock_samples["parallel_self_product"],
                    clock_samples["parallel_swiglu"],
                    271828 + 100 * cell_index + (0 if clock == "wall_ms" else 1),
                ),
            }
        results[f"prefill_{batch}x{tokens}"] = cell
        del inputs
    return results


def time_decode(
    models: dict[str, torch.nn.Module], randomizer: random.Random, warmups: int, repetitions: int
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for cell_index, (batch, context) in enumerate(DECODE_CELLS):
        prompt = torch.randint(0, 49152, (batch, context), device="cuda")
        next_token = torch.randint(0, 49152, (batch, 1), device="cuda")
        caches = {}
        for arm, model in models.items():
            output = service_forward(model, prompt, use_cache=True)
            caches[arm] = output[1]
            del output
            if caches[arm].get_seq_length() != context:
                raise RuntimeError("prefill cache length mismatch")
            for _ in range(warmups):
                output = service_forward(model, next_token, caches[arm], use_cache=True)
                caches[arm] = output[1]
                del output
                caches[arm].crop(context)
                if caches[arm].get_seq_length() != context:
                    raise RuntimeError("warmup cache did not reset to frozen context")
        torch.cuda.synchronize()
        samples = {clock: {arm: [] for arm in VALID_ARMS} for clock in ("wall_ms", "cuda_event_ms")}
        for _ in range(repetitions):
            order = list(VALID_ARMS)
            randomizer.shuffle(order)
            for arm in order:
                wall, event, output = timed_call(
                    lambda arm=arm: service_forward(models[arm], next_token, caches[arm], use_cache=True)
                )
                caches[arm] = output[1]
                samples["wall_ms"][arm].append(wall)
                samples["cuda_event_ms"][arm].append(event)
                del output
                caches[arm].crop(context)
                if caches[arm].get_seq_length() != context:
                    raise RuntimeError("timed cache did not reset to frozen context")
        cell = {}
        for clock, clock_samples in samples.items():
            cell[clock] = {
                "arms": {
                    arm: {
                        "median": float(np.median(clock_samples[arm])),
                        "p95": percentile(clock_samples[arm], 0.95),
                        "samples": clock_samples[arm],
                    }
                    for arm in VALID_ARMS
                },
                "candidate_bootstrap_ratio": bootstrap_median_ratio(
                    clock_samples["parallel_self_product"],
                    clock_samples["parallel_swiglu"],
                    272828 + 100 * cell_index + (0 if clock == "wall_ms" else 1),
                ),
            }
        results[f"decode_{batch}x{context}_plus1"] = cell
        del prompt, next_token, caches
    return results


def measure_one_peak(model: torch.nn.Module, inputs: torch.Tensor, core_only: bool) -> dict[str, int]:
    def call():
        if core_only:
            return model.model(input_ids=inputs, use_cache=False, return_dict=True)
        return service_forward(model, inputs)

    output = call()
    torch.cuda.synchronize()
    del output
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_allocated = torch.cuda.memory_allocated()
    base_reserved = torch.cuda.memory_reserved()
    output = call()
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    del output
    return {
        "base_allocated": int(base_allocated),
        "peak_allocated": int(peak_allocated),
        "allocated_increment": int(peak_allocated - base_allocated),
        "base_reserved": int(base_reserved),
        "peak_reserved": int(peak_reserved),
        "reserved_increment": int(peak_reserved - base_reserved),
    }


def peak_ledger(device: torch.device, runtime_floor: dict[str, int]) -> dict[str, Any]:
    peaks: dict[str, Any] = {}
    for arm in VALID_ARMS:
        gc.collect()
        torch.cuda.empty_cache()
        current_floor = {
            "allocated": int(torch.cuda.memory_allocated()),
            "reserved": int(torch.cuda.memory_reserved()),
        }
        if current_floor != runtime_floor:
            raise RuntimeError(f"unstable CUDA runtime floor before isolated {arm} peak: {current_floor} != {runtime_floor}")
        model, modules = build_folded(device, arm)
        peaks[arm] = {"static": static_ledger(model), "cells": {}}
        for batch, tokens in PREFILL_CELLS:
            inputs = torch.randint(0, 49152, (batch, tokens), device=device)
            peaks[arm]["cells"][f"prefill_{batch}x{tokens}"] = {
                "decoder_core": measure_one_peak(model, inputs, core_only=True),
                "service_last_token_logits": measure_one_peak(model, inputs, core_only=False),
            }
            del inputs
        del model, modules
    gc.collect()
    torch.cuda.empty_cache()
    return peaks


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("frozen serving gate requires H100")
    runtime = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
    }
    expected_runtime = {
        "torch": "2.5.1+cu124",
        "cuda": "12.4",
        "transformers": "4.57.6",
    }
    if runtime != expected_runtime:
        raise RuntimeError(f"frozen serving runtime mismatch: {runtime} != {expected_runtime}")
    if (args.warmups, args.repetitions, args.seed) != (5, 30, 271828):
        raise ValueError("frozen scale H100 protocol changed")
    device = torch.device("cuda")
    built = {arm: build_folded(device, arm) for arm in VALID_ARMS}
    models = {arm: value[0] for arm, value in built.items()}
    static = {arm: static_ledger(model) for arm, model in models.items()}
    if len({json.dumps(value, sort_keys=True) for value in static.values()}) != 1:
        raise RuntimeError(f"static BF16 ledger mismatch: {static}")
    randomizer = random.Random(args.seed)
    latency = time_prefill(models, randomizer, args.warmups, args.repetitions)
    latency.update(time_decode(models, randomizer, args.warmups, args.repetitions))
    del built, models
    gc.collect()
    torch.cuda.empty_cache()
    runtime_floor = {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
    }
    peaks = peak_ledger(device, runtime_floor)
    latency_checks = {
        f"{cell}_{clock}": values[clock]["candidate_bootstrap_ratio"]["median_ratio"] <= 1.0
        and values[clock]["candidate_bootstrap_ratio"]["upper_95"] <= 1.02
        for cell, values in latency.items()
        for clock in ("wall_ms", "cuda_event_ms")
    }
    peak_checks = {}
    for cell in peaks["parallel_swiglu"]["cells"]:
        for scope in ("decoder_core", "service_last_token_logits"):
            for metric in ("peak_allocated", "peak_reserved", "allocated_increment", "reserved_increment"):
                peak_checks[f"{cell}_{scope}_{metric}"] = (
                    peaks["parallel_self_product"]["cells"][cell][scope][metric]
                    <= peaks["parallel_swiglu"]["cells"][cell][scope][metric]
                )
    return {
        "schema": "self-product-ffn-scale-h100-v2",
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "protocol": {
            "seed": args.seed,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "prefill_cells": [list(cell) for cell in PREFILL_CELLS],
            "decode_cells": [list(cell) for cell in DECODE_CELLS],
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "weight_dtype": "torch.bfloat16",
        },
        "static_bf16_ledger": static,
        "elementwise_ledger_per_layer_token": {
            "parallel_swiglu": {"silu": 1792, "pointwise_multiply": 1792},
            "parallel_self_product": {"silu": 2688, "pointwise_multiply": 2688},
        },
        "hashes": {
            "source": sha256_file(Path(__file__)),
            "preregistration": sha256_file(PREREGISTRATION),
            "test": sha256_file(TEST_SOURCE),
            "lm_source": sha256_file(LM_SOURCE),
            "lm_preregistration": sha256_file(LM_PREREGISTRATION),
            "lm_test": sha256_file(LM_TEST),
            "base_lm_source": sha256_file(BASE_LM_SOURCE),
            "triangular_source": sha256_file(TRIANGULAR_SOURCE),
            "coalesced_source": sha256_file(COALESCED_SOURCE),
            "reflex_source": sha256_file(REFLEX_SOURCE),
        },
        "latency": latency,
        "peak_memory": peaks,
        "persistent_cuda_runtime_floor_bytes": runtime_floor,
        "latency_checks": latency_checks,
        "peak_checks": peak_checks,
        "h100_feasibility_pass": all(latency_checks.values()),
        "peak_allocation_pass": all(peak_checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--output", type=Path, default=Path("results/self-product-ffn-scale-h100.json"))
    args = parser.parse_args()
    payload = run(args)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "h100_feasibility_pass": payload["h100_feasibility_pass"],
        "peak_allocation_pass": payload["peak_allocation_pass"],
        "latency_checks": payload["latency_checks"],
        "peak_checks": payload["peak_checks"],
        "candidate_ratios": {
            cell: {clock: values[clock]["candidate_bootstrap_ratio"] for clock in ("wall_ms", "cuda_event_ms")}
            for cell, values in payload["latency"].items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
