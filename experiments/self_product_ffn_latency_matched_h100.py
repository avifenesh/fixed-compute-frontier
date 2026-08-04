#!/usr/bin/env python3
"""Single-point H100 serving gate for the smaller self-product FFN."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random

import torch
import triton
import transformers

from experiments import self_product_ffn_lm_screen as base
from experiments import self_product_ffn_scale as scale
from experiments.self_product_ffn_lm_screen import sha256_file
from experiments.self_product_ffn_scale_fused_h100 import (
    FusedSwiGLU,
    FusedWideAtom,
    PackedFusedSwiGLU,
    exhaustive_bf16_activation_equivalence,
    measure_inference_peak,
    tensor_equivalence,
    cache_equivalence,
)
from experiments.self_product_ffn_scale_h100 import (
    PREFILL_CELLS,
    service_forward,
    static_ledger,
    time_decode,
    time_prefill,
)


LATENCY_MATCHED_WIDE = 2560
BASELINE_WIDTH = 1792
HIDDEN_SIZE = 640
LAYERS = 16
SEED = 271828
VALID_ARMS = base.VALID_ARMS
PREREGISTRATION = Path("results/self-product-ffn-latency-matched-h100-preregistration.md")
OUTPUT = Path("results/self-product-ffn-latency-matched-h100.json")
DEPENDENCIES = (
    Path("experiments/self_product_ffn_lm_screen.py"),
    Path("experiments/self_product_ffn_scale.py"),
    Path("experiments/self_product_ffn_scale_h100.py"),
    Path("experiments/self_product_ffn_scale_fused_h100.py"),
    Path("experiments/triangular_microdepth_lm_screen.py"),
    Path("experiments/coalesced_attention_ffn_lm_screen.py"),
)
SEMANTIC_ATOL = 0.03
SEMANTIC_RTOL = 0.02


def configure_latency_matched_globals() -> None:
    scale.configure_scale_globals()
    base.WIDE = LATENCY_MATCHED_WIDE
    base.PLAIN_SCALE = math.sqrt(BASELINE_WIDTH / LATENCY_MATCHED_WIDE)


def build_latency_matched_model(device: torch.device, arm: str):
    configure_latency_matched_globals()
    return scale.ORIGINAL_BUILD_MODEL(device, arm)


def build_folded(device: torch.device, arm: str):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model, modules = build_latency_matched_model(device, arm)
    for module in modules:
        module.fold_for_deployment()
    model.to(dtype=torch.bfloat16)
    return model.eval(), modules


def build_fused(device: torch.device, arm: str, packed_swiglu: bool = False):
    model, modules = build_folded(device, arm)
    for layer in model.model.layers:
        source = layer.mlp
        if arm == "parallel_swiglu":
            layer.mlp = PackedFusedSwiGLU(source) if packed_swiglu else FusedSwiGLU(source)
        else:
            layer.mlp = FusedWideAtom(source, arm == "parallel_self_product")
    del modules
    return model.eval()


@torch.inference_mode()
def semantic_equivalence(device: torch.device) -> dict[str, object]:
    generator = torch.Generator(device=device).manual_seed(8675309)
    prompt = torch.randint(0, 1000, (2, 16), generator=generator, device=device)
    next_token = torch.randint(0, 1000, (2, 1), generator=generator, device=device)
    arms: dict[str, object] = {}
    cases = (
        ("parallel_swiglu_split", "parallel_swiglu", False),
        ("parallel_swiglu_packed", "parallel_swiglu", True),
        ("parallel_wide_silu", "parallel_wide_silu", False),
        ("parallel_self_product", "parallel_self_product", False),
    )
    for label, arm, packed in cases:
        eager, modules = build_folded(device, arm)
        fused = build_fused(device, arm, packed)
        eager_logits, eager_cache = service_forward(eager, prompt, use_cache=True)
        fused_logits, fused_cache = service_forward(fused, prompt, use_cache=True)
        prefill_logits = tensor_equivalence(fused_logits, eager_logits)
        prefill_cache = cache_equivalence(fused_cache, eager_cache)
        eager_next, eager_cache = service_forward(eager, next_token, eager_cache, use_cache=True)
        fused_next, fused_cache = service_forward(fused, next_token, fused_cache, use_cache=True)
        next_logits = tensor_equivalence(fused_next, eager_next)
        next_cache = cache_equivalence(fused_cache, eager_cache)
        checks = {
            "prefill_logits": prefill_logits["close"],
            "prefill_cache": prefill_cache["close"],
            "next_logits": next_logits["close"],
            "next_cache": next_cache["close"],
            "cache_length": eager_cache.get_seq_length() == fused_cache.get_seq_length() == 17,
        }
        arms[label] = {
            "checks": checks,
            "pass": all(checks.values()),
            "prefill_logits": prefill_logits,
            "next_logits": next_logits,
        }
        del eager, fused, modules, eager_logits, fused_logits, eager_next, fused_next
        del eager_cache, fused_cache
        gc.collect(); torch.cuda.empty_cache()
    return {
        "atol": SEMANTIC_ATOL,
        "rtol": SEMANTIC_RTOL,
        "arms": arms,
        "pass": all(value["pass"] for value in arms.values()),
    }


def isolated_peaks(device: torch.device, runtime_floor: dict[str, int]) -> dict[str, object]:
    peaks: dict[str, object] = {}
    cases = (
        ("parallel_swiglu_split", "parallel_swiglu", False),
        ("parallel_swiglu_packed", "parallel_swiglu", True),
        ("parallel_wide_silu", "parallel_wide_silu", False),
        ("parallel_self_product", "parallel_self_product", False),
    )
    for label, arm, packed in cases:
        gc.collect(); torch.cuda.empty_cache()
        current = {
            "allocated": int(torch.cuda.memory_allocated()),
            "reserved": int(torch.cuda.memory_reserved()),
        }
        if current != runtime_floor:
            raise RuntimeError(f"unstable CUDA runtime floor before {label}: {current} != {runtime_floor}")
        model = build_fused(device, arm, packed)
        cells = {}
        for batch, tokens in PREFILL_CELLS:
            inputs = torch.randint(0, 49152, (batch, tokens), device=device)
            cells[f"prefill_{batch}x{tokens}"] = {
                "decoder_core": measure_inference_peak(model, inputs, core_only=True),
                "service_last_token_logits": measure_inference_peak(model, inputs, core_only=False),
            }
            del inputs
        peaks[label] = {"static": static_ledger(model), "cells": cells}
        del model
    gc.collect(); torch.cuda.empty_cache()
    return peaks


def run(args: argparse.Namespace) -> dict[str, object]:
    runtime = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__, "triton": triton.__version__,
    }
    expected = {
        "torch": "2.5.1+cu124", "cuda": "12.4",
        "transformers": "4.57.6", "triton": "3.1.0",
    }
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("this gate requires an H100")
    if runtime != expected or (args.warmups, args.repetitions, args.seed) != (5, 30, SEED):
        raise RuntimeError(f"frozen protocol mismatch: runtime={runtime}, args={vars(args)}")
    device = torch.device("cuda")
    activation = exhaustive_bf16_activation_equivalence(device)
    semantics = semantic_equivalence(device)

    split_models = {arm: build_fused(device, arm) for arm in VALID_ARMS}
    split_static = {arm: static_ledger(model) for arm, model in split_models.items()}
    randomizer = random.Random(args.seed)
    split_latency = time_prefill(split_models, randomizer, args.warmups, args.repetitions)
    split_latency.update(time_decode(split_models, randomizer, args.warmups, args.repetitions))
    del split_models
    gc.collect(); torch.cuda.empty_cache()

    packed_models = {
        arm: build_fused(device, arm, packed_swiglu=(arm == "parallel_swiglu"))
        for arm in VALID_ARMS
    }
    packed_static = {arm: static_ledger(model) for arm, model in packed_models.items()}
    packed_latency = time_prefill(packed_models, randomizer, args.warmups, args.repetitions)
    packed_latency.update(time_decode(packed_models, randomizer, args.warmups, args.repetitions))
    del packed_models
    gc.collect(); torch.cuda.empty_cache()

    runtime_floor = {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
    }
    peaks = isolated_peaks(device, runtime_floor)
    latency = {"split_swiglu": split_latency, "packed_swiglu": packed_latency}
    latency_checks = {}
    for baseline, cells in latency.items():
        for cell, values in cells.items():
            for clock in ("wall_ms", "cuda_event_ms"):
                ratio = values[clock]["candidate_bootstrap_ratio"]
                latency_checks[f"{baseline}_{cell}_{clock}"] = (
                    ratio["median_ratio"] <= 1.0 and ratio["upper_95"] <= 1.02
                )
    static_checks = {
        f"candidate_bytes_lt_{baseline}":
        peaks["parallel_self_product"]["static"]["parameter_bytes"]
        < peaks[baseline]["static"]["parameter_bytes"]
        for baseline in ("parallel_swiglu_split", "parallel_swiglu_packed")
    }
    peak_checks = {}
    for baseline in ("parallel_swiglu_split", "parallel_swiglu_packed"):
        for cell in peaks[baseline]["cells"]:
            for scope in ("decoder_core", "service_last_token_logits"):
                for metric in ("peak_allocated", "peak_reserved", "allocated_increment", "reserved_increment"):
                    peak_checks[f"{baseline}_{cell}_{scope}_{metric}"] = (
                        peaks["parallel_self_product"]["cells"][cell][scope][metric]
                        <= peaks[baseline]["cells"][cell][scope][metric]
                    )
    semantic_pass = bool(activation["pass"] and semantics["pass"])
    payload = {
        "schema": "self-product-ffn-latency-matched-h100-v1",
        "candidate": "self_product_width_2560",
        "shape": {"hidden_size": HIDDEN_SIZE, "baseline_width": BASELINE_WIDTH,
                  "candidate_width": LATENCY_MATCHED_WIDE, "layers": LAYERS},
        "analytic_cost": {
            "baseline_ffn_dense_per_layer": 3 * HIDDEN_SIZE * BASELINE_WIDTH,
            "candidate_ffn_dense_per_layer": 2 * HIDDEN_SIZE * LATENCY_MATCHED_WIDE,
            "candidate_ffn_dense_ratio": (2 * HIDDEN_SIZE * LATENCY_MATCHED_WIDE) / (3 * HIDDEN_SIZE * BASELINE_WIDTH),
            "candidate_elementwise_ratio": LATENCY_MATCHED_WIDE / BASELINE_WIDTH,
        },
        "device": torch.cuda.get_device_name(0), "runtime": runtime,
        "protocol": {"seed": args.seed, "warmups": args.warmups, "repetitions": args.repetitions,
                     "prefill_cells": [list(cell) for cell in PREFILL_CELLS],
                     "decode_cells": [[1, 512], [8, 512]], "bootstrap_repetitions": 5000},
        "hashes": {"source": sha256_file(Path(__file__)),
                   "preregistration": sha256_file(PREREGISTRATION),
                   **{path.name: sha256_file(path) for path in DEPENDENCIES}},
        "activation_equivalence": activation, "semantic_equivalence": semantics,
        "split_static": split_static, "packed_static": packed_static,
        "latency": latency, "peak_memory": peaks, "runtime_floor": runtime_floor,
        "semantic_pass": semantic_pass, "latency_checks": latency_checks,
        "static_checks": static_checks, "peak_checks": peak_checks,
        "serving_pass": semantic_pass and all(latency_checks.values())
                        and all(static_checks.values()) and all(peak_checks.values()),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    payload = run(args)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    ratios = {
        baseline: {
            cell: {clock: values[clock]["candidate_bootstrap_ratio"] for clock in ("wall_ms", "cuda_event_ms")}
            for cell, values in cells.items()
        }
        for baseline, cells in payload["latency"].items()
    }
    print(json.dumps({"serving_pass": payload["serving_pass"], "ratios": ratios,
                      "static_checks": payload["static_checks"],
                      "failed_latency": [key for key, value in payload["latency_checks"].items() if not value],
                      "failed_peak": [key for key, value in payload["peak_checks"].items() if not value]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
