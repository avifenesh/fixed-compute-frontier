#!/usr/bin/env python3
"""H100 gate for a cheap piecewise-linear approximation to self-product."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random

import torch
import torch.nn as nn
import triton
import triton.language as tl
import transformers

from experiments import self_product_ffn_lm_screen as base
from experiments.self_product_ffn_lm_screen import sha256_file
from experiments.self_product_ffn_latency_matched_h100 import (
    BASELINE_WIDTH,
    HIDDEN_SIZE,
    LATENCY_MATCHED_WIDE,
    SEED,
    VALID_ARMS,
    build_latency_matched_model,
)
from experiments.self_product_ffn_scale_fused_h100 import (
    FusedSwiGLU,
    FusedWideAtom,
    PackedFusedSwiGLU,
    cache_equivalence,
    measure_inference_peak,
    tensor_equivalence,
)
from experiments.self_product_ffn_scale_h100 import (
    PREFILL_CELLS,
    service_forward,
    static_ledger,
    time_decode,
    time_prefill,
)


HARD_SLOPE = 0.14709223807891283
HARD_SCALE = 0.47128177407143074
PREREGISTRATION = Path("results/hard-gated-self-product-h100-preregistration.md")
OUTPUT = Path("results/hard-gated-self-product-h100.json")
FAILED_PREDECESSOR = Path("results/self-product-ffn-latency-matched-h100.json")


def hard_gate_reference(values: torch.Tensor) -> torch.Tensor:
    slope_term = (values * HARD_SLOPE).to(torch.bfloat16)
    gate = (slope_term + 0.5).to(torch.bfloat16).clamp(0.0, 1.0)
    # Gate first so saturated negative BF16 values become zero rather than inf*0.
    weighted = (values * gate).to(torch.bfloat16)
    return (weighted * values).to(torch.bfloat16)


@triton.jit
def hard_gate_inplace_kernel(
    values_ptr, elements: tl.constexpr, slope: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    values = tl.load(values_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    slope_term = (values * slope).to(tl.bfloat16).to(tl.float32)
    gate = (slope_term + 0.5).to(tl.bfloat16).to(tl.float32)
    gate = tl.maximum(0.0, tl.minimum(1.0, gate))
    weighted = (values * gate).to(tl.bfloat16).to(tl.float32)
    tl.store(values_ptr + offsets, weighted * values, mask=mask)


def hard_gate_inplace(values: torch.Tensor) -> torch.Tensor:
    if not values.is_contiguous() or values.dtype != torch.bfloat16:
        raise ValueError("hard gate requires contiguous BF16")
    hard_gate_inplace_kernel[(triton.cdiv(values.numel(), 256),)](
        values, elements=values.numel(), slope=HARD_SLOPE, BLOCK=256
    )
    return values


class EagerHardGate(nn.Module):
    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.generator_proj = source.generator_proj
        self.down_proj = source.down_proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(hard_gate_reference(self.generator_proj(hidden_states)))


class FusedHardGate(nn.Module):
    def __init__(self, source: EagerHardGate) -> None:
        super().__init__()
        self.generator_proj = source.generator_proj
        self.down_proj = source.down_proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        values = self.generator_proj(hidden_states)
        return self.down_proj(hard_gate_inplace(values))


def build_eager(device: torch.device, arm: str):
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model, modules = build_latency_matched_model(device, arm)
    if arm == "parallel_self_product":
        with torch.no_grad():
            for layer in model.model.layers:
                layer.mlp.down_proj.weight.mul_(HARD_SCALE)
                layer.mlp = EagerHardGate(layer.mlp)
    else:
        for module in modules:
            module.fold_for_deployment()
    model.to(dtype=torch.bfloat16)
    return model.eval()


def build_fused(device: torch.device, arm: str, packed_swiglu: bool = False):
    model = build_eager(device, arm)
    for layer in model.model.layers:
        source = layer.mlp
        if arm == "parallel_swiglu":
            layer.mlp = PackedFusedSwiGLU(source) if packed_swiglu else FusedSwiGLU(source)
        elif arm == "parallel_wide_silu":
            layer.mlp = FusedWideAtom(source, False)
        else:
            layer.mlp = FusedHardGate(source)
    return model.eval()


@torch.inference_mode()
def activation_equivalence(device: torch.device) -> dict[str, object]:
    bits = torch.arange(-32768, 32768, device=device, dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16)
    values = values[torch.isfinite(values)].contiguous()
    expected = hard_gate_reference(values)
    actual = hard_gate_inplace(values.clone())
    finite_pair = torch.isfinite(actual) & torch.isfinite(expected)
    return {
        "finite_bf16_values": int(values.numel()),
        "bitwise_equal": bool(torch.equal(actual, expected)),
        "max_abs_error_on_finite_outputs": float(
            (actual[finite_pair].float() - expected[finite_pair].float()).abs().max()
        ),
        "pass": bool(torch.equal(actual, expected)),
    }


@torch.inference_mode()
def semantic_equivalence(device: torch.device) -> dict[str, object]:
    generator = torch.Generator(device=device).manual_seed(8675309)
    prompt = torch.randint(0, 1000, (2, 16), generator=generator, device=device)
    next_token = torch.randint(0, 1000, (2, 1), generator=generator, device=device)
    arms = {}
    for label, arm, packed in (
        ("parallel_swiglu_split", "parallel_swiglu", False),
        ("parallel_swiglu_packed", "parallel_swiglu", True),
        ("parallel_wide_silu", "parallel_wide_silu", False),
        ("parallel_self_product", "parallel_self_product", False),
    ):
        eager = build_eager(device, arm); fused = build_fused(device, arm, packed)
        eager_logits, eager_cache = service_forward(eager, prompt, use_cache=True)
        fused_logits, fused_cache = service_forward(fused, prompt, use_cache=True)
        prefill_logits = tensor_equivalence(fused_logits, eager_logits)
        prefill_cache = cache_equivalence(fused_cache, eager_cache)
        eager_next, eager_cache = service_forward(eager, next_token, eager_cache, use_cache=True)
        fused_next, fused_cache = service_forward(fused, next_token, fused_cache, use_cache=True)
        next_logits = tensor_equivalence(fused_next, eager_next)
        next_cache = cache_equivalence(fused_cache, eager_cache)
        checks = {"prefill_logits": prefill_logits["close"], "prefill_cache": prefill_cache["close"],
                  "next_logits": next_logits["close"], "next_cache": next_cache["close"],
                  "cache_length": eager_cache.get_seq_length() == fused_cache.get_seq_length() == 17}
        arms[label] = {"checks": checks, "pass": all(checks.values()),
                       "prefill_logits": prefill_logits, "next_logits": next_logits}
        del eager, fused, eager_logits, fused_logits, eager_next, fused_next, eager_cache, fused_cache
        gc.collect(); torch.cuda.empty_cache()
    return {"arms": arms, "pass": all(value["pass"] for value in arms.values())}


def peak_ledger(device: torch.device, runtime_floor: dict[str, int]) -> dict[str, object]:
    peaks = {}
    for label, arm, packed in (
        ("parallel_swiglu_split", "parallel_swiglu", False),
        ("parallel_swiglu_packed", "parallel_swiglu", True),
        ("parallel_self_product", "parallel_self_product", False),
    ):
        gc.collect(); torch.cuda.empty_cache()
        current = {"allocated": int(torch.cuda.memory_allocated()), "reserved": int(torch.cuda.memory_reserved())}
        if current != runtime_floor:
            raise RuntimeError(f"unstable floor before {label}: {current} != {runtime_floor}")
        model = build_fused(device, arm, packed)
        cells = {}
        for batch, tokens in PREFILL_CELLS:
            inputs = torch.randint(0, 49152, (batch, tokens), device=device)
            cells[f"prefill_{batch}x{tokens}"] = {
                "decoder_core": measure_inference_peak(model, inputs, True),
                "service_last_token_logits": measure_inference_peak(model, inputs, False),
            }
            del inputs
        peaks[label] = {"static": static_ledger(model), "cells": cells}
        del model
    gc.collect(); torch.cuda.empty_cache()
    return peaks


def run(args: argparse.Namespace) -> dict[str, object]:
    expected_runtime = {"torch": "2.5.1+cu124", "cuda": "12.4", "transformers": "4.57.6", "triton": "3.1.0"}
    runtime = {"torch": torch.__version__, "cuda": torch.version.cuda,
               "transformers": transformers.__version__, "triton": triton.__version__}
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    if runtime != expected_runtime or (args.warmups, args.repetitions, args.seed) != (5, 30, SEED):
        raise RuntimeError(f"frozen protocol mismatch: {runtime}, {vars(args)}")
    predecessor = json.loads(FAILED_PREDECESSOR.read_text())
    if predecessor["serving_pass"] or predecessor["shape"]["candidate_width"] != LATENCY_MATCHED_WIDE:
        raise RuntimeError("invalid predecessor")
    device = torch.device("cuda")
    activation = activation_equivalence(device); semantics = semantic_equivalence(device)
    randomizer = random.Random(args.seed)
    split = {arm: build_fused(device, arm) for arm in VALID_ARMS}
    split_latency = time_prefill(split, randomizer, args.warmups, args.repetitions)
    split_latency.update(time_decode(split, randomizer, args.warmups, args.repetitions))
    del split; gc.collect(); torch.cuda.empty_cache()
    packed = {arm: build_fused(device, arm, arm == "parallel_swiglu") for arm in VALID_ARMS}
    packed_latency = time_prefill(packed, randomizer, args.warmups, args.repetitions)
    packed_latency.update(time_decode(packed, randomizer, args.warmups, args.repetitions))
    del packed; gc.collect(); torch.cuda.empty_cache()
    floor = {"allocated": int(torch.cuda.memory_allocated()), "reserved": int(torch.cuda.memory_reserved())}
    peaks = peak_ledger(device, floor)
    latency = {"split_swiglu": split_latency, "packed_swiglu": packed_latency}
    latency_checks = {}
    for baseline, cells in latency.items():
        for cell, values in cells.items():
            for clock in ("wall_ms", "cuda_event_ms"):
                ratio = values[clock]["candidate_bootstrap_ratio"]
                latency_checks[f"{baseline}_{cell}_{clock}"] = ratio["median_ratio"] <= 1 and ratio["upper_95"] <= 1.02
    static_checks = {f"static_lt_{baseline}": peaks["parallel_self_product"]["static"]["parameter_bytes"] < peaks[baseline]["static"]["parameter_bytes"]
                     for baseline in ("parallel_swiglu_split", "parallel_swiglu_packed")}
    allocated_checks = {}
    reserved_diagnostics = {}
    for baseline in ("parallel_swiglu_split", "parallel_swiglu_packed"):
        for cell in peaks[baseline]["cells"]:
            for scope in ("decoder_core", "service_last_token_logits"):
                for metric in ("peak_allocated", "allocated_increment"):
                    allocated_checks[f"{baseline}_{cell}_{scope}_{metric}"] = peaks["parallel_self_product"]["cells"][cell][scope][metric] <= peaks[baseline]["cells"][cell][scope][metric]
                for metric in ("peak_reserved", "reserved_increment"):
                    reserved_diagnostics[f"{baseline}_{cell}_{scope}_{metric}"] = peaks["parallel_self_product"]["cells"][cell][scope][metric] <= peaks[baseline]["cells"][cell][scope][metric]
    semantic_pass = activation["pass"] and semantics["pass"]
    payload = {"schema": "hard-gated-self-product-h100-v1", "device": torch.cuda.get_device_name(0),
               "runtime": runtime, "shape": {"hidden": HIDDEN_SIZE, "baseline_width": BASELINE_WIDTH,
               "candidate_width": LATENCY_MATCHED_WIDE, "hard_slope": HARD_SLOPE, "hard_scale": HARD_SCALE},
               "protocol": {"seed": args.seed, "warmups": args.warmups, "repetitions": args.repetitions,
               "prefill_cells": [list(x) for x in PREFILL_CELLS], "decode_cells": [[1,512],[8,512]], "bootstrap_repetitions": 5000},
               "hashes": {"source": sha256_file(Path(__file__)), "preregistration": sha256_file(PREREGISTRATION),
                           "failed_predecessor": sha256_file(FAILED_PREDECESSOR)},
               "activation_equivalence": activation, "semantic_equivalence": semantics,
               "latency": latency, "peak_memory": peaks, "runtime_floor": floor,
               "latency_checks": latency_checks, "static_checks": static_checks,
               "allocated_checks": allocated_checks, "reserved_diagnostics": reserved_diagnostics,
               "serving_pass": semantic_pass and all(latency_checks.values()) and all(static_checks.values()) and all(allocated_checks.values())}
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30); parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT); args = parser.parse_args()
    payload = run(args); args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    ratios = {baseline: {cell: {clock: values[clock]["candidate_bootstrap_ratio"] for clock in ("wall_ms", "cuda_event_ms")}
                         for cell, values in cells.items()} for baseline, cells in payload["latency"].items()}
    print(json.dumps({"serving_pass": payload["serving_pass"], "activation": payload["activation_equivalence"],
                      "ratios": ratios, "failed_latency": [k for k,v in payload["latency_checks"].items() if not v],
                      "failed_allocated": [k for k,v in payload["allocated_checks"].items() if not v]}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
