#!/usr/bin/env python3
"""Fair in-place fused serving gate for the scale self-product FFN."""

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
from triton.language.extra import libdevice
import transformers

from experiments.self_product_ffn_scale import VALID_ARMS, build_scale_model
from experiments.self_product_ffn_scale_h100 import (
    PREFILL_CELLS,
    build_folded,
    sha256_file,
    service_forward,
    static_ledger,
    time_decode,
    time_prefill,
)


PREREGISTRATION = Path("results/self-product-ffn-scale-fused-h100-preregistration.md")
TEST_SOURCE = Path("tests/test_self_product_ffn_scale_fused_h100.py")
INTEGRITY_MANIFEST = Path("results/self-product-ffn-scale-fused-h100-integrity-manifest.json")
EAGER_SOURCE = Path("experiments/self_product_ffn_scale_h100.py")
EAGER_PREREGISTRATION = Path("results/self-product-ffn-scale-h100-preregistration.md")
EAGER_TEST = Path("tests/test_self_product_ffn_scale_h100.py")
EAGER_RESULT = Path("results/self-product-ffn-scale-h100.json")
SCALE_SOURCE = Path("experiments/self_product_ffn_scale.py")
SCALE_PREREGISTRATION = Path("results/self-product-ffn-scale-preregistration.md")
SCALE_TEST = Path("tests/test_self_product_ffn_scale.py")
BASE_LM_SOURCE = Path("experiments/self_product_ffn_lm_screen.py")
TRIANGULAR_SOURCE = Path("experiments/triangular_microdepth_lm_screen.py")
COALESCED_SOURCE = Path("experiments/coalesced_attention_ffn_lm_screen.py")
REFLEX_SOURCE = Path("experiments/reflex_swiglu_lm_screen.py")
ADMISSION_SOURCE = Path("experiments/self_product_ffn_scale_fused_admission.py")
ADMISSION_PREREGISTRATION = Path("results/self-product-ffn-scale-fused-admission-preregistration.md")
ADMISSION_TEST = Path("tests/test_self_product_ffn_scale_fused_admission.py")
CONFIRMATION = Path("results/self-product-ffn-confirmation.json")
DATA_MANIFEST = Path("results/self-product-ffn-scale-data-manifest.json")
EAGER_RESULT_SHA256 = "7e4c7b6b1686a1631c1b90aad8437141ff0299dd15c312aeb76fe62f27415fc8"
SEMANTIC_ATOL = 0.03
SEMANTIC_RTOL = 0.02


@triton.jit
def swiglu_inplace_kernel(gate_ptr, up_ptr, elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sigmoid = 1.0 / (1.0 + libdevice.exp(-gate))
    silu = (gate * sigmoid).to(tl.bfloat16).to(tl.float32)
    tl.store(gate_ptr + offsets, silu * up, mask=mask)


@triton.jit
def swiglu_packed_inplace_kernel(values_ptr, rows: tl.constexpr, width: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    elements = rows * width
    mask = offsets < elements
    row = offsets // width
    column = offsets - row * width
    gate_offsets = row * (2 * width) + column
    up_offsets = gate_offsets + width
    gate = tl.load(values_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(values_ptr + up_offsets, mask=mask, other=0.0).to(tl.float32)
    sigmoid = 1.0 / (1.0 + libdevice.exp(-gate))
    silu = (gate * sigmoid).to(tl.bfloat16).to(tl.float32)
    tl.store(values_ptr + gate_offsets, silu * up, mask=mask)


@triton.jit
def self_product_inplace_kernel(values_ptr, elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    values = tl.load(values_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sigmoid = 1.0 / (1.0 + libdevice.exp(-values))
    silu = (values * sigmoid).to(tl.bfloat16).to(tl.float32)
    tl.store(values_ptr + offsets, silu * values, mask=mask)


@triton.jit
def silu_inplace_kernel(values_ptr, elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    values = tl.load(values_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sigmoid = 1.0 / (1.0 + libdevice.exp(-values))
    tl.store(values_ptr + offsets, values * sigmoid, mask=mask)


def swiglu_inplace(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if not gate.is_contiguous() or not up.is_contiguous() or gate.shape != up.shape:
        raise ValueError("fused SwiGLU requires equal contiguous tensors")
    elements = gate.numel()
    swiglu_inplace_kernel[(triton.cdiv(elements, 256),)](gate, up, elements=elements, BLOCK=256)
    return gate


def swiglu_packed_inplace(values: torch.Tensor) -> torch.Tensor:
    if not values.is_contiguous() or values.shape[-1] % 2:
        raise ValueError("fused packed SwiGLU requires a contiguous even-width tensor")
    width = values.shape[-1] // 2
    rows = values.numel() // (2 * width)
    swiglu_packed_inplace_kernel[(triton.cdiv(rows * width, 256),)](
        values, rows=rows, width=width, BLOCK=256
    )
    return values[..., :width]


def self_product_inplace(values: torch.Tensor) -> torch.Tensor:
    if not values.is_contiguous():
        raise ValueError("fused self-product requires a contiguous tensor")
    elements = values.numel()
    self_product_inplace_kernel[(triton.cdiv(elements, 256),)](values, elements=elements, BLOCK=256)
    return values


def silu_inplace(values: torch.Tensor) -> torch.Tensor:
    if not values.is_contiguous():
        raise ValueError("fused SiLU requires a contiguous tensor")
    elements = values.numel()
    silu_inplace_kernel[(triton.cdiv(elements, 256),)](values, elements=elements, BLOCK=256)
    return values


class FusedSwiGLU(nn.Module):
    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.gate_proj = source.gate_proj
        self.up_proj = source.up_proj
        self.down_proj = source.down_proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        swiglu_inplace(gate, up)
        del up
        return self.down_proj(gate)


class PackedFusedSwiGLU(nn.Module):
    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        hidden_size = source.gate_proj.in_features
        intermediate_size = source.gate_proj.out_features
        self.gate_up_proj = nn.Linear(
            hidden_size,
            2 * intermediate_size,
            bias=False,
            device=source.gate_proj.weight.device,
            dtype=source.gate_proj.weight.dtype,
        )
        with torch.no_grad():
            self.gate_up_proj.weight.copy_(
                torch.cat((source.gate_proj.weight, source.up_proj.weight), dim=0)
            )
        self.down_proj = source.down_proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(hidden_states)
        gate = swiglu_packed_inplace(gate_up)
        return self.down_proj(gate)


class FusedWideAtom(nn.Module):
    def __init__(self, source: nn.Module, self_product: bool) -> None:
        super().__init__()
        self.generator_proj = source.generator_proj
        self.down_proj = source.down_proj
        self.self_product = self_product

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        values = self.generator_proj(hidden_states)
        (self_product_inplace if self.self_product else silu_inplace)(values)
        return self.down_proj(values)


def build_fused(device: torch.device, arm: str, packed_swiglu: bool = False):
    torch.manual_seed(271828)
    torch.cuda.manual_seed_all(271828)
    model, modules = build_scale_model(device, arm)
    for module in modules:
        module.fold_for_deployment()
    for layer in model.model.layers:
        source = layer.mlp
        if arm == "parallel_swiglu":
            layer.mlp = PackedFusedSwiGLU(source) if packed_swiglu else FusedSwiGLU(source)
        else:
            layer.mlp = FusedWideAtom(source, arm == "parallel_self_product")
    model.to(dtype=torch.bfloat16)
    floating_dtypes = {tensor.dtype for tensor in (*model.parameters(), *model.buffers()) if tensor.is_floating_point()}
    if floating_dtypes != {torch.bfloat16}:
        raise RuntimeError(f"fused deployment is not BF16-resident: {floating_dtypes}")
    return model.eval()


@torch.inference_mode()
def exhaustive_bf16_activation_equivalence(device: torch.device) -> dict[str, object]:
    bits = torch.arange(-32768, 32768, device=device, dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16)
    values = values[torch.isfinite(values)].contiguous()
    if values.numel() != 65280:
        raise RuntimeError(f"unexpected finite BF16 domain size: {values.numel()}")
    expected_self_product = torch.nn.functional.silu(values) * values
    actual_self_product = self_product_inplace(values.clone())
    expected_silu = torch.nn.functional.silu(values)
    actual_silu = silu_inplace(values.clone())
    paired_up = values.roll(7919)
    expected_swiglu = torch.nn.functional.silu(values) * paired_up
    actual_swiglu = swiglu_inplace(values.clone(), paired_up)
    packed_values = torch.cat((values.reshape(255, 256), paired_up.reshape(255, 256)), dim=-1)
    actual_packed_swiglu = swiglu_packed_inplace(packed_values).reshape(-1)
    checks = {
        "self_product_all_finite_bf16_bitwise_equal": torch.equal(actual_self_product, expected_self_product),
        "silu_all_finite_bf16_bitwise_equal": torch.equal(actual_silu, expected_silu),
        "swiglu_all_gate_values_fixed_pairing_bitwise_equal": torch.equal(actual_swiglu, expected_swiglu),
        "packed_swiglu_all_gate_values_fixed_pairing_bitwise_equal": torch.equal(actual_packed_swiglu, expected_swiglu),
    }
    return {
        "finite_bf16_values": int(values.numel()),
        "swiglu_pairing_rotation": 7919,
        "packed_swiglu_shape": [255, 512],
        "checks": checks,
        "pass": all(checks.values()),
    }


def tensor_equivalence(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    actual_float = actual.float()
    expected_float = expected.float()
    return {
        "close": bool(torch.allclose(actual_float, expected_float, atol=SEMANTIC_ATOL, rtol=SEMANTIC_RTOL)),
        "max_abs_error": float((actual_float - expected_float).abs().max().item()),
    }


def cache_equivalence(actual, expected) -> dict[str, object]:
    actual_legacy = actual.to_legacy_cache()
    expected_legacy = expected.to_legacy_cache()
    if len(actual_legacy) != len(expected_legacy):
        return {"close": False, "layers": len(actual_legacy), "expected_layers": len(expected_legacy)}
    components = {}
    for layer_index, ((actual_key, actual_value), (expected_key, expected_value)) in enumerate(
        zip(actual_legacy, expected_legacy)
    ):
        components[f"layer_{layer_index}_key"] = tensor_equivalence(actual_key, expected_key)
        components[f"layer_{layer_index}_value"] = tensor_equivalence(actual_value, expected_value)
    return {
        "close": all(component["close"] for component in components.values()),
        "layers": len(actual_legacy),
        "components": components,
    }


@torch.inference_mode()
def semantic_equivalence_ledger(device: torch.device) -> dict[str, object]:
    generator = torch.Generator(device=device).manual_seed(8675309)
    prompt = torch.randint(0, 1000, (2, 16), generator=generator, device=device)
    next_token = torch.randint(0, 1000, (2, 1), generator=generator, device=device)
    arms = {}
    cases = (
        ("parallel_swiglu_split", "parallel_swiglu", False),
        ("parallel_swiglu_packed", "parallel_swiglu", True),
        ("parallel_wide_silu", "parallel_wide_silu", False),
        ("parallel_self_product", "parallel_self_product", False),
    )
    for label, arm, packed_swiglu in cases:
        eager, eager_modules = build_folded(device, arm)
        fused = build_fused(device, arm, packed_swiglu=packed_swiglu)
        eager_logits, eager_cache = service_forward(eager, prompt, use_cache=True)
        fused_logits, fused_cache = service_forward(fused, prompt, use_cache=True)
        prefill_logits = tensor_equivalence(fused_logits, eager_logits)
        prefill_cache = cache_equivalence(fused_cache, eager_cache)
        eager_next_logits, eager_cache = service_forward(eager, next_token, eager_cache, use_cache=True)
        fused_next_logits, fused_cache = service_forward(fused, next_token, fused_cache, use_cache=True)
        next_logits = tensor_equivalence(fused_next_logits, eager_next_logits)
        updated_cache = cache_equivalence(fused_cache, eager_cache)
        checks = {
            "prefill_logits": prefill_logits["close"],
            "prefill_cache": prefill_cache["close"],
            "next_token_logits": next_logits["close"],
            "updated_next_token_cache": updated_cache["close"],
            "cache_length": eager_cache.get_seq_length() == fused_cache.get_seq_length() == 17,
        }
        arms[label] = {
            "checks": checks,
            "pass": all(checks.values()),
            "prefill_logits": prefill_logits,
            "prefill_cache": prefill_cache,
            "next_token_logits": next_logits,
            "updated_next_token_cache": updated_cache,
        }
        del eager_logits, fused_logits, eager_next_logits, fused_next_logits, eager_cache, fused_cache
        del eager, eager_modules, fused
        gc.collect(); torch.cuda.empty_cache()
    del prompt, next_token
    return {
        "seed": 8675309,
        "atol": SEMANTIC_ATOL,
        "rtol": SEMANTIC_RTOL,
        "arms": arms,
        "pass": all(result["pass"] for result in arms.values()),
    }


@torch.inference_mode()
def measure_inference_peak(model: torch.nn.Module, inputs: torch.Tensor, core_only: bool) -> dict[str, int]:
    def call():
        if core_only:
            return model.model(input_ids=inputs, use_cache=False, return_dict=True)
        return service_forward(model, inputs)

    output = call(); torch.cuda.synchronize(); del output
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base_allocated = torch.cuda.memory_allocated(); base_reserved = torch.cuda.memory_reserved()
    output = call(); torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated(); peak_reserved = torch.cuda.max_memory_reserved()
    del output
    return {"base_allocated": int(base_allocated), "peak_allocated": int(peak_allocated), "allocated_increment": int(peak_allocated - base_allocated), "base_reserved": int(base_reserved), "peak_reserved": int(peak_reserved), "reserved_increment": int(peak_reserved - base_reserved)}


def fused_peak_ledger(device: torch.device, runtime_floor: dict[str, int]) -> dict[str, object]:
    peaks = {}
    cases = (
        ("parallel_swiglu_split", "parallel_swiglu", False),
        ("parallel_swiglu_packed", "parallel_swiglu", True),
        ("parallel_wide_silu", "parallel_wide_silu", False),
        ("parallel_self_product", "parallel_self_product", False),
    )
    for label, arm, packed_swiglu in cases:
        gc.collect(); torch.cuda.empty_cache()
        current = {"allocated": int(torch.cuda.memory_allocated()), "reserved": int(torch.cuda.memory_reserved())}
        if current != runtime_floor:
            raise RuntimeError(f"unstable fused runtime floor before {label}: {current} != {runtime_floor}")
        model = build_fused(device, arm, packed_swiglu=packed_swiglu)
        peaks[label] = {"static": static_ledger(model), "cells": {}}
        for batch, tokens in PREFILL_CELLS:
            inputs = torch.randint(0, 49152, (batch, tokens), device=device)
            peaks[label]["cells"][f"prefill_{batch}x{tokens}"] = {
                "decoder_core": measure_inference_peak(model, inputs, core_only=True),
                "service_last_token_logits": measure_inference_peak(model, inputs, core_only=False),
            }
            del inputs
        del model
    gc.collect(); torch.cuda.empty_cache()
    return peaks


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("fused scale gate requires H100")
    runtime = {"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__, "triton": triton.__version__}
    if runtime != {"torch": "2.5.1+cu124", "cuda": "12.4", "transformers": "4.57.6", "triton": "3.1.0"}:
        raise RuntimeError(f"frozen fused runtime mismatch: {runtime}")
    if (args.warmups, args.repetitions, args.seed) != (5, 30, 271828):
        raise ValueError("frozen fused timing protocol changed")
    integrity = json.loads(INTEGRITY_MANIFEST.read_text())
    integrity_paths = {
        "source": Path(__file__), "preregistration": PREREGISTRATION, "test": TEST_SOURCE,
        "eager_source": EAGER_SOURCE, "eager_preregistration": EAGER_PREREGISTRATION,
        "eager_test": EAGER_TEST, "eager_result": EAGER_RESULT,
        "scale_source": SCALE_SOURCE, "scale_preregistration": SCALE_PREREGISTRATION,
        "scale_test": SCALE_TEST, "base_lm_source": BASE_LM_SOURCE,
        "triangular_source": TRIANGULAR_SOURCE, "coalesced_source": COALESCED_SOURCE,
        "reflex_source": REFLEX_SOURCE, "admission_source": ADMISSION_SOURCE,
        "admission_preregistration": ADMISSION_PREREGISTRATION, "admission_test": ADMISSION_TEST,
        "confirmation": CONFIRMATION, "data_manifest": DATA_MANIFEST,
    }
    integrity_checks = {key: integrity.get(key + "_sha256") == sha256_file(path) for key, path in integrity_paths.items()}
    if not all(integrity_checks.values()):
        raise ValueError(f"invalid fused H100 integrity manifest: {integrity_checks}")
    if sha256_file(EAGER_RESULT) != EAGER_RESULT_SHA256:
        raise ValueError("exact frozen eager failure digest changed")
    eager = json.loads(EAGER_RESULT.read_text())
    if eager.get("h100_feasibility_pass") or eager.get("peak_allocation_pass"):
        raise ValueError("fused refinement requires the frozen eager failure")
    eager_hash_checks = {
        "source": eager.get("hashes", {}).get("source") == sha256_file(EAGER_SOURCE),
        "preregistration": eager.get("hashes", {}).get("preregistration") == sha256_file(EAGER_PREREGISTRATION),
        "test": eager.get("hashes", {}).get("test") == sha256_file(EAGER_TEST),
        "scale_source": eager.get("hashes", {}).get("lm_source") == sha256_file(SCALE_SOURCE),
        "base_lm_source": eager.get("hashes", {}).get("base_lm_source") == sha256_file(BASE_LM_SOURCE),
        "triangular_source": eager.get("hashes", {}).get("triangular_source") == sha256_file(TRIANGULAR_SOURCE),
        "coalesced_source": eager.get("hashes", {}).get("coalesced_source") == sha256_file(COALESCED_SOURCE),
        "reflex_source": eager.get("hashes", {}).get("reflex_source") == sha256_file(REFLEX_SOURCE),
    }
    if not all(eager_hash_checks.values()):
        raise ValueError(f"stale eager failure dependency: {eager_hash_checks}")
    device = torch.device("cuda")
    activation_equivalence = exhaustive_bf16_activation_equivalence(device)
    semantic_equivalence = semantic_equivalence_ledger(device)
    semantic_equivalence_pass = activation_equivalence["pass"] and semantic_equivalence["pass"]
    split_models = {arm: build_fused(device, arm) for arm in VALID_ARMS}
    static = {"parallel_swiglu_split": static_ledger(split_models["parallel_swiglu"])}
    static.update({arm: static_ledger(model) for arm, model in split_models.items() if arm != "parallel_swiglu"})
    randomizer = random.Random(args.seed)
    split_latency = time_prefill(split_models, randomizer, args.warmups, args.repetitions)
    split_latency.update(time_decode(split_models, randomizer, args.warmups, args.repetitions))
    del split_models
    gc.collect(); torch.cuda.empty_cache()
    packed_models = {arm: build_fused(device, arm, packed_swiglu=(arm == "parallel_swiglu")) for arm in VALID_ARMS}
    static["parallel_swiglu_packed"] = static_ledger(packed_models["parallel_swiglu"])
    if len({json.dumps(value, sort_keys=True) for value in static.values()}) != 1:
        raise RuntimeError(f"fused static ledger mismatch: {static}")
    packed_latency = time_prefill(packed_models, randomizer, args.warmups, args.repetitions)
    packed_latency.update(time_decode(packed_models, randomizer, args.warmups, args.repetitions))
    del packed_models
    gc.collect(); torch.cuda.empty_cache()
    runtime_floor = {"allocated": int(torch.cuda.memory_allocated()), "reserved": int(torch.cuda.memory_reserved())}
    peaks = fused_peak_ledger(device, runtime_floor)
    latency = {"split_projection_baseline": split_latency, "packed_projection_baseline": packed_latency}
    latency_checks = {}
    for baseline, cells in latency.items():
        for cell, values in cells.items():
            for clock in ("wall_ms", "cuda_event_ms"):
                ratio = values[clock]["candidate_bootstrap_ratio"]
                latency_checks[f"{baseline}_{cell}_{clock}"] = ratio["median_ratio"] <= 1.0 and ratio["upper_95"] <= 1.02
    peak_checks = {}
    for baseline in ("parallel_swiglu_split", "parallel_swiglu_packed"):
        for cell in peaks[baseline]["cells"]:
            for scope in ("decoder_core", "service_last_token_logits"):
                for metric in ("peak_allocated", "peak_reserved", "allocated_increment", "reserved_increment"):
                    peak_checks[f"{baseline}_{cell}_{scope}_{metric}"] = peaks["parallel_self_product"]["cells"][cell][scope][metric] <= peaks[baseline]["cells"][cell][scope][metric]
    return {
        "schema": "self-product-ffn-scale-fused-h100-v1",
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__, "triton_version": triton.__version__,
        "protocol": {"seed": args.seed, "warmups": args.warmups, "repetitions": args.repetitions, "prefill_cells": [[1, 512], [8, 512], [32, 512]], "decode_cells": [[1, 512], [8, 512]], "bootstrap_repetitions": 5000, "weight_dtype": "torch.bfloat16"},
        "static_bf16_ledger": static,
        "elementwise_ledger_per_layer_token": {"parallel_swiglu_split": {"silu": 1792, "pointwise_multiply": 1792}, "parallel_swiglu_packed": {"silu": 1792, "pointwise_multiply": 1792}, "parallel_self_product": {"silu": 2688, "pointwise_multiply": 2688}},
        "baseline_implementations": {
            "split_projection": "two D-to-M input GEMMs plus in-place activation",
            "packed_projection": "one D-to-2M input GEMM plus strided in-place activation",
            "decision_rule": "candidate must pass every latency and peak gate against both baselines",
        },
        "integrity_checks": integrity_checks,
        "hashes": {"source": sha256_file(Path(__file__)), "preregistration": sha256_file(PREREGISTRATION), "test": sha256_file(TEST_SOURCE), "precommit": sha256_file(INTEGRITY_MANIFEST), "eager_source": sha256_file(EAGER_SOURCE), "eager_preregistration": sha256_file(EAGER_PREREGISTRATION), "eager_test": sha256_file(EAGER_TEST), "eager_result": sha256_file(EAGER_RESULT), "lm_source": sha256_file(SCALE_SOURCE), "lm_preregistration": sha256_file(SCALE_PREREGISTRATION), "lm_test": sha256_file(SCALE_TEST), "base_lm_source": sha256_file(BASE_LM_SOURCE), "triangular_source": sha256_file(TRIANGULAR_SOURCE), "coalesced_source": sha256_file(COALESCED_SOURCE), "reflex_source": sha256_file(REFLEX_SOURCE), "admission_source": sha256_file(ADMISSION_SOURCE), "admission_preregistration": sha256_file(ADMISSION_PREREGISTRATION), "admission_test": sha256_file(ADMISSION_TEST), "confirmation": sha256_file(CONFIRMATION), "data_manifest": sha256_file(DATA_MANIFEST)},
        "activation_equivalence": activation_equivalence,
        "semantic_equivalence": semantic_equivalence,
        "semantic_equivalence_pass": semantic_equivalence_pass,
        "latency": latency, "peak_memory": peaks,
        "persistent_cuda_runtime_floor_bytes": runtime_floor,
        "eager_embedded_hash_checks": eager_hash_checks,
        "latency_checks": latency_checks, "peak_checks": peak_checks,
        "h100_feasibility_pass": all(latency_checks.values()) and semantic_equivalence_pass,
        "peak_allocation_pass": all(peak_checks.values()),
        "fused_serving_pass": all(latency_checks.values()) and all(peak_checks.values()) and semantic_equivalence_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--warmups", type=int, default=5); parser.add_argument("--repetitions", type=int, default=30); parser.add_argument("--seed", type=int, default=271828); parser.add_argument("--output", type=Path, default=Path("results/self-product-ffn-scale-fused-h100.json")); args = parser.parse_args()
    payload = run(args); args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fused_serving_pass": payload["fused_serving_pass"], "semantic_equivalence_pass": payload["semantic_equivalence_pass"], "latency_checks": payload["latency_checks"], "peak_checks": payload["peak_checks"], "ratios": {baseline: {cell: {clock: values[clock]["candidate_bootstrap_ratio"] for clock in ("wall_ms", "cuda_event_ms")} for cell, values in cells.items()} for baseline, cells in payload["latency"].items()}}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
