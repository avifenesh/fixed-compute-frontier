#!/usr/bin/env python3
"""Matched LM quality pilot for dense zero-pivot G1 attention."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import platform
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import triton
import transformers
from transformers import AutoModelForCausalLM

from experiments.gauge_slot_lm_pilot import (
    attention_diagnostics,
    optimizer_for,
    optimizer_state_bytes,
)
from experiments.gauge_zero_g1_attention import (
    GaugeZeroG1Attention,
    replace_llama_attention,
)
from experiments.gauge_zero_g1_h100 import rope_inplace
from experiments.reflex_swiglu_lm_screen import (
    MODEL,
    MODEL_REVISION,
    TokenFile,
    evaluate,
    lr_multiplier,
    paired_loss_interval,
    sha256_file,
    validate_data_ledger,
    write_payload,
)
from experiments.triangular_microdepth_lm_screen import causal_loss, scratch_config


ARMS = ("raw_baseline", "canonical_bilinear_control", "gauge_zero_g1")
OUTPUT = Path("results/gauge-zero-g1-lm-pilot.json")
PREREGISTRATION = Path("results/gauge-zero-g1-lm-preregistration.md")
INTEGRITY_MANIFEST = Path("results/gauge-zero-g1-lm-integrity-manifest.json")
ATTENTION_SOURCE = Path("experiments/gauge_zero_g1_attention.py")
TEST_SOURCE = Path("tests/test_gauge_zero_g1_lm_pilot.py")
ATTENTION_TEST = Path("tests/test_gauge_zero_g1_attention.py")
INITIAL_EVAL = Path("results/gauge-zero-g1-initial-eval.json")
INITIAL_EVAL_SOURCE = Path("experiments/gauge_zero_g1_initial_eval.py")
H100_V3_RESULT = Path("results/gauge-zero-g1-h100-v3.json")
H100_V3_PREREGISTRATION = Path("results/gauge-zero-g1-h100-v3-preregistration.md")
H100_V3_INTEGRITY_MANIFEST = Path("results/gauge-zero-g1-h100-v3-integrity-manifest.json")
H100_V3_SOURCE = Path("experiments/gauge_zero_g1_h100_v3.py")
H100_BASE_SOURCE = Path("experiments/gauge_zero_g1_h100.py")
H100_TEST = Path("tests/test_gauge_zero_g1_h100.py")
GAUGE_SLOT_LM_SOURCE = Path("experiments/gauge_slot_lm_pilot.py")
REFLEX_SOURCE = Path("experiments/reflex_swiglu_lm_screen.py")
TRIANGULAR_SOURCE = Path("experiments/triangular_microdepth_lm_screen.py")
DATA_MANIFEST = Path("results/block-algebra-scratch-data-manifest.json")


def validate_protocol(args: argparse.Namespace) -> dict[str, Any]:
    config = scratch_config()
    expected = {
        "device_contains": "H100",
        "python_version": "3.11.10",
        "numpy_version": "2.1.2",
        "torch_version": "2.5.1+cu124",
        "cuda_version": "12.4",
        "transformers_version": "4.57.6",
        "triton_version": "3.1.0",
        "sequence_length": 512,
        "micro_batch_size": 32,
        "gradient_accumulation": 2,
        "eval_batch_size": 32,
        "eval_batches": 64,
        "steps": 305,
        "eval_steps": [61, 305],
        "warmup_steps": 50,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
        "seed": 2207,
        "prediction_tokens": 9_994_240,
        "formal": True,
        "output": str(OUTPUT),
        "checkpoint_dir": "results/gauge-zero-g1-lm-pilot-checkpoints",
        "train_file": "data/block-algebra-scratch/train.uint16.bin",
        "validation_file": "data/block-algebra-scratch/validation.uint16.bin",
        "data_manifest": str(DATA_MANIFEST),
        "hidden_size": 384,
        "layers": 12,
        "query_heads": 6,
        "kv_heads": 2,
        "head_dim": 64,
        "intermediate_size": 1024,
    }
    actual = {
        "device_contains": torch.cuda.get_device_name(0),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "triton_version": triton.__version__,
        "sequence_length": args.sequence_length,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "eval_batch_size": args.eval_batch_size,
        "eval_batches": args.eval_batches,
        "steps": args.steps,
        "eval_steps": args.eval_steps,
        "warmup_steps": args.warmup_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "seed": args.seed,
        "prediction_tokens": (
            args.steps * args.gradient_accumulation * args.micro_batch_size
            * args.sequence_length
        ),
        "formal": args.formal,
        "output": str(args.output),
        "checkpoint_dir": str(args.checkpoint_dir),
        "train_file": str(args.train_file),
        "validation_file": str(args.validation_file),
        "data_manifest": str(args.data_manifest),
        "hidden_size": config.hidden_size,
        "layers": config.num_hidden_layers,
        "query_heads": config.num_attention_heads,
        "kv_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "intermediate_size": config.intermediate_size,
    }
    checks = {
        key: (
            expected_value in actual[key]
            if key == "device_contains"
            else actual[key] == expected_value
        )
        for key, expected_value in expected.items()
    }
    if args.formal and not all(checks.values()):
        failures = {
            key: {"expected": expected[key], "actual": actual[key]}
            for key, valid in checks.items()
            if not valid
        }
        raise ValueError(f"invalid frozen protocol: {json.dumps(failures, sort_keys=True)}")
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "expected": expected,
        "actual": actual,
    }


def validate_formal_integrity() -> dict[str, Any]:
    manifest = json.loads(INTEGRITY_MANIFEST.read_text())
    paths = {
        "source": Path(__file__),
        "attention_source": ATTENTION_SOURCE,
        "test": TEST_SOURCE,
        "attention_test": ATTENTION_TEST,
        "preregistration": PREREGISTRATION,
        "initial_eval": INITIAL_EVAL,
        "initial_eval_source": INITIAL_EVAL_SOURCE,
        "h100_v3_result": H100_V3_RESULT,
        "h100_v3_preregistration": H100_V3_PREREGISTRATION,
        "h100_v3_integrity_manifest": H100_V3_INTEGRITY_MANIFEST,
        "h100_v3_source": H100_V3_SOURCE,
        "h100_base_source": H100_BASE_SOURCE,
        "h100_test": H100_TEST,
        "gauge_slot_lm_source": GAUGE_SLOT_LM_SOURCE,
        "reflex_source": REFLEX_SOURCE,
        "triangular_source": TRIANGULAR_SOURCE,
        "data_manifest": DATA_MANIFEST,
    }
    hash_checks = {
        name: manifest.get(name + "_sha256") == sha256_file(path)
        for name, path in paths.items()
    }
    initial_eval = json.loads(INITIAL_EVAL.read_text())
    h100_v3 = json.loads(H100_V3_RESULT.read_text())
    prerequisite_checks = {
        "initial_eval_all_checks_pass": all(initial_eval["checks"].values()),
        "unified_one_round_h100_v3_passed": h100_v3["all_gates_pass"] is True,
        "h100_v3_integrity_was_valid": all(h100_v3["integrity"].values()),
        "h100_v3_manifest_binding_valid": (
            h100_v3["integrity_manifest_sha256"]
            == sha256_file(H100_V3_INTEGRITY_MANIFEST)
        ),
        "unified_one_round_contract_recorded": h100_v3["numerical_contract"] == (
            "BF16 dense projection and coefficient; shear and split-half RoPE in FP32; one BF16 final store"
        ),
    }
    checks = {**hash_checks, **prerequisite_checks}
    if not all(checks.values()):
        raise ValueError(f"invalid frozen integrity: {checks}")
    return {
        "valid": True,
        "checks": checks,
        "manifest_sha256": sha256_file(INTEGRITY_MANIFEST),
    }


def build_model(
    device: torch.device, arm: str
) -> tuple[nn.Module, list[GaugeZeroG1Attention]]:
    if arm not in ARMS:
        raise ValueError(f"invalid arm: {arm}")
    model = AutoModelForCausalLM.from_config(
        scratch_config(), attn_implementation="sdpa"
    ).to(device)
    model.config.use_cache = False
    modules: list[GaugeZeroG1Attention] = []
    if arm != "raw_baseline":
        modules = replace_llama_attention(model, arm)
    return model, modules


@torch.no_grad()
def coefficient_serving_diagnostics(
    modules: list[GaugeZeroG1Attention],
) -> dict[str, float | bool]:
    if not modules:
        return {
            "coefficient_bf16_nonzero_fraction": 0.0,
            "coefficient_bf16_abs_max": 0.0,
            "bf16_export_coefficient_bit_exact": True,
        }
    fp32 = torch.cat([
        module.curvature_coefficients().float().reshape(-1) for module in modules
    ])
    training_bf16 = fp32.to(torch.bfloat16)
    exported = []
    for module in modules:
        bf16_weight = module.key_weight.detach().to(torch.bfloat16)
        physical = bf16_weight.reshape(
            module.kv_heads, module.head_dim, module.hidden_size
        )
        exported.append(
            physical[:, module.pairs:, :module.pairs]
            .diagonal(dim1=-2, dim2=-1)
            .reshape(-1)
            / module.tau
        )
    exported_bf16 = torch.cat(exported)
    return {
        "coefficient_bf16_nonzero_fraction": float(
            (training_bf16 != 0).float().mean()
        ),
        "coefficient_bf16_abs_max": float(training_bf16.abs().max()),
        "bf16_export_coefficient_bit_exact": bool(torch.equal(
            training_bf16, exported_bf16
        )),
    }


@torch.no_grad()
def terminal_serving_bridge_diagnostics(
    modules: list[GaugeZeroG1Attention],
) -> dict[str, Any]:
    """Compare every trained layer to the exact fused H100 export contract."""
    if not modules:
        return {"layers": 0, "all_layers_bf16_bit_exact": True, "max_abs": 0.0}
    generator = torch.Generator(device="cuda").manual_seed(9473)
    exact = []
    max_abs = []
    cache_bytes = []
    for module in modules:
        rows = 7
        hidden = torch.randn(
            1,
            rows,
            module.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        angles = torch.randn(
            1,
            rows,
            module.pairs,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        cosine_half = angles.cos().to(torch.bfloat16)
        sine_half = angles.sin().to(torch.bfloat16)
        position = (
            torch.cat((cosine_half, cosine_half), dim=-1),
            torch.cat((sine_half, sine_half), dim=-1),
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            query, key, value = module.project_qkv_rope(hidden, position)
        exported_weight = torch.cat((
            module.query_weight.detach().to(torch.bfloat16),
            module.key_weight.detach().to(torch.bfloat16),
            module.v_proj.weight.detach().to(torch.bfloat16),
        ), dim=0).contiguous()
        serving = hidden.squeeze(0) @ exported_weight.T
        rope_inplace(
            serving,
            exported_weight,
            cosine_half.squeeze(0),
            sine_half.squeeze(0),
            g1=True,
            hidden=module.hidden_size,
            query_heads=module.query_heads,
            kv_heads=module.kv_heads,
            head_dim=module.head_dim,
            tau=module.tau,
        )
        q_width = module.query_heads * module.head_dim
        k_width = module.kv_heads * module.head_dim
        prototype = torch.cat((
            query.transpose(1, 2).reshape(rows, q_width),
            key.transpose(1, 2).reshape(rows, k_width),
            value.transpose(1, 2).reshape(rows, k_width),
        ), dim=-1)
        difference = (serving.float() - prototype.float()).abs()
        exact.append(bool(torch.equal(serving, prototype)))
        max_abs.append(float(difference.max()))
        cache_bytes.append(key.numel() * key.element_size())
    torch.cuda.synchronize()
    return {
        "layers": len(modules),
        "all_layers_bf16_bit_exact": all(exact),
        "max_abs": max(max_abs),
        "key_cache_bytes_per_checked_layer": cache_bytes,
        "numerical_contract": (
            "BF16 dense projection and coefficient; shear and split-half RoPE in FP32; one BF16 final store"
        ),
    }


def save_terminal_artifacts(
    model: nn.Module,
    modules: list[GaugeZeroG1Attention],
    arm: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not args.formal:
        return {"saved": False}
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / f"seed-{args.seed}-{arm}.pt"
    state = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    torch.save({"arm": arm, "seed": args.seed, "state_dict": state}, checkpoint_path)
    result: dict[str, Any] = {
        "saved": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
    }
    del state
    if arm == "gauge_zero_g1":
        export_path = (
            args.checkpoint_dir
            / f"seed-{args.seed}-candidate-bf16-dense-attention.pt"
        )
        exported_layers = []
        for module in modules:
            exported_layers.append({
                key: (
                    value.detach().to(torch.bfloat16).cpu()
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in module.export_dense().items()
            })
        torch.save({"layers": exported_layers}, export_path)
        result.update({
            "bf16_attention_export": str(export_path),
            "bf16_attention_export_sha256": sha256_file(export_path),
            "bf16_attention_export_bytes": export_path.stat().st_size,
        })
    return result


@torch.no_grad()
def evaluate_disable_shear(
    model: nn.Module,
    modules: list[GaugeZeroG1Attention],
    validation_file: TokenFile,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    # Keep every trained dense Q/K weight fixed and disable only the proposed
    # quadratic reuse. Zeroing the coefficient-bearing K weights would also
    # remove their ordinary linear contribution and confound the ablation.
    saved_arms = [module.arm for module in modules]
    try:
        for module in modules:
            module.arm = "canonical_bilinear_control"
        return evaluate(
            model, validation_file, args.eval_batches, args.eval_batch_size, device
        )
    finally:
        for module, saved_arm in zip(modules, saved_arms):
            module.arm = saved_arm


def train_arm(
    arm: str,
    args: argparse.Namespace,
    train_file: TokenFile,
    validation_file: TokenFile,
    device: torch.device,
) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, modules = build_model(device, arm)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    parameter_tensors = len(list(model.parameters()))
    model_buffers = list(model.buffers())
    model_buffer_values = sum(buffer.numel() for buffer in model_buffers)
    model_buffer_tensors = len(model_buffers)
    state_values = sum(value.numel() for value in model.state_dict().values())
    state_bytes = sum(
        value.numel() * value.element_size()
        for value in model.state_dict().values()
    )
    optimizer = optimizer_for(model, args.learning_rate, args.weight_decay)
    diagnostics = {
        "initial": attention_diagnostics(model, modules, validation_file, device)
    }
    serving_diagnostics = {
        "initial": coefficient_serving_diagnostics(modules)
    }
    evaluations = {
        "0": evaluate(
            model, validation_file, args.eval_batches, args.eval_batch_size, device
        )
    }
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    losses: list[float] = []
    gradient_norms: list[float] = []
    durations: list[float] = []
    for step in range(args.steps):
        model.train()
        multiplier = lr_multiplier(step, args.steps, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * multiplier
        started = time.perf_counter()
        accumulated = 0.0
        for micro in range(args.gradient_accumulation):
            batch_index = step * args.gradient_accumulation + micro
            inputs, targets = train_file.batch(
                batch_index, args.micro_batch_size, device
            )
            loss = causal_loss(model, inputs, targets)
            (loss / args.gradient_accumulation).backward()
            accumulated += float(loss.detach()) / args.gradient_accumulation
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_clip
        ))
        if not math.isfinite(accumulated) or not math.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite {arm} at step {step + 1}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        losses.append(accumulated)
        gradient_norms.append(gradient_norm)
        durations.append(time.perf_counter() - started)
        if step + 1 in args.eval_steps:
            evaluations[str(step + 1)] = evaluate(
                model, validation_file, args.eval_batches, args.eval_batch_size, device
            )
            print(json.dumps({
                "arm": arm,
                "step": step + 1,
                "train_loss": accumulated,
                "validation_loss": evaluations[str(step + 1)]["loss"],
                "gradient_norm": gradient_norm,
            }), flush=True)
    diagnostics["terminal"] = attention_diagnostics(
        model, modules, validation_file, device
    )
    serving_diagnostics["terminal"] = coefficient_serving_diagnostics(modules)
    ablations = {}
    if arm == "gauge_zero_g1":
        ablations["disable_shear"] = evaluate_disable_shear(
            model, modules, validation_file, args, device
        )
    serving_bridge = (
        terminal_serving_bridge_diagnostics(modules)
        if arm == "gauge_zero_g1"
        else {"layers": 0, "all_layers_bf16_bit_exact": True, "max_abs": 0.0}
    )
    terminal_artifacts = save_terminal_artifacts(model, modules, arm, args)
    prediction_tokens = (
        args.steps * args.gradient_accumulation * args.micro_batch_size
        * args.sequence_length
    )
    result = {
        "arm": arm,
        "total_parameters": total_parameters,
        "parameter_tensors": parameter_tensors,
        "model_buffer_values": model_buffer_values,
        "model_buffer_tensors": model_buffer_tensors,
        "state_dict_values": state_values,
        "state_dict_bytes": state_bytes,
        "optimizer_state_bytes": optimizer_state_bytes(optimizer),
        "dense_attention_values_per_layer": [
            module.dense_serialized_values().numel() for module in modules
        ],
        "pivot_metadata_bits": 0,
        "pivot_rule": "column equals RoPE pair index",
        "runtime_scope": (
            "quality pilot; confirmed incremental H100 executor only, not full production admission"
        ),
        "evaluations": evaluations,
        "attention_diagnostics": diagnostics,
        "serving_diagnostics": serving_diagnostics,
        "terminal_serving_bridge": serving_bridge,
        "terminal_artifacts": terminal_artifacts,
        "terminal_ablations": ablations,
        "train": {
            "prediction_tokens": prediction_tokens,
            "mean_loss": float(np.mean(losses)),
            "final_loss": losses[-1],
            "max_loss": max(losses),
            "max_preclip_gradient_norm": max(gradient_norms),
            "elapsed_seconds": sum(durations),
            "tokens_per_second": prediction_tokens / sum(durations),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "nonfinite": False,
        },
    }
    del optimizer, modules, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def decide(results: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if set(results) != set(ARMS):
        return {"complete": False, "missing_arms": sorted(set(ARMS) - set(results))}
    terminal = str(args.steps)
    losses = {
        arm: result["evaluations"][terminal]["loss"]
        for arm, result in results.items()
    }
    baseline = results["raw_baseline"]
    control = results["canonical_bilinear_control"]
    candidate = results["gauge_zero_g1"]
    initial_control_vs_raw = paired_loss_interval(control, baseline, "0")
    terminal_control_vs_raw = paired_loss_interval(control, baseline, terminal)
    candidate_vs_raw = paired_loss_interval(candidate, baseline, terminal)
    candidate_vs_control = paired_loss_interval(candidate, control, terminal)
    full = {"evaluations": {terminal: candidate["evaluations"][terminal]}}
    shear_disabled = {"evaluations": {
        terminal: candidate["terminal_ablations"]["disable_shear"]
    }}
    full_vs_shear_disabled = paired_loss_interval(full, shear_disabled, terminal)
    all_diagnostics = [
        value
        for result in results.values()
        for phase in ("initial", "terminal")
        for value in result["attention_diagnostics"][phase].values()
    ]
    finite_metrics = [
        value
        for result in results.values()
        for evaluation in result["evaluations"].values()
        for value in [evaluation["loss"], *evaluation["per_batch_loss"]]
    ]
    finite_metrics.extend(all_diagnostics)
    finite_metrics.extend(
        value
        for interval in (
            initial_control_vs_raw,
            terminal_control_vs_raw,
            candidate_vs_raw,
            candidate_vs_control,
            full_vs_shear_disabled,
        )
        for value in interval.values()
    )
    gates = {
        "exact_frozen_parameter_count": all(
            result["total_parameters"] == 37_758_336
            for result in results.values()
        ),
        "all_parameter_counts_equal": len({
            result["total_parameters"] for result in results.values()
        }) == 1,
        "all_parameter_tensor_counts_equal": len({
            result["parameter_tensors"] for result in results.values()
        }) == 1,
        "all_model_buffer_value_counts_equal": len({
            result["model_buffer_values"] for result in results.values()
        }) == 1,
        "all_model_buffer_tensor_counts_equal": len({
            result["model_buffer_tensors"] for result in results.values()
        }) == 1,
        "all_state_dict_value_counts_equal": len({
            result["state_dict_values"] for result in results.values()
        }) == 1,
        "all_state_dict_bytes_equal": len({
            result["state_dict_bytes"] for result in results.values()
        }) == 1,
        "all_optimizer_state_bytes_equal": len({
            result["optimizer_state_bytes"] for result in results.values()
        }) == 1,
        "all_reported_metrics_finite": (
            all(not result["train"]["nonfinite"] for result in results.values())
            and all(math.isfinite(value) for value in finite_metrics)
        ),
        "all_key_nonfinite_fractions_zero": all(
            result["attention_diagnostics"][phase]["key_nonfinite_fraction_layer_max"] == 0.0
            for result in results.values()
            for phase in ("initial", "terminal")
        ),
        "canonical_initial_nll_matches_raw_within_1e_4": abs(
            control["evaluations"]["0"]["loss"]
            - baseline["evaluations"]["0"]["loss"]
        ) <= 1e-4,
        "canonical_initial_paired_interval_within_1e_4": (
            initial_control_vs_raw["lower_95"] >= -1e-4
            and initial_control_vs_raw["upper_95"] <= 1e-4
        ),
        "candidate_initial_nll_and_batches_bit_exact_to_control": (
            candidate["evaluations"]["0"]["loss"]
            == control["evaluations"]["0"]["loss"]
            and candidate["evaluations"]["0"]["per_batch_loss"]
            == control["evaluations"]["0"]["per_batch_loss"]
        ),
        "control_terminal_noninferior_raw_within_0_05_percent": (
            terminal_control_vs_raw["upper_95"]
            <= 0.0005 * losses["raw_baseline"]
        ),
        "candidate_vs_raw_upper_interval_clears_0_025_percent": (
            candidate_vs_raw["upper_95"]
            <= -0.00025 * losses["raw_baseline"]
        ),
        "candidate_vs_control_upper_interval_clears_0_025_percent": (
            candidate_vs_control["upper_95"]
            <= -0.00025 * losses["canonical_bilinear_control"]
        ),
        "disabling_shear_hurts_0_01_percent": (
            shear_disabled["evaluations"][terminal]["loss"] - losses["gauge_zero_g1"]
        ) / losses["gauge_zero_g1"] >= 0.0001,
        "full_vs_shear_disabled_upper_interval_clears_0_01_percent": (
            full_vs_shear_disabled["upper_95"]
            <= -0.0001 * losses["gauge_zero_g1"]
        ),
        "bf16_export_coefficient_bit_exact": candidate["serving_diagnostics"]["terminal"]["bf16_export_coefficient_bit_exact"],
        "some_bf16_coefficients_survive": candidate["serving_diagnostics"]["terminal"]["coefficient_bf16_nonzero_fraction"] > 0.0,
        "coefficient_abs_max_at_most_0_5": candidate["serving_diagnostics"]["terminal"]["coefficient_bf16_abs_max"] <= 0.5,
        "zero_pivot_metadata_bits": all(
            result["pivot_metadata_bits"] == 0 for result in results.values()
        ),
        "trained_candidate_matches_fused_h100_export_bit_exact": (
            candidate["terminal_serving_bridge"]["layers"] == 12
            and candidate["terminal_serving_bridge"]["all_layers_bf16_bit_exact"]
            and candidate["terminal_serving_bridge"]["max_abs"] == 0.0
        ),
        "formal_terminal_artifacts_saved_and_hashed": (
            not args.formal
            or (
                all(
                    result["terminal_artifacts"].get("saved")
                    and len(result["terminal_artifacts"].get("checkpoint_sha256", "")) == 64
                    for result in results.values()
                )
                and len(
                    candidate["terminal_artifacts"].get(
                        "bf16_attention_export_sha256", ""
                    )
                ) == 64
            )
        ),
    }
    return {
        "complete": True,
        "terminal_losses": losses,
        "relative_candidate_improvement": {
            reference: (losses[reference] - losses["gauge_zero_g1"])
            / losses[reference]
            for reference in ("raw_baseline", "canonical_bilinear_control")
        },
        "paired_intervals": {
            "initial_control_vs_raw": initial_control_vs_raw,
            "terminal_control_vs_raw": terminal_control_vs_raw,
            "candidate_vs_raw": candidate_vs_raw,
            "candidate_vs_control": candidate_vs_control,
            "full_vs_shear_disabled": full_vs_shear_disabled,
        },
        "gates": gates,
        "pilot_pass": all(gates.values()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps not in args.eval_steps:
        raise ValueError("terminal step must be included in --eval-steps")
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    if args.formal:
        frozen_outputs = [
            args.output,
            *(
                args.checkpoint_dir / f"seed-{args.seed}-{arm}.pt"
                for arm in ARMS
            ),
            (
                args.checkpoint_dir
                / f"seed-{args.seed}-candidate-bf16-dense-attention.pt"
            ),
        ]
        existing = [str(path) for path in frozen_outputs if path.exists()]
        if existing:
            raise FileExistsError(f"formal outputs already exist: {existing}")
    train_file = TokenFile(args.train_file, args.sequence_length)
    validation_file = TokenFile(args.validation_file, args.sequence_length)
    ledger = validate_data_ledger(
        args.data_manifest, args.train_file, args.validation_file, args.sequence_length
    )
    protocol = validate_protocol(args)
    integrity = (
        validate_formal_integrity()
        if args.formal
        else {"valid": False, "checks": {}, "manifest_sha256": None}
    )
    required_train = args.steps * args.gradient_accumulation * args.micro_batch_size
    required_validation = args.eval_batches * args.eval_batch_size
    if (
        train_file.sequence_count < required_train
        or validation_file.sequence_count < required_validation
    ):
        raise ValueError("token files are too short for the experiment protocol")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    payload: dict[str, Any] = {
        "schema": (
            "gauge-zero-g1-lm-discovery-formal-v1"
            if args.formal
            else "gauge-zero-g1-lm-pilot-development-v3"
        ),
        "candidate": "dense-zero-pivot-G1-attention",
        "scope": (
            "matched one-seed 10M-token from-scratch discovery; replication required"
        ),
        "source_sha256": sha256_file(Path(__file__)),
        "attention_source_sha256": sha256_file(ATTENTION_SOURCE),
        "preregistration_sha256": (
            sha256_file(PREREGISTRATION) if args.formal else None
        ),
        "formal_integrity": integrity,
        "experiment_protocol": protocol,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "data_manifest_sha256": sha256_file(args.data_manifest),
        "data_ledger": ledger,
        "device": torch.cuda.get_device_name(0),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "triton": triton.__version__,
        },
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "arms": {},
    }
    for arm in ARMS:
        print(json.dumps({"starting_arm": arm}), flush=True)
        payload["arms"][arm] = train_arm(
            arm, args, train_file, validation_file, device
        )
        payload["decision"] = decide(payload["arms"], args)
        if args.formal and payload["decision"].get("complete"):
            payload["decision"]["gates"]["formal_integrity_protocol_and_data_valid"] = (
                integrity["valid"] and protocol["valid"] and ledger["valid"]
            )
            payload["decision"]["pilot_pass"] = all(
                payload["decision"]["gates"].values()
            )
        write_payload(args.output, payload)
    return payload


def parse_steps(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("data/block-algebra-scratch/train.uint16.bin"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/block-algebra-scratch/validation.uint16.bin"))
    parser.add_argument("--data-manifest", type=Path, default=DATA_MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--steps", type=int, default=305)
    parser.add_argument("--eval-steps", type=parse_steps, default=[61, 305])
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2207)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("results/gauge-zero-g1-lm-pilot-checkpoints"),
    )
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
