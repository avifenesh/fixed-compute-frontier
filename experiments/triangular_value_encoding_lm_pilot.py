#!/usr/bin/env python3
"""Matched 10M-token LM discovery pilot for triangular value encoding."""

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
import transformers
from transformers import AutoModelForCausalLM

from experiments.gauge_slot_lm_pilot import optimizer_for, optimizer_state_bytes
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
from experiments.triangular_value_encoding_attention import (
    TriangularValueEncodingAttention,
    apply_rotary_fp32_one_store,
    replace_llama_attention,
)
from experiments.triangular_value_encoding_cache_h100 import (
    QK_TOKEN_BLOCK,
    VALUE_TOKEN_BLOCK,
    cache_write_epilogue_kernel,
)


ARMS = (
    "packed_raw_control",
    "canonical_value_control",
    "triangular_value_encoding",
)
OUTPUT = Path("results/triangular-value-encoding-lm-pilot.json")
PREREGISTRATION = Path("results/triangular-value-encoding-lm-preregistration.md")
INTEGRITY_MANIFEST = Path("results/triangular-value-encoding-lm-integrity-manifest.json")
DATA_MANIFEST = Path("results/block-algebra-scratch-data-manifest.json")
H100_RESULT = Path("results/triangular-value-encoding-cache-h100-formal-v4.json")
H100_PREREGISTRATION = Path("results/triangular-value-encoding-cache-h100-preregistration-v4.md")
H100_MANIFEST = Path("results/triangular-value-encoding-cache-h100-integrity-manifest-v4.json")
H100_FORMAL_SOURCE = Path("experiments/triangular_value_encoding_cache_h100_formal_v4.py")
H100_EXECUTOR_SOURCE = Path("experiments/triangular_value_encoding_cache_h100.py")
ATTENTION_SOURCE = Path("experiments/triangular_value_encoding_attention.py")
ATTENTION_TEST = Path("tests/test_triangular_value_encoding_attention.py")
BRIDGE_TEST = Path("tests/test_triangular_value_encoding_cache_h100.py")
LM_TEST = Path("tests/test_triangular_value_encoding_lm_pilot.py")


def validate_protocol(args: argparse.Namespace) -> dict[str, Any]:
    config = scratch_config()
    expected = {
        "device_contains": "H100",
        "python_version": "3.11.10",
        "numpy_version": "2.1.2",
        "torch_version": "2.5.1+cu124",
        "cuda_version": "12.4",
        "transformers_version": "4.57.6",
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
        "seed": 3307,
        "prediction_tokens": 9_994_240,
        "formal": True,
        "output": str(OUTPUT),
        "checkpoint_dir": "results/triangular-value-encoding-lm-pilot-checkpoints",
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
        "prediction_tokens": args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length,
        "formal": args.formal,
        "output": str(args.output),
        "checkpoint_dir": str(args.checkpoint_dir),
        "hidden_size": config.hidden_size,
        "layers": config.num_hidden_layers,
        "query_heads": config.num_attention_heads,
        "kv_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "intermediate_size": config.intermediate_size,
    }
    checks = {
        key: (expected_value in actual[key] if key == "device_contains" else actual[key] == expected_value)
        for key, expected_value in expected.items()
    }
    if args.formal and not all(checks.values()):
        raise ValueError(f"invalid frozen protocol: {checks}")
    return {"valid": all(checks.values()), "checks": checks, "expected": expected, "actual": actual}


def validate_integrity() -> dict[str, Any]:
    manifest = json.loads(INTEGRITY_MANIFEST.read_text())
    paths = {
        "source": Path(__file__),
        "attention_source": ATTENTION_SOURCE,
        "attention_test": ATTENTION_TEST,
        "bridge_test": BRIDGE_TEST,
        "lm_test": LM_TEST,
        "preregistration": PREREGISTRATION,
        "h100_result": H100_RESULT,
        "h100_preregistration": H100_PREREGISTRATION,
        "h100_manifest": H100_MANIFEST,
        "h100_formal_source": H100_FORMAL_SOURCE,
        "h100_executor_source": H100_EXECUTOR_SOURCE,
        "data_manifest": DATA_MANIFEST,
    }
    checks = {
        name: manifest.get(name + "_sha256") == sha256_file(path)
        for name, path in paths.items()
    }
    h100 = json.loads(H100_RESULT.read_text())
    checks.update({
        "h100_all_gates_pass": h100["all_gates_pass"] is True,
        "h100_integrity_valid": all(h100["integrity"].values()),
        "h100_manifest_binding_valid": h100["integrity_manifest_sha256"] == sha256_file(H100_MANIFEST),
    })
    if not all(checks.values()):
        raise ValueError(f"invalid TVE LM integrity: {checks}")
    return {"valid": True, "checks": checks, "manifest_sha256": sha256_file(INTEGRITY_MANIFEST)}


def build_model(device: torch.device, arm: str) -> tuple[torch.nn.Module, list[TriangularValueEncodingAttention]]:
    if arm not in ARMS:
        raise ValueError(arm)
    model = AutoModelForCausalLM.from_config(
        scratch_config(), attn_implementation="sdpa"
    ).to(device)
    model.config.use_cache = False
    modules = replace_llama_attention(model, arm, block_size=16)
    return model, modules


@torch.no_grad()
def attention_diagnostics(
    model: torch.nn.Module,
    modules: list[TriangularValueEncodingAttention],
    validation_file: TokenFile,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    inputs, _ = validation_file.batch(0, 2, device)
    for module in modules:
        module.record_diagnostics = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=inputs, use_cache=False)
    records = [module.last_diagnostics for module in modules]
    for module in modules:
        module.record_diagnostics = False
    if any(record is None for record in records):
        raise RuntimeError("missing TVE diagnostics")
    typed = [record for record in records if record is not None]
    return {
        "coefficient_rms_layer_median": float(np.median([r["coefficient_rms"] for r in typed])),
        "coefficient_abs_max_layer_max": max(r["coefficient_abs_max"] for r in typed),
        "value_delta_rms_layer_median": float(np.median([r["value_delta_rms"] for r in typed])),
        "value_delta_abs_max_layer_max": max(r["value_delta_abs_max"] for r in typed),
        "value_nonfinite_fraction_layer_max": max(r["value_nonfinite_fraction"] for r in typed),
    }


@torch.no_grad()
def coefficient_diagnostics(modules: list[TriangularValueEncodingAttention]) -> dict[str, float | bool]:
    training = torch.cat([module.coefficient_values().float().reshape(-1) for module in modules])
    exported = []
    for module in modules:
        output = module.output_weight.detach().to(torch.bfloat16).reshape(
            module.hidden_size, module.query_heads, module.head_dim
        )
        representative = output[:module.head_dim, ::module.num_key_value_groups, :].permute(1, 0, 2)
        values = []
        for start in range(0, module.head_dim, module.block_size):
            block = representative[:, start:start + module.block_size, start:start + module.block_size]
            values.extend(block[:, row, :row] for row in range(1, module.block_size))
        exported.append(torch.cat(values, -1) / module.tau)
    exported_values = torch.cat(exported).reshape(-1)
    training_bf16 = training.to(torch.bfloat16)
    return {
        "coefficient_bf16_nonzero_fraction": float((exported_values != 0).float().mean()),
        "coefficient_bf16_abs_max": float(exported_values.abs().max()),
        "bf16_export_coefficient_bit_exact": bool(torch.equal(training_bf16, exported_values)),
    }


def cache_executor(
    qkv: torch.Tensor,
    output_weight: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    module: TriangularValueEncodingAttention,
) -> None:
    rows = qkv.shape[0]
    q_width = module.query_heads * module.head_dim
    k_width = module.kv_heads * module.head_dim
    qk_tiles = (rows + QK_TOKEN_BLOCK - 1) // QK_TOKEN_BLOCK
    value_tiles = (rows + VALUE_TOKEN_BLOCK - 1) // VALUE_TOKEN_BLOCK
    tasks = qk_tiles * (module.query_heads + module.kv_heads) + value_tiles * module.kv_heads * module.blocks
    cache_write_epilogue_kernel[(tasks,)](
        qkv, output_weight, cosine, sine, key_cache, value_cache,
        M=rows, N=q_width + 2 * k_width, OUTPUT_WIDTH=module.hidden_size,
        Q_HEADS=module.query_heads, K_HEADS=module.kv_heads,
        HEAD=module.head_dim, HALF_HEAD=module.head_dim // 2,
        Q_WIDTH=q_width, K_WIDTH=k_width,
        GROUP_SIZE=module.num_key_value_groups, BLOCK_SIZE=module.block_size,
        TAU_VALUE=module.tau, QK_TOKEN_BLOCK_SIZE=QK_TOKEN_BLOCK,
        VALUE_TOKEN_BLOCK_SIZE=VALUE_TOKEN_BLOCK, TVE=True, num_warps=4,
    )


@torch.no_grad()
def terminal_serving_bridge(modules: list[TriangularValueEncodingAttention]) -> dict[str, Any]:
    exact, max_abs = [], []
    generator = torch.Generator(device="cuda").manual_seed(1217)
    for module in modules:
        rows = 65
        hidden = torch.randn(1, rows, module.hidden_size, device="cuda", dtype=torch.bfloat16, generator=generator)
        angles = torch.randn(1, rows, module.head_dim // 2, device="cuda", dtype=torch.float32, generator=generator)
        cosine_half, sine_half = angles.cos().to(torch.bfloat16), angles.sin().to(torch.bfloat16)
        cosine = torch.cat((cosine_half, cosine_half), -1)
        sine = torch.cat((sine_half, sine_half), -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            packed = module.project_qkv_packed(hidden)
            q_width = module.query_heads * module.head_dim
            k_width = module.kv_heads * module.head_dim
            q, k, v = packed.split((q_width, k_width, k_width), -1)
            q = q.view(1, rows, module.query_heads, module.head_dim).transpose(1, 2)
            k = k.view(1, rows, module.kv_heads, module.head_dim).transpose(1, 2)
            v = v.view(1, rows, module.kv_heads, module.head_dim).transpose(1, 2)
            q, k = apply_rotary_fp32_one_store(q, k, cosine, sine)
            v = module.apply_value_encoding(v)
        packed_weight = module.packed_qkv_weight().detach().to(torch.bfloat16).contiguous()
        serving = hidden.squeeze(0) @ packed_weight.T
        output_weight = module.output_weight.detach().to(torch.bfloat16).contiguous()
        key_cache = torch.empty(rows, k_width, device="cuda", dtype=torch.bfloat16)
        value_cache = torch.empty_like(key_cache)
        cache_executor(serving, output_weight, cosine_half.squeeze(0), sine_half.squeeze(0), key_cache, value_cache, module)
        prototype = torch.cat((
            q.transpose(1, 2).reshape(rows, q_width),
            k.transpose(1, 2).reshape(rows, k_width),
            v.transpose(1, 2).reshape(rows, k_width),
        ), -1)
        differences = [
            (serving.float() - prototype.float()).abs(),
            (key_cache.float() - prototype[:, q_width:q_width + k_width].float()).abs(),
            (value_cache.float() - prototype[:, q_width + k_width:].float()).abs(),
        ]
        exact.append(all(torch.equal(left, right) for left, right in (
            (serving, prototype),
            (key_cache, prototype[:, q_width:q_width + k_width]),
            (value_cache, prototype[:, q_width + k_width:]),
        )))
        max_abs.append(max(float(value.max()) for value in differences))
    return {"layers": len(modules), "all_layers_bit_exact": all(exact), "max_abs": max(max_abs)}


@torch.no_grad()
def evaluate_disable_encoding(model, modules, validation_file, args, device):
    saved = [module.arm for module in modules]
    try:
        for module in modules:
            module.arm = "canonical_value_control"
        return evaluate(model, validation_file, args.eval_batches, args.eval_batch_size, device)
    finally:
        for module, arm in zip(modules, saved):
            module.arm = arm


def save_artifacts(model, modules, arm, args) -> dict[str, Any]:
    if not args.formal:
        return {"saved": False}
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_dir / f"seed-{args.seed}-{arm}.pt"
    torch.save({"arm": arm, "seed": args.seed, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, checkpoint)
    result = {"saved": True, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size}
    if arm == "triangular_value_encoding":
        export = args.checkpoint_dir / f"seed-{args.seed}-candidate-bf16-attention.pt"
        torch.save({"layers": [{k: (v.detach().to(torch.bfloat16).cpu() if isinstance(v, torch.Tensor) else v) for k, v in module.export_dense().items()} for module in modules]}, export)
        result.update({"bf16_attention_export": str(export), "bf16_attention_export_sha256": sha256_file(export), "bf16_attention_export_bytes": export.stat().st_size})
    return result


def train_arm(arm, args, train_file, validation_file, device) -> dict[str, Any]:
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model, modules = build_model(device, arm)
    parameter_count = sum(p.numel() for p in model.parameters())
    parameter_tensors = len(list(model.parameters()))
    buffers = list(model.buffers())
    state = model.state_dict()
    optimizer = optimizer_for(model, args.learning_rate, args.weight_decay)
    diagnostics = {"initial": attention_diagnostics(model, modules, validation_file, device)}
    serving = {"initial": coefficient_diagnostics(modules)}
    evaluations = {"0": evaluate(model, validation_file, args.eval_batches, args.eval_batch_size, device)}
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    losses, norms, durations = [], [], []
    for step in range(args.steps):
        model.train()
        multiplier = lr_multiplier(step, args.steps, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * multiplier
        started, accumulated = time.perf_counter(), 0.0
        for micro in range(args.gradient_accumulation):
            inputs, targets = train_file.batch(step * args.gradient_accumulation + micro, args.micro_batch_size, device)
            loss = causal_loss(model, inputs, targets)
            (loss / args.gradient_accumulation).backward()
            accumulated += float(loss.detach()) / args.gradient_accumulation
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip))
        if not math.isfinite(accumulated) or not math.isfinite(norm):
            raise RuntimeError(f"non-finite {arm} at {step + 1}")
        optimizer.step(); optimizer.zero_grad(set_to_none=True); torch.cuda.synchronize()
        losses.append(accumulated); norms.append(norm); durations.append(time.perf_counter() - started)
        if step + 1 in args.eval_steps:
            evaluations[str(step + 1)] = evaluate(model, validation_file, args.eval_batches, args.eval_batch_size, device)
            print(json.dumps({"arm": arm, "step": step + 1, "train_loss": accumulated, "validation_loss": evaluations[str(step + 1)]["loss"], "gradient_norm": norm}), flush=True)
    diagnostics["terminal"] = attention_diagnostics(model, modules, validation_file, device)
    serving["terminal"] = coefficient_diagnostics(modules)
    ablations = {}
    bridge = {"layers": 0, "all_layers_bit_exact": True, "max_abs": 0.0}
    if arm == "triangular_value_encoding":
        ablations["disable_encoding"] = evaluate_disable_encoding(model, modules, validation_file, args, device)
        bridge = terminal_serving_bridge(modules)
    artifacts = save_artifacts(model, modules, arm, args)
    prediction_tokens = args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length
    result = {
        "arm": arm,
        "total_parameters": parameter_count,
        "parameter_tensors": parameter_tensors,
        "model_buffer_values": sum(v.numel() for v in buffers),
        "model_buffer_tensors": len(buffers),
        "state_dict_values": sum(v.numel() for v in state.values()),
        "state_dict_bytes": sum(v.numel() * v.element_size() for v in state.values()),
        "optimizer_state_bytes": optimizer_state_bytes(optimizer),
        "dense_attention_values_per_layer": [module.dense_serialized_values().numel() for module in modules],
        "metadata_bits": 0,
        "evaluations": evaluations,
        "attention_diagnostics": diagnostics,
        "serving_diagnostics": serving,
        "terminal_serving_bridge": bridge,
        "terminal_ablations": ablations,
        "terminal_artifacts": artifacts,
        "train": {"prediction_tokens": prediction_tokens, "mean_loss": float(np.mean(losses)), "final_loss": losses[-1], "max_loss": max(losses), "max_preclip_gradient_norm": max(norms), "elapsed_seconds": sum(durations), "tokens_per_second": prediction_tokens / sum(durations), "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "nonfinite": False},
    }
    del optimizer, modules, model
    gc.collect(); torch.cuda.empty_cache()
    return result


def decide(results: dict[str, Any], args) -> dict[str, Any]:
    if set(results) != set(ARMS):
        return {"complete": False, "missing_arms": sorted(set(ARMS) - set(results))}
    terminal = str(args.steps)
    raw, control, candidate = (results[arm] for arm in ARMS)
    losses = {arm: results[arm]["evaluations"][terminal]["loss"] for arm in ARMS}
    initial_control_raw = paired_loss_interval(control, raw, "0")
    terminal_control_raw = paired_loss_interval(control, raw, terminal)
    candidate_raw = paired_loss_interval(candidate, raw, terminal)
    candidate_control = paired_loss_interval(candidate, control, terminal)
    disabled = {"evaluations": {terminal: candidate["terminal_ablations"]["disable_encoding"]}}
    full = {"evaluations": {terminal: candidate["evaluations"][terminal]}}
    full_disabled = paired_loss_interval(full, disabled, terminal)
    finite = [value for result in results.values() for evaluation in result["evaluations"].values() for value in [evaluation["loss"], *evaluation["per_batch_loss"]]]
    gates = {
        "exact_parameter_count": all(r["total_parameters"] == 37_758_336 for r in results.values()),
        "all_parameter_counts_equal": len({r["total_parameters"] for r in results.values()}) == 1,
        "all_parameter_tensor_counts_equal": len({r["parameter_tensors"] for r in results.values()}) == 1,
        "all_buffer_counts_equal": len({(r["model_buffer_values"], r["model_buffer_tensors"]) for r in results.values()}) == 1,
        "all_state_and_optimizer_bytes_equal": len({(r["state_dict_bytes"], r["optimizer_state_bytes"]) for r in results.values()}) == 1,
        "all_metrics_finite": all(math.isfinite(v) for v in finite) and all(not r["train"]["nonfinite"] for r in results.values()),
        "all_value_nonfinite_fractions_zero": all(r["attention_diagnostics"][phase]["value_nonfinite_fraction_layer_max"] == 0 for r in results.values() for phase in ("initial", "terminal")),
        "canonical_initial_nll_matches_raw_within_0_02_percent": abs(control["evaluations"]["0"]["loss"] - raw["evaluations"]["0"]["loss"]) <= 0.0002 * raw["evaluations"]["0"]["loss"],
        "canonical_initial_paired_interval_within_0_02_percent": initial_control_raw["lower_95"] >= -0.0002 * raw["evaluations"]["0"]["loss"] and initial_control_raw["upper_95"] <= 0.0002 * raw["evaluations"]["0"]["loss"],
        "candidate_initial_bit_exact_to_control": candidate["evaluations"]["0"]["loss"] == control["evaluations"]["0"]["loss"] and candidate["evaluations"]["0"]["per_batch_loss"] == control["evaluations"]["0"]["per_batch_loss"],
        "control_terminal_noninferior_raw_within_0_05_percent": terminal_control_raw["upper_95"] <= 0.0005 * losses[ARMS[0]],
        "candidate_vs_raw_clears_0_025_percent": candidate_raw["upper_95"] <= -0.00025 * losses[ARMS[0]],
        "candidate_vs_control_clears_0_025_percent": candidate_control["upper_95"] <= -0.00025 * losses[ARMS[1]],
        "disabling_encoding_hurts_0_01_percent": (disabled["evaluations"][terminal]["loss"] - losses[ARMS[2]]) / losses[ARMS[2]] >= 0.0001,
        "ablation_interval_clears_0_01_percent": full_disabled["upper_95"] <= -0.0001 * losses[ARMS[2]],
        "some_coefficients_survive_bf16": candidate["serving_diagnostics"]["terminal"]["coefficient_bf16_nonzero_fraction"] > 0,
        "coefficients_bounded": candidate["serving_diagnostics"]["terminal"]["coefficient_bf16_abs_max"] <= 0.5,
        "coefficient_export_bit_exact": candidate["serving_diagnostics"]["terminal"]["bf16_export_coefficient_bit_exact"],
        "trained_candidate_serving_bridge_bit_exact": candidate["terminal_serving_bridge"] == {"layers": 12, "all_layers_bit_exact": True, "max_abs": 0.0},
        "zero_metadata": all(r["metadata_bits"] == 0 for r in results.values()),
        "formal_artifacts_saved": not args.formal or all(r["terminal_artifacts"].get("saved") and len(r["terminal_artifacts"].get("checkpoint_sha256", "")) == 64 for r in results.values()),
    }
    return {
        "complete": True,
        "terminal_losses": losses,
        "relative_candidate_improvement": {reference: (losses[reference] - losses[ARMS[2]]) / losses[reference] for reference in ARMS[:2]},
        "paired_intervals": {"initial_control_vs_raw": initial_control_raw, "terminal_control_vs_raw": terminal_control_raw, "candidate_vs_raw": candidate_raw, "candidate_vs_control": candidate_control, "full_vs_disabled": full_disabled},
        "gates": gates,
        "pilot_pass": all(gates.values()),
    }


def run(args) -> dict[str, Any]:
    if args.steps not in args.eval_steps:
        raise ValueError("terminal eval missing")
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    if args.formal:
        frozen = [args.output, *(args.checkpoint_dir / f"seed-{args.seed}-{arm}.pt" for arm in ARMS), args.checkpoint_dir / f"seed-{args.seed}-candidate-bf16-attention.pt"]
        existing = [str(path) for path in frozen if path.exists()]
        if existing:
            raise FileExistsError(existing)
    train_file, validation_file = TokenFile(args.train_file, args.sequence_length), TokenFile(args.validation_file, args.sequence_length)
    ledger = validate_data_ledger(args.data_manifest, args.train_file, args.validation_file, args.sequence_length)
    protocol = validate_protocol(args)
    integrity = validate_integrity() if args.formal else {"valid": False, "checks": {}, "manifest_sha256": None}
    if train_file.sequence_count < args.steps * args.gradient_accumulation * args.micro_batch_size or validation_file.sequence_count < args.eval_batches * args.eval_batch_size:
        raise ValueError("token files too short")
    torch.set_float32_matmul_precision("high"); torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    payload = {
        "schema": "triangular-value-encoding-lm-discovery-formal-v1" if args.formal else "triangular-value-encoding-lm-development-v1",
        "candidate": "gauge-canonical triangular value encoding",
        "scope": "matched one-seed 10M-token from-scratch discovery; replication required",
        "source_sha256": sha256_file(Path(__file__)),
        "attention_source_sha256": sha256_file(ATTENTION_SOURCE),
        "formal_integrity": integrity,
        "experiment_protocol": protocol,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "data_ledger": ledger,
        "device": torch.cuda.get_device_name(0),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__},
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "arms": {},
    }
    for arm in ARMS:
        print(json.dumps({"starting_arm": arm}), flush=True)
        payload["arms"][arm] = train_arm(arm, args, train_file, validation_file, device)
        payload["decision"] = decide(payload["arms"], args)
        if args.formal and payload["decision"].get("complete"):
            payload["decision"]["gates"]["formal_integrity_protocol_and_data_valid"] = integrity["valid"] and protocol["valid"] and ledger["valid"]
            payload["decision"]["pilot_pass"] = all(payload["decision"]["gates"].values())
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
    parser.add_argument("--seed", type=int, default=3307)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/triangular-value-encoding-lm-pilot-checkpoints"))
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args)["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
