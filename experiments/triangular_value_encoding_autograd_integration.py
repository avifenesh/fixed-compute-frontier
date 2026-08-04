#!/usr/bin/env python3
"""Three-seed full-model integration gate for the TVE custom backward."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time
import types
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import transformers
import triton

import experiments.triangular_value_encoding_lm_pilot as pilot
from experiments.triangular_value_encoding_autograd import (
    triangular_value_encoding_autograd,
)


OUTPUT = Path("results/triangular-value-encoding-autograd-integration.json")
CHECKPOINT_DIR = Path(
    "results/triangular-value-encoding-autograd-integration-checkpoints"
)
PREREGISTRATION = Path(
    "results/triangular-value-encoding-autograd-integration-preregistration.md"
)
MANIFEST = Path(
    "results/triangular-value-encoding-autograd-integration-integrity-manifest.json"
)
TRAIN_FILE = Path("data/self-product-ffn-scale/train.uint16.bin")
VALIDATION_FILE = Path("data/self-product-ffn-scale/validation.uint16.bin")
DATA_MANIFEST = Path("results/self-product-ffn-scale-data-manifest.json")
AUTOGRAD_V2_RESULT = Path("results/triangular-value-encoding-autograd-v2.json")
AUTOGRAD_V2_MANIFEST = Path(
    "results/triangular-value-encoding-autograd-v2-integrity-manifest.json"
)
H100_V6_RESULT = Path("results/triangular-value-encoding-cache-h100-formal-v6.json")
H100_V6_MANIFEST = Path(
    "results/triangular-value-encoding-cache-h100-integrity-manifest-v6.json"
)

SEEDS = (17107, 18211, 19319)
ARMS = ("canonical_baseline", "reference_tve", "custom_tve")
ARM_ORDERS = {
    17107: ("canonical_baseline", "reference_tve", "custom_tve"),
    18211: ("reference_tve", "custom_tve", "canonical_baseline"),
    19319: ("custom_tve", "canonical_baseline", "reference_tve"),
}
EVAL_STEPS = (61, 152, 305)
STEPS = 305
T_CRITICAL_95_DF2 = 4.302652729911275

DEPENDENCIES = {
    "integration_source": Path(__file__),
    "preregistration": PREREGISTRATION,
    "pilot_source": Path("experiments/triangular_value_encoding_lm_pilot.py"),
    "attention_source": Path("experiments/triangular_value_encoding_attention.py"),
    "autograd_source": Path("experiments/triangular_value_encoding_autograd.py"),
    "autograd_test": Path("tests/test_triangular_value_encoding_autograd.py"),
    "autograd_v2_source": Path(
        "experiments/triangular_value_encoding_autograd_v2_gate.py"
    ),
    "autograd_v2_preregistration": Path(
        "results/triangular-value-encoding-autograd-v2-preregistration.md"
    ),
    "autograd_v2_manifest": AUTOGRAD_V2_MANIFEST,
    "autograd_v2_result": AUTOGRAD_V2_RESULT,
    "h100_v6_result": H100_V6_RESULT,
    "h100_v6_manifest": H100_V6_MANIFEST,
    "data_manifest": DATA_MANIFEST,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_integrity() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text())
    checks = {
        name: manifest.get(name + "_sha256") == sha256_file(path)
        for name, path in DEPENDENCIES.items()
    }
    v2 = json.loads(AUTOGRAD_V2_RESULT.read_text())
    h100 = json.loads(H100_V6_RESULT.read_text())
    checks.update({
        "python_version": manifest.get("python_version") == platform.python_version(),
        "numpy_version": manifest.get("numpy_version") == np.__version__,
        "torch_version": manifest.get("torch_version") == torch.__version__,
        "cuda_version": manifest.get("cuda_version") == torch.version.cuda,
        "transformers_version": (
            manifest.get("transformers_version") == transformers.__version__
        ),
        "triton_version": manifest.get("triton_version") == triton.__version__,
        "autograd_v2_all_gates_pass": v2.get("all_gates_pass") is True,
        "autograd_v2_bound_manifest": (
            v2.get("integrity_manifest_sha256") == sha256_file(AUTOGRAD_V2_MANIFEST)
        ),
        "h100_v6_all_gates_pass": h100.get("all_gates_pass") is True,
        "h100_v6_bound_manifest": (
            h100.get("integrity_manifest_sha256") == sha256_file(H100_V6_MANIFEST)
        ),
    })
    if not all(checks.values()):
        raise ValueError(f"invalid integration integrity: {checks}")
    return {
        "valid": True,
        "checks": checks,
        "manifest_sha256": sha256_file(MANIFEST),
    }


def validate_protocol(args: argparse.Namespace) -> dict[str, Any]:
    config = pilot.scratch_config()
    expected = {
        "device_contains": "H100",
        "train_file": str(TRAIN_FILE),
        "validation_file": str(VALIDATION_FILE),
        "data_manifest": str(DATA_MANIFEST),
        "output": str(OUTPUT),
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "seeds": list(SEEDS),
        "arm_orders": {str(key): list(value) for key, value in ARM_ORDERS.items()},
        "sequence_length": 512,
        "micro_batch_size": 32,
        "gradient_accumulation": 2,
        "eval_batch_size": 32,
        "eval_batches": 64,
        "steps": STEPS,
        "eval_steps": list(EVAL_STEPS),
        "compile_warmups": 3,
        "timing_start_step": 11,
        "warmup_steps": 50,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
        "prediction_tokens": 9_994_240,
        "formal": True,
        "hidden_size": 384,
        "layers": 12,
        "query_heads": 6,
        "kv_heads": 2,
        "head_dim": 64,
        "intermediate_size": 1024,
    }
    actual = {
        "device_contains": torch.cuda.get_device_name(0),
        "train_file": str(args.train_file),
        "validation_file": str(args.validation_file),
        "data_manifest": str(args.data_manifest),
        "output": str(args.output),
        "checkpoint_dir": str(args.checkpoint_dir),
        "seeds": args.seeds,
        "arm_orders": {str(key): list(value) for key, value in ARM_ORDERS.items()},
        "sequence_length": args.sequence_length,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "eval_batch_size": args.eval_batch_size,
        "eval_batches": args.eval_batches,
        "steps": args.steps,
        "eval_steps": args.eval_steps,
        "compile_warmups": args.compile_warmups,
        "timing_start_step": args.timing_start_step,
        "warmup_steps": args.warmup_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "prediction_tokens": (
            args.steps * args.gradient_accumulation * args.micro_batch_size
            * args.sequence_length
        ),
        "formal": args.formal,
        "hidden_size": config.hidden_size,
        "layers": config.num_hidden_layers,
        "query_heads": config.num_attention_heads,
        "kv_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "intermediate_size": config.intermediate_size,
    }
    checks = {
        key: (value in actual[key] if key == "device_contains" else actual[key] == value)
        for key, value in expected.items()
    }
    if not all(checks.values()):
        raise ValueError(f"invalid integration protocol: {checks}")
    return {"valid": True, "checks": checks, "expected": expected, "actual": actual}


def reset_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _custom_apply_value_encoding(self, values: torch.Tensor) -> torch.Tensor:
    if self.arm != "triangular_value_encoding":
        return values
    return triangular_value_encoding_autograd(
        values,
        self.output_weight,
        query_groups=self.num_key_value_groups,
        block_size=self.block_size,
        tau=self.tau,
    )


def build_experiment_model(
    device: torch.device, arm: str
) -> tuple[torch.nn.Module, list[Any]]:
    if arm not in ARMS:
        raise ValueError(arm)
    architecture_arm = (
        "canonical_value_control"
        if arm == "canonical_baseline"
        else "triangular_value_encoding"
    )
    model, modules = pilot.build_model(device, architecture_arm)
    if arm == "custom_tve":
        for module in modules:
            module.apply_value_encoding = types.MethodType(
                _custom_apply_value_encoding, module
            )
    return model, modules


def state_dict_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def vector_metrics(actual: list[torch.Tensor], expected: list[torch.Tensor]) -> dict[str, float]:
    if len(actual) != len(expected):
        raise ValueError("gradient tensor count mismatch")
    difference_sq = 0.0
    actual_sq = 0.0
    expected_sq = 0.0
    dot = 0.0
    max_abs = 0.0
    values = 0
    for left, right in zip(actual, expected):
        if left.shape != right.shape:
            raise ValueError("gradient shape mismatch")
        left64 = left.detach().double()
        right64 = right.detach().double()
        delta = left64 - right64
        difference_sq += float(torch.sum(delta.square()))
        actual_sq += float(torch.sum(left64.square()))
        expected_sq += float(torch.sum(right64.square()))
        dot += float(torch.sum(left64 * right64))
        max_abs = max(max_abs, float(delta.abs().max()))
        values += left.numel()
    tiny = torch.finfo(torch.float64).tiny
    return {
        "values": values,
        "relative_l2": math.sqrt(difference_sq) / max(math.sqrt(expected_sq), tiny),
        "cosine": dot / max(math.sqrt(actual_sq * expected_sq), tiny),
        "rmse": math.sqrt(difference_sq / values),
        "max_abs": max_abs,
    }


def first_batch_loss_and_logits(
    model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=inputs, use_cache=False).logits
    loss = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1)
    )
    return loss, logits


def first_backward_probe(
    seed: int,
    train_file: Any,
    validation_file: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    reset_seed(seed)
    reference_model, reference_modules = build_experiment_model(device, "reference_tve")
    reset_seed(seed)
    custom_model, custom_modules = build_experiment_model(device, "custom_tve")
    reference_hash = state_dict_sha256(reference_model)
    custom_hash = state_dict_sha256(custom_model)
    reference_evaluation = pilot.evaluate(
        reference_model, validation_file, args.eval_batches, args.eval_batch_size, device
    )
    custom_evaluation = pilot.evaluate(
        custom_model, validation_file, args.eval_batches, args.eval_batch_size, device
    )
    inputs, targets = train_file.batch(0, args.micro_batch_size, device)
    reference_loss, reference_logits = first_batch_loss_and_logits(
        reference_model, inputs, targets
    )
    custom_loss, custom_logits = first_batch_loss_and_logits(custom_model, inputs, targets)
    reference_loss.backward()
    custom_loss.backward()
    reference_gradients = [
        parameter.grad for parameter in reference_model.parameters()
        if parameter.grad is not None
    ]
    custom_gradients = [
        parameter.grad for parameter in custom_model.parameters()
        if parameter.grad is not None
    ]
    full_gradient = vector_metrics(custom_gradients, reference_gradients)
    per_layer = []
    outside_active_exact = True
    for index, (custom_module, reference_module) in enumerate(
        zip(custom_modules, reference_modules)
    ):
        hidden = custom_module.hidden_size
        mask = torch.zeros(
            hidden, hidden, device=device, dtype=torch.bool
        )
        for kv_head in range(custom_module.kv_heads):
            representative = (
                kv_head * custom_module.num_key_value_groups * custom_module.head_dim
            )
            for start in range(0, custom_module.head_dim, custom_module.block_size):
                for target in range(1, custom_module.block_size):
                    mask[
                        start + target,
                        representative + start:representative + start + target,
                    ] = True
        custom_output = custom_module.output_weight.grad
        reference_output = reference_module.output_weight.grad
        outside_exact = torch.equal(custom_output[~mask], reference_output[~mask])
        outside_active_exact = outside_active_exact and outside_exact
        per_layer.append({
            "layer": index,
            "active_output_slots": int(mask.sum()),
            "active_output_gradient": vector_metrics(
                [custom_output[mask]], [reference_output[mask]]
            ),
            "outside_active_output_bit_exact": outside_exact,
            "value_weight_gradient": vector_metrics(
                [custom_module.value_weight.grad],
                [reference_module.value_weight.grad],
            ),
        })
    finite = all(
        torch.isfinite(gradient).all().item()
        for gradient in (*reference_gradients, *custom_gradients)
    )
    gates = {
        "initial_state_bit_exact": reference_hash == custom_hash,
        "initial_per_batch_nll_bit_exact": (
            reference_evaluation["per_batch_loss"]
            == custom_evaluation["per_batch_loss"]
        ),
        "first_batch_logits_bit_exact": torch.equal(reference_logits, custom_logits),
        "first_batch_loss_bit_exact": float(reference_loss) == float(custom_loss),
        "full_gradient_relative_l2_at_most_0_005": (
            full_gradient["relative_l2"] <= 0.005
        ),
        "full_gradient_cosine_at_least_0_99998": (
            full_gradient["cosine"] >= 0.99998
        ),
        "outside_active_output_slots_bit_exact": outside_active_exact,
        "exactly_12_layers_compared": len(per_layer) == 12,
        "all_layers_have_exactly_960_active_output_slots": all(
            layer["active_output_slots"] == 960 for layer in per_layer
        ),
        "all_layer_active_output_gradient_metrics_pass": all(
            layer["active_output_gradient"]["relative_l2"] <= 0.005
            and layer["active_output_gradient"]["cosine"] >= 0.99998
            and layer["active_output_gradient"]["rmse"] <= 0.05
            for layer in per_layer
        ),
        "all_layer_value_weight_gradient_direction_metrics_pass": all(
            layer["value_weight_gradient"]["relative_l2"] <= 0.005
            and layer["value_weight_gradient"]["cosine"] >= 0.99998
            for layer in per_layer
        ),
        "all_gradients_finite": finite,
    }
    result = {
        "initial_state_sha256": {
            "reference_tve": reference_hash,
            "custom_tve": custom_hash,
        },
        "initial_evaluations": {
            "reference_tve": reference_evaluation,
            "custom_tve": custom_evaluation,
        },
        "first_batch": {
            "reference_loss": float(reference_loss),
            "custom_loss": float(custom_loss),
            "logits_shape": list(reference_logits.shape),
        },
        "full_gradient": full_gradient,
        "per_layer": per_layer,
        "gates": gates,
        "pass": all(gates.values()),
    }
    del reference_logits, custom_logits, reference_model, custom_model
    del reference_modules, custom_modules, reference_gradients, custom_gradients
    gc.collect()
    torch.cuda.empty_cache()
    return result


def warmup_model(
    model: torch.nn.Module,
    train_file: Any,
    args: argparse.Namespace,
) -> None:
    model.train()
    for warmup in range(args.compile_warmups):
        inputs, targets = train_file.batch(warmup, args.micro_batch_size, torch.device("cuda"))
        loss = pilot.causal_loss(model, inputs, targets)
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()


def save_artifacts(
    model: torch.nn.Module,
    modules: list[Any],
    seed: int,
    arm: str,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"seed-{seed}-{arm}.pt"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    torch.save({
        "arm": arm,
        "seed": seed,
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
    }, checkpoint)
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    if arm in {"reference_tve", "custom_tve"}:
        export = checkpoint_dir / f"seed-{seed}-{arm}-bf16-attention.pt"
        if export.exists():
            raise FileExistsError(export)
        torch.save({
            "arm": arm,
            "seed": seed,
            "layers": [
                {
                    key: (
                        value.detach().to(torch.bfloat16).cpu()
                        if isinstance(value, torch.Tensor) else value
                    )
                    for key, value in module.export_dense().items()
                }
                for module in modules
            ],
        }, export)
        result.update({
            "bf16_attention_export": str(export),
            "bf16_attention_export_bytes": export.stat().st_size,
            "bf16_attention_export_sha256": sha256_file(export),
        })
    return result


def train_arm(
    arm: str,
    seed: int,
    args: argparse.Namespace,
    train_file: Any,
    validation_file: Any,
    device: torch.device,
) -> dict[str, Any]:
    reset_seed(seed)
    model, modules = build_experiment_model(device, arm)
    initial_state_hash = state_dict_sha256(model)
    optimizer = pilot.optimizer_for(model, args.learning_rate, args.weight_decay)
    warmup_model(model, train_file, args)
    evaluations = {
        "0": pilot.evaluate(
            model, validation_file, args.eval_batches, args.eval_batch_size, device
        )
    }
    initial_diagnostics = pilot.attention_diagnostics(
        model, modules, validation_file, device
    )
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    step_losses: list[float] = []
    step_seconds: list[float] = []
    gradient_norms: list[float] = []
    for step in range(args.steps):
        model.train()
        multiplier = pilot.lr_multiplier(step, args.steps, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * multiplier
        prepared_batches = [
            train_file.batch(
                step * args.gradient_accumulation + micro,
                args.micro_batch_size,
                device,
            )
            for micro in range(args.gradient_accumulation)
        ]
        started = time.perf_counter()
        accumulated_tensor = None
        for inputs, targets in prepared_batches:
            loss = pilot.causal_loss(model, inputs, targets)
            (loss / args.gradient_accumulation).backward()
            contribution = loss.detach() / args.gradient_accumulation
            accumulated_tensor = (
                contribution if accumulated_tensor is None
                else accumulated_tensor + contribution
            )
        gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_clip
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        duration = time.perf_counter() - started
        if accumulated_tensor is None:
            raise RuntimeError("no micro-batches executed")
        accumulated = float(accumulated_tensor)
        gradient_norm = float(gradient_norm_tensor)
        if not math.isfinite(accumulated) or not math.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite {arm} seed {seed} step {step + 1}")
        step_losses.append(accumulated)
        step_seconds.append(duration)
        gradient_norms.append(gradient_norm)
        if step + 1 in args.eval_steps:
            evaluations[str(step + 1)] = pilot.evaluate(
                model, validation_file, args.eval_batches, args.eval_batch_size, device
            )
            print(json.dumps({
                "integration_seed": seed,
                "arm": arm,
                "step": step + 1,
                "train_loss": accumulated,
                "validation_loss": evaluations[str(step + 1)]["loss"],
                "gradient_norm": gradient_norm,
            }), flush=True)
    training_peak_allocated_bytes = torch.cuda.max_memory_allocated()
    terminal_diagnostics = pilot.attention_diagnostics(
        model, modules, validation_file, device
    )
    serving = pilot.coefficient_diagnostics(modules)
    bridge = (
        pilot.terminal_serving_bridge(modules)
        if arm in {"reference_tve", "custom_tve"}
        else {"layers": 0, "all_layers_bit_exact": True, "max_abs": 0.0}
    )
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    state = model.state_dict()
    artifacts = save_artifacts(model, modules, seed, arm, args.checkpoint_dir)
    result = {
        "arm": arm,
        "initial_state_sha256": initial_state_hash,
        "total_parameters": sum(value.numel() for value in parameters),
        "parameter_tensors": len(parameters),
        "model_buffer_values": sum(value.numel() for value in buffers),
        "model_buffer_tensors": len(buffers),
        "state_dict_values": sum(value.numel() for value in state.values()),
        "state_dict_bytes": sum(
            value.numel() * value.element_size() for value in state.values()
        ),
        "optimizer_state_bytes": pilot.optimizer_state_bytes(optimizer),
        "metadata_bits": 0,
        "evaluations": evaluations,
        "attention_diagnostics": {
            "initial": initial_diagnostics,
            "terminal": terminal_diagnostics,
        },
        "serving_diagnostics": serving,
        "terminal_serving_bridge": bridge,
        "terminal_artifacts": artifacts,
        "train": {
            "step_losses": step_losses,
            "step_seconds": step_seconds,
            "gradient_norms": gradient_norms,
            "prediction_tokens": (
                args.steps * args.gradient_accumulation * args.micro_batch_size
                * args.sequence_length
            ),
            "timing_start_step": args.timing_start_step,
            "timed_step_median_seconds": float(
                np.median(step_seconds[args.timing_start_step - 1:])
            ),
            "elapsed_seconds": float(sum(step_seconds)),
            "tokens_per_second": (
                args.steps * args.gradient_accumulation * args.micro_batch_size
                * args.sequence_length / sum(step_seconds)
            ),
            "peak_allocated_bytes": training_peak_allocated_bytes,
            "nonfinite": False,
        },
    }
    del optimizer, modules, model, parameters, buffers, state
    gc.collect()
    torch.cuda.empty_cache()
    return result


def seed_interval(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError("three seed values required")
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / math.sqrt(3))
    radius = T_CRITICAL_95_DF2 * standard_error
    return {
        "seed_means": array.tolist(),
        "mean": mean,
        "standard_error": standard_error,
        "lower_95": mean - radius,
        "upper_95": mean + radius,
        "t_critical_95_df2": T_CRITICAL_95_DF2,
    }


def paired_validation_mean(left: dict[str, Any], right: dict[str, Any], step: str) -> float:
    left_values = np.asarray(left["evaluations"][step]["per_batch_loss"])
    right_values = np.asarray(right["evaluations"][step]["per_batch_loss"])
    if left_values.shape != (64,) or right_values.shape != (64,):
        raise ValueError("64 paired validation batches required")
    return float((left_values - right_values).mean())


def artifact_valid(
    result: dict[str, Any], seed: int, arm: str, checkpoint_dir: Path
) -> bool:
    artifact = result["terminal_artifacts"]
    expected_checkpoint = checkpoint_dir / f"seed-{seed}-{arm}.pt"
    checkpoint = Path(artifact["checkpoint"])
    valid = (
        result["arm"] == arm
        and checkpoint == expected_checkpoint
        and checkpoint.exists()
        and checkpoint.stat().st_size == artifact["checkpoint_bytes"]
        and sha256_file(checkpoint) == artifact["checkpoint_sha256"]
    )
    if result["arm"] == "canonical_baseline":
        return valid
    expected_export = (
        checkpoint_dir
        / f"seed-{seed}-{arm}-bf16-attention.pt"
    )
    export = Path(artifact["bf16_attention_export"])
    return valid and (
        export == expected_export
        and export.exists()
        and export.stat().st_size == artifact["bf16_attention_export_bytes"]
        and sha256_file(export) == artifact["bf16_attention_export_sha256"]
    )


def decide(
    seed_results: dict[str, dict[str, Any]],
    probes: dict[str, Any],
    integrity: dict[str, Any],
    protocol: dict[str, Any],
    ledger: dict[str, Any],
) -> dict[str, Any]:
    if set(seed_results) != {str(seed) for seed in SEEDS} or any(
        set(results) != set(ARMS) for results in seed_results.values()
    ):
        return {"complete": False, "completed_seeds": sorted(seed_results)}

    equivalence = {}
    equivalence_ok = True
    for step in EVAL_STEPS:
        key = str(step)
        differences = [
            paired_validation_mean(
                seed_results[str(seed)]["custom_tve"],
                seed_results[str(seed)]["reference_tve"],
                key,
            )
            for seed in SEEDS
        ]
        interval = seed_interval(differences)
        mean_reference = float(np.mean([
            seed_results[str(seed)]["reference_tve"]["evaluations"][key]["loss"]
            for seed in SEEDS
        ]))
        bound = 0.0002 * mean_reference
        step_ok = interval["lower_95"] >= -bound and interval["upper_95"] <= bound
        equivalence_ok = equivalence_ok and step_ok
        equivalence[key] = {
            "custom_minus_reference": interval,
            "mean_reference_nll": mean_reference,
            "absolute_bound": bound,
            "pass": step_ok,
        }

    trajectory = {}
    trajectory_ok = True
    for seed in SEEDS:
        results = seed_results[str(seed)]
        reference = np.asarray(results["reference_tve"]["train"]["step_losses"])
        custom = np.asarray(results["custom_tve"]["train"]["step_losses"])
        difference = custom - reference
        mean_reference = float(reference.mean())
        rmse_relative = float(np.sqrt(np.mean(difference ** 2)) / mean_reference)
        bias_relative = float(abs(difference.mean()) / mean_reference)
        passed = rmse_relative <= 0.0005 and bias_relative <= 0.0002
        trajectory_ok = trajectory_ok and passed
        trajectory[str(seed)] = {
            "rmse_relative_to_reference_mean": rmse_relative,
            "absolute_bias_relative_to_reference_mean": bias_relative,
            "pass": passed,
        }

    terminal_reference = float(np.mean([
        seed_results[str(seed)]["reference_tve"]["evaluations"][str(STEPS)]["loss"]
        for seed in SEEDS
    ]))
    terminal_custom = float(np.mean([
        seed_results[str(seed)]["custom_tve"]["evaluations"][str(STEPS)]["loss"]
        for seed in SEEDS
    ]))
    terminal_relative_difference = abs(terminal_custom - terminal_reference) / terminal_reference

    architecture = {}
    architecture_ok = True
    mean_canonical = float(np.mean([
        seed_results[str(seed)]["canonical_baseline"]["evaluations"][str(STEPS)]["loss"]
        for seed in SEEDS
    ]))
    for arm in ("reference_tve", "custom_tve"):
        values = [
            paired_validation_mean(
                seed_results[str(seed)][arm],
                seed_results[str(seed)]["canonical_baseline"],
                str(STEPS),
            )
            for seed in SEEDS
        ]
        interval = seed_interval(values)
        passed = (
            all(value < 0 for value in values)
            and interval["upper_95"] <= -0.00025 * mean_canonical
        )
        architecture_ok = architecture_ok and passed
        architecture[arm] = {"candidate_minus_canonical": interval, "pass": passed}

    timing = {}
    ratios = []
    recoveries = []
    every_ratio = True
    every_recovery = True
    memory_ok = True
    for seed in SEEDS:
        results = seed_results[str(seed)]
        medians = {
            arm: results[arm]["train"]["timed_step_median_seconds"] for arm in ARMS
        }
        ratio = medians["custom_tve"] / medians["reference_tve"]
        denominator = medians["reference_tve"] - medians["canonical_baseline"]
        recovery = (
            (medians["reference_tve"] - medians["custom_tve"]) / denominator
            if denominator > 0 else float("-inf")
        )
        seed_memory_ok = (
            results["custom_tve"]["train"]["peak_allocated_bytes"]
            <= results["reference_tve"]["train"]["peak_allocated_bytes"]
        )
        ratios.append(ratio)
        recoveries.append(recovery)
        every_ratio = every_ratio and ratio <= 0.95
        every_recovery = every_recovery and denominator > 0 and recovery >= 0.10
        memory_ok = memory_ok and seed_memory_ok
        timing[str(seed)] = {
            "median_step_seconds": medians,
            "custom_over_reference": ratio,
            "reference_overhead_seconds": denominator,
            "recovered_overhead": recovery,
            "custom_peak_not_above_reference": seed_memory_ok,
        }
    log_ratio_interval = seed_interval([math.log(value) for value in ratios])
    timing_gates = {
        "custom_reference_ratio_at_most_0_95_every_seed": every_ratio,
        "log_ratio_cluster_upper_at_most_log_0_95": (
            log_ratio_interval["upper_95"] <= math.log(0.95)
        ),
        "recovered_overhead_at_least_0_10_every_seed": every_recovery,
        "mean_recovered_overhead_at_least_0_20": float(np.mean(recoveries)) >= 0.20,
        "custom_peak_not_above_reference_every_seed": memory_ok,
    }

    structural_ok = True
    artifact_ok = True
    equal_fields = (
        "total_parameters", "parameter_tensors", "model_buffer_values",
        "model_buffer_tensors", "state_dict_values", "state_dict_bytes",
        "optimizer_state_bytes", "metadata_bits",
    )
    for seed in SEEDS:
        results = seed_results[str(seed)]
        baseline = results["canonical_baseline"]
        structural_ok = structural_ok and baseline["total_parameters"] == 37_758_336
        structural_ok = structural_ok and len({
            results[arm]["initial_state_sha256"] for arm in ARMS
        }) == 1
        structural_ok = structural_ok and all(
            results[arm]["evaluations"]["0"]["per_batch_loss"]
            == baseline["evaluations"]["0"]["per_batch_loss"]
            for arm in ARMS
        )
        for field in equal_fields:
            structural_ok = structural_ok and all(
                results[arm][field] == baseline[field] for arm in ARMS
            )
        for arm in ("reference_tve", "custom_tve"):
            candidate = results[arm]
            structural_ok = structural_ok and (
                candidate["metadata_bits"] == 0
                and candidate["serving_diagnostics"]["coefficient_bf16_nonzero_fraction"] > 0
                and candidate["serving_diagnostics"]["coefficient_bf16_abs_max"] <= 0.5
                and candidate["serving_diagnostics"]["bf16_export_coefficient_bit_exact"]
                and candidate["terminal_serving_bridge"]
                == {"layers": 12, "all_layers_bit_exact": True, "max_abs": 0.0}
            )
        artifact_ok = artifact_ok and all(
            artifact_valid(
                results[arm], seed, arm,
                Path(protocol["actual"]["checkpoint_dir"]),
            )
            for arm in ARMS
        )

    gates = {
        "integrity_protocol_and_data_valid": (
            integrity["valid"] and protocol["valid"] and ledger["valid"]
        ),
        "all_first_backward_probes_pass": all(
            probes[str(seed)]["pass"] for seed in SEEDS
        ),
        "all_evaluation_equivalence_intervals_pass": equivalence_ok,
        "terminal_mean_relative_difference_at_most_0_01_percent": (
            terminal_relative_difference <= 0.0001
        ),
        "all_training_trajectory_equivalence_gates_pass": trajectory_ok,
        "both_tve_arms_retain_architecture_signal": architecture_ok,
        "all_full_model_timing_gates_pass": all(timing_gates.values()),
        "all_structural_gates_pass": structural_ok,
        "all_artifact_hashes_valid": artifact_ok,
    }
    return {
        "complete": True,
        "equivalence_by_step": equivalence,
        "terminal_mean_nll": {
            "reference_tve": terminal_reference,
            "custom_tve": terminal_custom,
            "absolute_relative_difference": terminal_relative_difference,
        },
        "training_trajectory": trajectory,
        "architecture_signal": architecture,
        "timing_by_seed": timing,
        "log_timing_ratio_seed_cluster": log_ratio_interval,
        "mean_recovered_overhead": float(np.mean(recoveries)),
        "timing_gates": timing_gates,
        "gates": gates,
        "integration_pass": all(gates.values()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    expected_artifacts = [
        args.checkpoint_dir / f"seed-{seed}-{arm}.pt"
        for seed in args.seeds for arm in ARMS
    ] + [
        args.checkpoint_dir / f"seed-{seed}-{arm}-bf16-attention.pt"
        for seed in args.seeds for arm in ("reference_tve", "custom_tve")
    ]
    existing = [str(path) for path in expected_artifacts if path.exists()]
    if existing:
        raise FileExistsError(existing)
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    integrity = validate_integrity()
    protocol = validate_protocol(args)
    ledger = pilot.validate_data_ledger(
        args.data_manifest, args.train_file, args.validation_file, args.sequence_length
    )
    train_file = pilot.TokenFile(args.train_file, args.sequence_length)
    validation_file = pilot.TokenFile(args.validation_file, args.sequence_length)
    required_sequences = args.steps * args.gradient_accumulation * args.micro_batch_size
    if (
        train_file.sequence_count < required_sequences
        or validation_file.sequence_count < args.eval_batches * args.eval_batch_size
    ):
        raise ValueError("token files too short")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    payload: dict[str, Any] = {
        "schema": "triangular-value-encoding-autograd-integration-formal-v1",
        "scope": "three-seed 37.8M full-model custom-backward admission gate",
        "source_sha256": sha256_file(Path(__file__)),
        "formal_integrity": integrity,
        "experiment_protocol": protocol,
        "data_ledger": ledger,
        "device": torch.cuda.get_device_name(0),
        "runtime": {
            "python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": transformers.__version__, "triton": triton.__version__,
        },
        "architecture_constants": {"block_size": 16, "tau": 0.125, "metadata_bits": 0},
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "first_backward_probes": {},
        "seeds": {},
        "decision": {"complete": False, "completed_seeds": []},
    }
    for seed in args.seeds:
        print(json.dumps({"starting_first_backward_probe": seed}), flush=True)
        payload["first_backward_probes"][str(seed)] = first_backward_probe(
            seed, train_file, validation_file, args, device
        )
        if not payload["first_backward_probes"][str(seed)]["pass"]:
            pilot.write_payload(args.output, payload)
            raise RuntimeError(f"first-backward probe failed for seed {seed}")
        payload["seeds"][str(seed)] = {}
        for arm in ARM_ORDERS[seed]:
            print(json.dumps({"starting_integration_seed": seed, "arm": arm}), flush=True)
            payload["seeds"][str(seed)][arm] = train_arm(
                arm, seed, args, train_file, validation_file, device
            )
            pilot.write_payload(args.output, payload)
        payload["decision"] = decide(
            payload["seeds"], payload["first_backward_probes"],
            integrity, protocol, ledger,
        )
        pilot.write_payload(args.output, payload)
    return payload


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=TRAIN_FILE)
    parser.add_argument("--validation-file", type=Path, default=VALIDATION_FILE)
    parser.add_argument("--data-manifest", type=Path, default=DATA_MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--seeds", type=parse_ints, default=list(SEEDS))
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--eval-steps", type=parse_ints, default=list(EVAL_STEPS))
    parser.add_argument("--compile-warmups", type=int, default=3)
    parser.add_argument("--timing-start-step", type=int, default=11)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--formal", action="store_true", default=True)
    payload = run(parser.parse_args())
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
