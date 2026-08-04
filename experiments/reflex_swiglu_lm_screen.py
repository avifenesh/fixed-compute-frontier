#!/usr/bin/env python3
"""Matched continuation-training screen for Reflex-SwiGLU."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM


MODEL = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
DATASET = "HuggingFaceTB/smollm-corpus"
DATASET_CONFIG = "fineweb-edu-dedup"
DATASET_REVISION = "3ba9d605774198c5868892d7a8deda78031a781f"
VALID_ARMS = ("baseline", "reflex", "reflex_detach", "gate_bias", "gate_temperature")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def effective_alpha(beta: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.tanh(beta / 2.0)


class ScreenMLP(nn.Module):
    """Llama MLP with a parameter-matched activation arm."""

    def __init__(self, original: nn.Module, arm: str) -> None:
        super().__init__()
        if arm not in VALID_ARMS:
            raise ValueError(f"unknown arm {arm}")
        self.arm = arm
        self.gate_proj = original.gate_proj
        self.up_proj = original.up_proj
        self.down_proj = original.down_proj
        self.act_fn = original.act_fn
        self.beta = nn.Parameter(
            torch.zeros(
                self.gate_proj.out_features,
                dtype=self.gate_proj.weight.dtype,
                device=self.gate_proj.weight.device,
            )
        )
        self.record_diagnostics = False
        self.last_diagnostics: dict[str, float | int] | None = None

    def activation(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        z0 = self.act_fn(gate) * up
        alpha = effective_alpha(self.beta).to(dtype=gate.dtype)
        if self.arm == "baseline":
            result = z0 + 0.0 * alpha
        elif self.arm == "reflex":
            result = self.act_fn(gate + alpha * z0.clamp(-1.0, 1.0)) * up
        elif self.arm == "reflex_detach":
            result = self.act_fn(gate + alpha * z0.detach().clamp(-1.0, 1.0)) * up
        elif self.arm == "gate_bias":
            result = self.act_fn(gate + alpha) * up
        elif self.arm == "gate_temperature":
            result = self.act_fn(gate * (1.0 + 0.5 * alpha)) * up
        else:  # pragma: no cover - constructor already validates
            raise AssertionError(self.arm)
        if self.record_diagnostics:
            with torch.no_grad():
                all_u = up.detach().float().reshape(-1)
                sampled_u = all_u[::127]
                flat_z0 = z0.detach().float().reshape(-1)
                self.last_diagnostics = {
                    "coordinates": flat_z0.numel(),
                    "negative_clipped": int((flat_z0 < -1.0).sum()),
                    "positive_clipped": int((flat_z0 > 1.0).sum()),
                    "u_abs_sample_p99": float(torch.quantile(sampled_u.abs(), 0.99)),
                    "u_abs_sample_p999": float(torch.quantile(sampled_u.abs(), 0.999)),
                    "u_abs_true_max": float(all_u.abs().max()),
                }
        return result

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(self.gate_proj(hidden_states), self.up_proj(hidden_states)))


def install_arm(model: nn.Module, arm: str) -> list[ScreenMLP]:
    modules = []
    for layer in model.model.layers:
        replacement = ScreenMLP(layer.mlp, arm)
        layer.mlp = replacement
        modules.append(replacement)
    return modules


class TokenFile:
    def __init__(self, path: Path, sequence_length: int) -> None:
        self.path = path
        self.sequence_length = sequence_length
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")

    @property
    def sequence_count(self) -> int:
        width = self.sequence_length + 1
        if self.tokens.size % width:
            raise ValueError(f"{self.path} is not divisible by sequence width {width}")
        return self.tokens.size // width

    def batch(self, index: int, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        width = self.sequence_length + 1
        first = index * batch_size * width
        last = first + batch_size * width
        if last > self.tokens.size:
            raise IndexError(f"batch {index} exceeds {self.path}")
        host = np.asarray(self.tokens[first:last], dtype=np.int64).reshape(batch_size, width)
        tensor = torch.from_numpy(host).to(device=device, non_blocking=False)
        return tensor[:, :-1], tensor[:, 1:]


def causal_loss(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=inputs, use_cache=False).logits
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    token_file: TokenFile,
    batches: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    losses = []
    started = time.perf_counter()
    for index in range(batches):
        inputs, targets = token_file.batch(index, batch_size, device)
        losses.append(float(causal_loss(model, inputs, targets)))
    torch.cuda.synchronize()
    return {
        "loss": float(np.mean(losses)),
        "per_batch_loss": losses,
        "prediction_tokens": batches * batch_size * token_file.sequence_length,
        "elapsed_seconds": time.perf_counter() - started,
    }


def lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step + 1 - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def beta_gradient_stats(modules: list[ScreenMLP]) -> dict[str, float | int]:
    gradients = [module.beta.grad.detach().float().reshape(-1) for module in modules if module.beta.grad is not None]
    if not gradients:
        return {"coordinates": 0}
    values = torch.cat(gradients).abs()
    return {
        "coordinates": values.numel(),
        "mean_abs": float(values.mean()),
        "rms": float(torch.sqrt(torch.mean(values.square()))),
        "p99_abs": float(torch.quantile(values, 0.99)),
        "p999_abs": float(torch.quantile(values, 0.999)),
        "max_abs": float(values.max()),
    }


def activation_diagnostics(
    model: nn.Module,
    modules: list[ScreenMLP],
    token_file: TokenFile,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    for module in modules:
        module.record_diagnostics = True
    inputs, _ = token_file.batch(0, 2, device)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(input_ids=inputs, use_cache=False)
    torch.cuda.synchronize()
    records = [module.last_diagnostics for module in modules]
    for module in modules:
        module.record_diagnostics = False
    if any(record is None for record in records):
        raise RuntimeError("missing activation diagnostics")
    typed_records = [record for record in records if record is not None]
    coordinates = sum(int(record["coordinates"]) for record in typed_records)
    alpha = torch.cat([effective_alpha(module.beta.detach().float()).reshape(-1) for module in modules])
    return {
        "clip_negative_fraction": sum(int(record["negative_clipped"]) for record in typed_records) / coordinates,
        "clip_positive_fraction": sum(int(record["positive_clipped"]) for record in typed_records) / coordinates,
        "u_abs_sample_p99_layer_median": float(
            np.median([float(record["u_abs_sample_p99"]) for record in typed_records])
        ),
        "u_abs_sample_p999_layer_max": max(
            float(record["u_abs_sample_p999"]) for record in typed_records
        ),
        "u_abs_true_max": max(float(record["u_abs_true_max"]) for record in typed_records),
        "alpha_mean_abs": float(alpha.abs().mean()),
        "alpha_max_abs": float(alpha.abs().max()),
        "alpha_saturation_fraction": float((alpha.abs() > 1.9).float().mean()),
    }


def make_optimizer(
    model: nn.Module,
    shared_lr: float,
    beta_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    beta_parameters = []
    shared_parameters = []
    for name, parameter in model.named_parameters():
        (beta_parameters if name.endswith(".beta") else shared_parameters).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": shared_parameters, "lr": shared_lr, "base_lr": shared_lr, "weight_decay": weight_decay},
            {"params": beta_parameters, "lr": beta_lr, "base_lr": beta_lr, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True,
    )


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_data_ledger(
    manifest_path: Path,
    train_path: Path,
    validation_path: Path,
    sequence_length: int,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    expected_scalars = {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "dataset": DATASET,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "sequence_length": sequence_length,
        "dtype": "uint16",
        "split_rule": "validation iff uint64_be(sha256(id)[:8]) mod 10 == 0",
    }
    mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in expected_scalars.items()
        if manifest.get(key) != expected
    }
    if train_path.resolve() == validation_path.resolve():
        mismatches["distinct_paths"] = {"expected": True, "actual": False}
    verified_files = {}
    for name, path in (("train", train_path), ("validation", validation_path)):
        record = manifest.get("files", {}).get(name, {})
        actual_bytes = path.stat().st_size
        actual_sha = sha256_file(path)
        expected_sequences = manifest.get(f"{name}_sequences")
        actual_sequences = actual_bytes // (2 * (sequence_length + 1))
        checks = {
            "bytes": actual_bytes == record.get("bytes"),
            "sha256": actual_sha == record.get("sha256"),
            "sequence_count": actual_sequences == expected_sequences,
            "whole_uint16_sequences": actual_bytes % (2 * (sequence_length + 1)) == 0,
        }
        if not all(checks.values()):
            mismatches[f"{name}_file"] = checks
        verified_files[name] = {
            "path": str(path.resolve()),
            "bytes": actual_bytes,
            "sha256": actual_sha,
            "sequences": actual_sequences,
            "checks": checks,
        }
    if mismatches:
        raise ValueError(f"invalid data ledger: {json.dumps(mismatches, sort_keys=True)}")
    return {"valid": True, "manifest": str(manifest_path.resolve()), "files": verified_files}


def validate_serving_ledger(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    key_cells = [cell for cell in payload.get("cells", []) if cell.get("batch") in {1, 8}]
    checks = {
        "all_frozen_gates_pass": payload.get("all_gates_pass") is True,
        "h100_device": "H100" in str(payload.get("device", "")),
        "both_key_cells_present": {cell.get("batch") for cell in key_cells} == {1, 8},
        "same_width_key_cells_at_most_1p02": bool(key_cells)
            and all(cell.get("same_width_over_baseline_median", math.inf) <= 1.02 for cell in key_cells),
        "equal_parameter_key_cells_at_most_1p02": bool(key_cells)
            and all(cell.get("equal_parameter_over_baseline_median", math.inf) <= 1.02 for cell in key_cells),
    }
    if not all(checks.values()):
        raise ValueError(f"invalid serving ledger: {json.dumps(checks, sort_keys=True)}")
    return {
        "valid": True,
        "scope": "target-scale D=4096 M=14336 fused serving executor; not 135M model latency",
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "checks": checks,
        "key_cells": key_cells,
    }


def validate_experiment_protocol(
    args: argparse.Namespace,
    arms: tuple[str, ...],
    train_file: TokenFile,
    validation_file: TokenFile,
) -> dict[str, Any]:
    expected = {
        "arms": VALID_ARMS,
        "device_contains": "H100",
        "torch_version": "2.5.1+cu124",
        "cuda_version": "12.4",
        "transformers_version": "4.57.6",
        "sequence_length": 512,
        "micro_batch_size": 32,
        "gradient_accumulation": 2,
        "eval_batch_size": 32,
        "eval_batches": 128,
        "steps": 640,
        "eval_steps": [40, 160, 640],
        "warmup_steps": 32,
        "shared_lr": 1e-4,
        "beta_lr": 5e-4,
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
        "seed": 101,
        "compile": False,
        "save_checkpoints": True,
        "train_sequences": 40_960,
        "validation_sequences": 4_096,
        "prediction_tokens": 20_971_520,
    }
    actual = {
        "arms": arms,
        "device_contains": torch.cuda.get_device_name(0),
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
        "shared_lr": args.shared_lr,
        "beta_lr": args.beta_lr,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "seed": args.seed,
        "compile": args.compile,
        "save_checkpoints": args.save_checkpoints,
        "train_sequences": train_file.sequence_count,
        "validation_sequences": validation_file.sequence_count,
        "prediction_tokens": (
            args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length
        ),
    }
    checks = {
        key: (expected_value in actual[key] if key == "device_contains" else actual[key] == expected_value)
        for key, expected_value in expected.items()
    }
    valid = all(checks.values())
    if args.strict_protocol and not valid:
        failed = {key: {"expected": expected[key], "actual": actual[key]} for key, ok in checks.items() if not ok}
        raise ValueError(f"invalid experiment protocol: {json.dumps(failed, sort_keys=True)}")
    return {"valid": valid, "strict": args.strict_protocol, "checks": checks, "expected": expected, "actual": actual}


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
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=MODEL_REVISION,
        dtype=torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_disable()
    modules = install_arm(model, arm)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = make_optimizer(model, args.shared_lr, args.beta_lr, args.weight_decay)
    execution_model = torch.compile(model, mode="reduce-overhead", fullgraph=False) if args.compile else model

    checkpoints = sorted({0, *args.eval_steps, args.steps})
    evaluations: dict[str, Any] = {}
    evaluations["0"] = evaluate(
        execution_model, validation_file, args.eval_batches, args.eval_batch_size, device
    )
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    step_losses = []
    step_seconds = []
    gradient_norms = []
    beta_gradients: dict[str, Any] = {}
    nonfinite = False

    for step in range(args.steps):
        execution_model.train()
        multiplier = lr_multiplier(step, args.steps, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = float(group["base_lr"]) * multiplier
        started = time.perf_counter()
        accumulated_loss = 0.0
        for micro in range(args.gradient_accumulation):
            batch_index = step * args.gradient_accumulation + micro
            inputs, targets = train_file.batch(batch_index, args.micro_batch_size, device)
            loss = causal_loss(execution_model, inputs, targets)
            (loss / args.gradient_accumulation).backward()
            accumulated_loss += float(loss.detach()) / args.gradient_accumulation
        if step + 1 == 1 or step + 1 in checkpoints:
            beta_gradients[str(step + 1)] = beta_gradient_stats(modules)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        norm_value = float(norm)
        if not math.isfinite(norm_value) or not math.isfinite(accumulated_loss):
            nonfinite = True
            raise RuntimeError(f"non-finite training state in {arm} at step {step + 1}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_losses.append(accumulated_loss)
        step_seconds.append(time.perf_counter() - started)
        gradient_norms.append(norm_value)
        if step + 1 in checkpoints:
            evaluations[str(step + 1)] = evaluate(
                execution_model, validation_file, args.eval_batches, args.eval_batch_size, device
            )
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "step": step + 1,
                        "train_loss": accumulated_loss,
                        "validation_loss": evaluations[str(step + 1)]["loss"],
                        "gradient_norm": norm_value,
                    }
                ),
                flush=True,
            )

    diagnostics = activation_diagnostics(model, modules, validation_file, device)
    checkpoint_path = args.checkpoint_dir / f"{arm}.pt"
    if args.save_checkpoints:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "arm": arm,
                "model": MODEL,
                "model_revision": MODEL_REVISION,
                "state_dict": model.state_dict(),
                "args": vars(args),
            },
            checkpoint_path,
        )
    measured_seconds = sum(step_seconds)
    prediction_tokens = args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length
    result = {
        "arm": arm,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "added_beta_parameters": sum(module.beta.numel() for module in modules),
        "evaluations": evaluations,
        "train": {
            "steps": args.steps,
            "prediction_tokens": prediction_tokens,
            "mean_loss": float(np.mean(step_losses)),
            "max_loss": max(step_losses),
            "final_loss": step_losses[-1],
            "max_preclip_gradient_norm": max(gradient_norms),
            "preclip_fraction_above_clip_threshold": float(
                np.mean(np.asarray(gradient_norms) > args.gradient_clip)
            ),
            "nonfinite": nonfinite,
            "elapsed_seconds": measured_seconds,
            "tokens_per_second": prediction_tokens / measured_seconds,
            "median_step_seconds_after_five": float(
                np.median(step_seconds[min(5, max(0, len(step_seconds) - 1)) :])
            ),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "beta_gradient_stats": beta_gradients,
        "activation_diagnostics": diagnostics,
        "checkpoint": str(checkpoint_path) if args.save_checkpoints else None,
    }
    del execution_model, optimizer, modules, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def all_reported_numbers_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(all_reported_numbers_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(all_reported_numbers_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def paired_loss_interval(candidate: dict[str, Any], reference: dict[str, Any], step: str) -> dict[str, float]:
    candidate_losses = np.asarray(candidate["evaluations"][step]["per_batch_loss"], dtype=np.float64)
    reference_losses = np.asarray(reference["evaluations"][step]["per_batch_loss"], dtype=np.float64)
    if candidate_losses.shape != reference_losses.shape or candidate_losses.size < 2:
        raise ValueError("paired validation losses require matching batches and at least two observations")
    difference = candidate_losses - reference_losses
    standard_error = float(difference.std(ddof=1) / math.sqrt(difference.size))
    mean = float(difference.mean())
    return {
        "candidate_minus_reference_mean": mean,
        "standard_error": standard_error,
        "lower_95": mean - 1.96 * standard_error,
        "upper_95": mean + 1.96 * standard_error,
    }


def decide(
    arm_results: dict[str, Any],
    endpoint_tolerance: float = 1e-7,
    data_protocol_valid: bool = False,
    serving_gate_pass: bool = False,
    experiment_protocol_valid: bool = False,
) -> dict[str, Any]:
    required = set(VALID_ARMS)
    if set(arm_results) != required:
        return {"complete": False, "missing_arms": sorted(required - set(arm_results))}
    baseline = arm_results["baseline"]
    terminal_step = str(baseline["train"]["steps"])
    baseline_terminal = baseline["evaluations"][terminal_step]["loss"]
    reflex_name = min(
        ("reflex", "reflex_detach"),
        key=lambda name: arm_results[name]["evaluations"][terminal_step]["loss"],
    )
    control_name = min(
        ("gate_bias", "gate_temperature"),
        key=lambda name: arm_results[name]["evaluations"][terminal_step]["loss"],
    )
    reflex_terminal = arm_results[reflex_name]["evaluations"][terminal_step]["loss"]
    control_terminal = arm_results[control_name]["evaluations"][terminal_step]["loss"]
    reflex_vs_baseline_interval = paired_loss_interval(
        arm_results[reflex_name], baseline, terminal_step
    )
    reflex_vs_control_interval = paired_loss_interval(
        arm_results[reflex_name], arm_results[control_name], terminal_step
    )
    counts = {result["total_parameters"] for result in arm_results.values()}
    trainable_counts = {result["trainable_parameters"] for result in arm_results.values()}
    initial = {name: result["evaluations"]["0"]["loss"] for name, result in arm_results.items()}
    endpoint_max_difference = max(abs(loss - initial["baseline"]) for loss in initial.values())
    mid_step = "160"
    mid_reflex = arm_results[reflex_name]["evaluations"].get(mid_step, {}).get("loss")
    mid_baseline = baseline["evaluations"].get(mid_step, {}).get("loss")
    gates = {
        "verified_data_protocol": data_protocol_valid,
        "verified_serving_gate": serving_gate_pass,
        "verified_experiment_protocol": experiment_protocol_valid,
        "identical_parameter_counts": len(counts) == 1,
        "identical_trainable_parameter_counts": len(trainable_counts) == 1,
        "exact_loss_endpoint": endpoint_max_difference <= endpoint_tolerance,
        "all_finite": all(
            not result["train"]["nonfinite"] and all_reported_numbers_finite(result)
            for result in arm_results.values()
        ),
        "bounded_training_stability": all(
            result["train"]["max_preclip_gradient_norm"] <= 100.0
            and result["train"]["max_loss"] <= 20.0
            and result["activation_diagnostics"]["alpha_saturation_fraction"] <= 0.01
            for result in arm_results.values()
        ),
        "reflex_at_least_0p2_percent_better_than_baseline":
            (baseline_terminal - reflex_terminal) / baseline_terminal >= 0.002,
        "reflex_at_least_0p1_percent_better_than_simple_controls":
            (control_terminal - reflex_terminal) / control_terminal >= 0.001,
        "reflex_advantage_present_at_step_160":
            mid_reflex is not None and mid_baseline is not None and mid_reflex < mid_baseline,
        "paired_95_percent_interval_favors_reflex_vs_baseline":
            reflex_vs_baseline_interval["upper_95"] < 0.0,
        "paired_95_percent_interval_favors_reflex_vs_simple_control":
            reflex_vs_control_interval["upper_95"] < 0.0,
    }
    return {
        "complete": True,
        "terminal_step": int(terminal_step),
        "best_reflex_arm": reflex_name,
        "best_simple_control": control_name,
        "baseline_terminal_loss": baseline_terminal,
        "reflex_terminal_loss": reflex_terminal,
        "simple_control_terminal_loss": control_terminal,
        "reflex_relative_improvement_vs_baseline": (baseline_terminal - reflex_terminal) / baseline_terminal,
        "reflex_relative_improvement_vs_simple_control": (control_terminal - reflex_terminal) / control_terminal,
        "endpoint_max_loss_difference": endpoint_max_difference,
        "paired_terminal_intervals": {
            "reflex_vs_baseline": reflex_vs_baseline_interval,
            "reflex_vs_simple_control": reflex_vs_control_interval,
        },
        "gates": gates,
        "advance_to_replication": all(gates.values()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    if len(arms) != len(set(arms)) or any(arm not in VALID_ARMS for arm in arms):
        raise ValueError(f"arms must be unique members of {VALID_ARMS}")
    train_file = TokenFile(args.train_file, args.sequence_length)
    validation_file = TokenFile(args.validation_file, args.sequence_length)
    data_ledger = validate_data_ledger(
        args.data_manifest, args.train_file, args.validation_file, args.sequence_length
    )
    serving_result = Path("results/reflex-swiglu-fused-h100-v2.json")
    serving_ledger = validate_serving_ledger(serving_result)
    required_train_sequences = args.steps * args.gradient_accumulation * args.micro_batch_size
    required_validation_sequences = args.eval_batches * args.eval_batch_size
    if train_file.sequence_count < required_train_sequences:
        raise ValueError("training token file is too short")
    if validation_file.sequence_count < required_validation_sequences:
        raise ValueError("validation token file is too short")
    experiment_protocol = validate_experiment_protocol(args, arms, train_file, validation_file)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    preregistration = Path("results/reflex-swiglu-lm-screen-preregistration.md")
    payload: dict[str, Any] = {
        "candidate": "Reflex-SwiGLU",
        "scope": "matched one-seed continuation-learning screen",
        "source_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(preregistration),
        "data_manifest_sha256": sha256_file(args.data_manifest),
        "serving_result_sha256": sha256_file(serving_result),
        "data_ledger": data_ledger,
        "serving_ledger": serving_ledger,
        "experiment_protocol": experiment_protocol,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "arms": {},
    }
    for arm in arms:
        print(json.dumps({"starting_arm": arm}), flush=True)
        payload["arms"][arm] = train_arm(arm, args, train_file, validation_file, device)
        payload["decision"] = decide(
            payload["arms"],
            data_protocol_valid=data_ledger["valid"],
            serving_gate_pass=serving_ledger["valid"],
            experiment_protocol_valid=experiment_protocol["valid"],
        )
        write_payload(args.output, payload)
    return payload


def parse_steps(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("data/reflex-lm-screen/train.uint16.bin"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/reflex-lm-screen/validation.uint16.bin"))
    parser.add_argument("--data-manifest", type=Path, default=Path("results/reflex-swiglu-lm-data-manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/reflex-swiglu-lm-screen.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/reflex-lm-screen"))
    parser.add_argument("--arms", default=",".join(VALID_ARMS))
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=128)
    parser.add_argument("--steps", type=int, default=640)
    parser.add_argument("--eval-steps", type=parse_steps, default=parse_steps("40,160,640"))
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--shared-lr", type=float, default=1e-4)
    parser.add_argument("--beta-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-protocol", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload.get("decision", {}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
