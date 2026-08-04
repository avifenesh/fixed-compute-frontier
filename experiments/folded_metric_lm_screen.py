#!/usr/bin/env python3
"""Matched scratch-LM screen for folded learned-metric optimizer state."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM

from experiments.block_algebra_swiglu_lm_screen import scratch_config
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


VALID_ARMS = ("baseline", "lr_matched", "fixed_metric", "learned_metric")
RANK = 16
SCALE = 1.0
FACTOR_SEED = 20_260_731
PREREGISTRATION = Path("results/folded-metric-lm-screen-preregistration.md")
INTEGRITY_MANIFEST = Path("results/folded-metric-integrity-manifest.json")
TEST_SOURCE = Path("tests/test_folded_metric_lm_screen.py")
ALGEBRA_SOURCE = Path("experiments/folded_metric_optimizer_gate.py")
ALGEBRA_TEST = Path("tests/test_folded_metric_optimizer_gate.py")
ALGEBRA_RESULT = Path("results/folded-metric-optimizer-stage0.json")
DEFAULT_PROBE = Path("results/folded-metric-first-step-probe.json")
DIAGNOSTIC_STEPS = {1, 100, 305, 1525}


def configure_matmul_precision() -> dict[str, Any]:
    """Apply and report the precision contract shared by probe and screen."""
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    }


def build_plain_model(device: torch.device) -> nn.Module:
    model = AutoModelForCausalLM.from_config(
        scratch_config(), attn_implementation="sdpa"
    ).to(device)
    model.config.use_cache = False
    return model


def targeted_weights(model: nn.Module) -> dict[str, nn.Parameter]:
    result: dict[str, nn.Parameter] = {}
    for layer_index, layer in enumerate(model.model.layers):
        modules = {
            "q": layer.self_attn.q_proj,
            "k": layer.self_attn.k_proj,
            "v": layer.self_attn.v_proj,
            "o": layer.self_attn.o_proj,
            "gate": layer.mlp.gate_proj,
            "up": layer.mlp.up_proj,
            "down": layer.mlp.down_proj,
        }
        for short_name, module in modules.items():
            if module.bias is not None:
                raise ValueError("frozen screen expects bias-free targeted linears")
            result[f"layer_{layer_index:02d}.{short_name}"] = module.weight
    return result


def causal_loss(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=inputs, use_cache=False).logits
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1)
    )


def deterministic_orthonormal(
    rows: int, rank: int, seed: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn(rows, rank, generator=generator, dtype=torch.float32)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q.to(device)


def adam_tensor_step(
    value: torch.Tensor,
    gradient: torch.Tensor,
    first_moment: torch.Tensor,
    second_moment: torch.Tensor,
    step: int,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.95,
    epsilon: float = 1e-8,
) -> None:
    first_moment.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
    second_moment.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
    corrected_first = first_moment / (1.0 - beta1**step)
    corrected_second = second_moment / (1.0 - beta2**step)
    value.addcdiv_(
        corrected_first,
        corrected_second.sqrt().add_(epsilon),
        value=-learning_rate,
    )


class FoldedMetricController:
    """Training-only factor state; model weights remain fused ordinary tensors."""

    def __init__(
        self,
        targets: dict[str, nn.Parameter],
        learn_a: bool,
        rank: int = RANK,
        scale: float = SCALE,
        factor_seed: int = FACTOR_SEED,
    ) -> None:
        self.targets = targets
        self.learn_a = learn_a
        self.rank = rank
        self.scale = scale
        self.step_number = 0
        self.states: dict[str, dict[str, torch.Tensor]] = {}
        for index, (name, parameter) in enumerate(targets.items()):
            output, input_ = parameter.shape
            if rank > min(output, input_):
                raise ValueError(f"rank {rank} exceeds {name} shape {parameter.shape}")
            a = deterministic_orthonormal(
                output, rank, factor_seed + index, parameter.device
            )
            b = torch.zeros(rank, input_, device=parameter.device, dtype=torch.float32)
            self.states[name] = {
                "a": a,
                "a_initial": a.clone(),
                "b": b,
                "m_a": torch.zeros_like(a),
                "v_a": torch.zeros_like(a),
                "m_b": torch.zeros_like(b),
                "v_b": torch.zeros_like(b),
            }
        self.prepared: dict[str, dict[str, torch.Tensor]] | None = None

    @property
    def factor_scalars(self) -> int:
        return sum(
            state["a"].numel() + state["b"].numel()
            for state in self.states.values()
        )

    def prepare(self, record_diagnostics: bool) -> None:
        if self.prepared is not None:
            raise RuntimeError("controller step already prepared")
        prepared: dict[str, dict[str, torch.Tensor]] = {}
        for name, parameter in self.targets.items():
            if parameter.grad is None:
                raise RuntimeError(f"missing gradient for {name}")
            state = self.states[name]
            gradient = parameter.grad.detach().float()
            a, b = state["a"], state["b"]
            record = {
                "old_product": a @ b,
                "grad_b": self.scale * (a.T @ gradient),
            }
            if self.learn_a:
                record["grad_a"] = self.scale * (gradient @ b.T)
            if record_diagnostics:
                record["weight_before"] = parameter.detach().clone()
            prepared[name] = record
        self.prepared = prepared

    @torch.no_grad()
    def finish(self, learning_rate: float, record_diagnostics: bool) -> dict[str, Any] | None:
        if self.prepared is None:
            raise RuntimeError("controller step was not prepared")
        self.step_number += 1
        records = []
        for name, parameter in self.targets.items():
            state = self.states[name]
            prepared = self.prepared[name]
            a, b = state["a"], state["b"]
            if self.learn_a:
                adam_tensor_step(
                    a,
                    prepared["grad_a"],
                    state["m_a"],
                    state["v_a"],
                    self.step_number,
                    learning_rate,
                )
            adam_tensor_step(
                b,
                prepared["grad_b"],
                state["m_b"],
                state["v_b"],
                self.step_number,
                learning_rate,
            )
            factor_delta = self.scale * (a @ b - prepared["old_product"])
            if record_diagnostics:
                base_delta = parameter.detach() - prepared["weight_before"]
                base_flat = base_delta.float().reshape(-1)
                factor_flat = factor_delta.float().reshape(-1)
                base_norm = float(torch.linalg.vector_norm(base_flat))
                factor_norm = float(torch.linalg.vector_norm(factor_flat))
                cosine = float(
                    torch.dot(base_flat, factor_flat)
                    / max(base_norm * factor_norm, 1e-30)
                )
                effective_norm = float(
                    torch.linalg.vector_norm(base_flat + factor_flat)
                )
                records.append(
                    {
                        "name": name,
                        "base_update_norm": base_norm,
                        "factor_update_norm": factor_norm,
                        "factor_to_base_norm": factor_norm / max(base_norm, 1e-30),
                        "effective_to_base_norm": effective_norm / max(base_norm, 1e-30),
                        "factor_base_cosine": cosine,
                        "a_norm": float(torch.linalg.vector_norm(a)),
                        "a_change_norm": float(
                            torch.linalg.vector_norm(a - state["a_initial"])
                        ),
                        "b_norm": float(torch.linalg.vector_norm(b)),
                    }
                )
            parameter.add_(factor_delta.to(parameter.dtype))
        self.prepared = None
        if not record_diagnostics:
            return None
        return {
            "per_matrix": records,
            "median_factor_to_base_norm": float(
                np.median([record["factor_to_base_norm"] for record in records])
            ),
            "median_effective_to_base_norm": float(
                np.median([record["effective_to_base_norm"] for record in records])
            ),
            "median_factor_base_cosine": float(
                np.median([record["factor_base_cosine"] for record in records])
            ),
            "median_a_change_norm": float(
                np.median([record["a_change_norm"] for record in records])
            ),
            "median_b_norm": float(
                np.median([record["b_norm"] for record in records])
            ),
        }


def make_optimizer(
    model: nn.Module,
    targets: dict[str, nn.Parameter],
    learning_rate: float,
    weight_decay: float,
    lr_multipliers: dict[str, float] | None = None,
) -> torch.optim.Optimizer:
    targeted_ids = {id(parameter) for parameter in targets.values()}
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in targeted_ids
    ]
    groups: list[dict[str, Any]] = [
        {
            "params": other_parameters,
            "lr": learning_rate,
            "base_lr": learning_rate,
            "weight_decay": weight_decay,
            "target_name": None,
        }
    ]
    for name, parameter in targets.items():
        multiplier = 1.0 if lr_multipliers is None else lr_multipliers[name]
        groups.append(
            {
                "params": [parameter],
                "lr": learning_rate * multiplier,
                "base_lr": learning_rate * multiplier,
                "weight_decay": weight_decay,
                "target_name": name,
            }
        )
    return torch.optim.AdamW(
        groups, betas=(0.9, 0.95), eps=1e-8, fused=True
    )


def set_schedule(optimizer: torch.optim.Optimizer, multiplier: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(group["base_lr"]) * multiplier


def factor_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    # Factor LR follows the unmultiplied global model group.
    return float(optimizer.param_groups[0]["lr"])


def probe_first_step(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    precision = configure_matmul_precision()
    data_ledger = validate_data_ledger(
        args.data_manifest,
        args.train_file,
        args.validation_file,
        args.sequence_length,
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_file = TokenFile(args.train_file, args.sequence_length)
    model = build_plain_model(device)
    targets = targeted_weights(model)
    controller = FoldedMetricController(targets, learn_a=True)
    optimizer = make_optimizer(
        model, targets, args.learning_rate, args.weight_decay
    )
    set_schedule(optimizer, lr_multiplier(0, args.steps, args.warmup_steps))
    optimizer.zero_grad(set_to_none=True)
    for micro in range(args.gradient_accumulation):
        inputs, labels = train_file.batch(
            micro, args.micro_batch_size, device
        )
        loss = causal_loss(model, inputs, labels)
        (loss / args.gradient_accumulation).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
    controller.prepare(record_diagnostics=True)
    optimizer.step()
    diagnostics = controller.finish(
        factor_learning_rate(optimizer), record_diagnostics=True
    )
    assert diagnostics is not None
    multipliers = {
        record["name"]: record["effective_to_base_norm"]
        for record in diagnostics["per_matrix"]
    }
    payload = {
        "scope": "first training step only; training data, no validation",
        "source_sha256": sha256_file(Path(__file__)),
        "data_manifest_sha256": sha256_file(args.data_manifest),
        "train_file_sha256": data_ledger["files"]["train"]["sha256"],
        "validation_file_sha256": data_ledger["files"]["validation"]["sha256"],
        "seed": args.seed,
        "factor_seed": FACTOR_SEED,
        "rank": RANK,
        "scale": SCALE,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "sequence_length": args.sequence_length,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate_at_probe": factor_learning_rate(optimizer),
        "precision": precision,
        "lr_multipliers": multipliers,
        "diagnostics": diagnostics,
    }
    del optimizer, controller, targets, model
    gc.collect()
    torch.cuda.empty_cache()
    return payload


@torch.no_grad()
def reload_check(model: nn.Module, validation_file: TokenFile, device: torch.device) -> dict[str, Any]:
    model.eval()
    clone = build_plain_model(device)
    with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
        torch.save(model.state_dict(), checkpoint.name)
        checkpoint.flush()
        serialized_bytes = Path(checkpoint.name).stat().st_size
        plain_state = torch.load(
            checkpoint.name, map_location=device, weights_only=True
        )
    clone.load_state_dict(plain_state, strict=True)
    clone.eval()
    inputs, labels = validation_file.batch(0, 2, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        original_logits = model(input_ids=inputs, use_cache=False).logits
        clone_logits = clone(input_ids=inputs, use_cache=False).logits
    original_loss = F.cross_entropy(
        original_logits.float().reshape(-1, original_logits.shape[-1]),
        labels.reshape(-1),
    )
    clone_loss = F.cross_entropy(
        clone_logits.float().reshape(-1, clone_logits.shape[-1]), labels.reshape(-1)
    )
    result = {
        "logits_bitwise_equal": bool(torch.equal(original_logits, clone_logits)),
        "max_logit_absolute_difference": float(
            (original_logits.float() - clone_logits.float()).abs().max()
        ),
        "loss_absolute_difference": abs(float(original_loss) - float(clone_loss)),
        "state_dict_contains_metric_factors": any(
            "metric" in key or "factor" in key for key in plain_state
        ),
        "disk_serialization_checked": True,
        "serialized_checkpoint_bytes": serialized_bytes,
        "reloaded_parameter_count": sum(p.numel() for p in clone.parameters()),
    }
    del clone, original_logits, clone_logits
    torch.cuda.empty_cache()
    return result


def validate_protocol(args: argparse.Namespace, arms: tuple[str, ...]) -> dict[str, Any]:
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
        "steps": 1525,
        "eval_steps": [305, 1525],
        "warmup_steps": 100,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
        "seed": 223,
        "rank": 16,
        "scale": 1.0,
        "factor_seed": FACTOR_SEED,
        "prediction_tokens": 49_971_200,
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
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "seed": args.seed,
        "rank": RANK,
        "scale": SCALE,
        "factor_seed": FACTOR_SEED,
        "prediction_tokens": args.steps
        * args.gradient_accumulation
        * args.micro_batch_size
        * args.sequence_length,
    }
    checks = {
        key: (
            expected_value in actual[key]
            if key == "device_contains"
            else actual[key] == expected_value
        )
        for key, expected_value in expected.items()
    }
    if args.strict_protocol and not all(checks.values()):
        failed = {
            key: {"expected": expected[key], "actual": actual[key]}
            for key, valid in checks.items()
            if not valid
        }
        raise ValueError(f"invalid protocol: {json.dumps(failed, sort_keys=True)}")
    return {"valid": all(checks.values()), "checks": checks, "expected": expected, "actual": actual}


def train_arm(
    arm: str,
    args: argparse.Namespace,
    train_file: TokenFile,
    validation_file: TokenFile,
    device: torch.device,
    lr_multipliers: dict[str, float],
) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = build_plain_model(device)
    targets = targeted_weights(model)
    controller = None
    if arm in {"fixed_metric", "learned_metric"}:
        controller = FoldedMetricController(
            targets, learn_a=arm == "learned_metric"
        )
    optimizer = make_optimizer(
        model,
        targets,
        args.learning_rate,
        args.weight_decay,
        lr_multipliers if arm == "lr_matched" else None,
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    evaluations = {
        "0": evaluate(model, validation_file, args.eval_batches, args.eval_batch_size, device)
    }
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    step_losses, step_seconds, gradient_norms = [], [], []
    metric_diagnostics: dict[str, Any] = {}
    for step in range(args.steps):
        model.train()
        schedule = lr_multiplier(step, args.steps, args.warmup_steps)
        set_schedule(optimizer, schedule)
        started = time.perf_counter()
        accumulated_loss = 0.0
        for micro in range(args.gradient_accumulation):
            batch_index = step * args.gradient_accumulation + micro
            inputs, labels = train_file.batch(
                batch_index, args.micro_batch_size, device
            )
            loss = causal_loss(model, inputs, labels)
            (loss / args.gradient_accumulation).backward()
            accumulated_loss += float(loss.detach()) / args.gradient_accumulation
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip))
        if not math.isfinite(norm) or not math.isfinite(accumulated_loss):
            raise RuntimeError(f"non-finite {arm} step {step + 1}")
        record = step + 1 in DIAGNOSTIC_STEPS
        if controller is not None:
            controller.prepare(record_diagnostics=record)
        optimizer.step()
        if controller is not None:
            diagnostics = controller.finish(
                factor_learning_rate(optimizer), record_diagnostics=record
            )
            if diagnostics is not None:
                metric_diagnostics[str(step + 1)] = diagnostics
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_losses.append(accumulated_loss)
        step_seconds.append(time.perf_counter() - started)
        gradient_norms.append(norm)
        if step + 1 in args.eval_steps:
            evaluations[str(step + 1)] = evaluate(
                model, validation_file, args.eval_batches, args.eval_batch_size, device
            )
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "step": step + 1,
                        "train_loss": accumulated_loss,
                        "validation_loss": evaluations[str(step + 1)]["loss"],
                        "gradient_norm": norm,
                    }
                ),
                flush=True,
            )
    served = reload_check(model, validation_file, device)
    prediction_tokens = args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length
    result = {
        "arm": arm,
        "model_parameters": total_parameters,
        "optimizer_factor_scalars": 0 if controller is None else controller.factor_scalars,
        "evaluations": evaluations,
        "metric_update_diagnostics": metric_diagnostics,
        "served_reload_check": served,
        "train": {
            "steps": args.steps,
            "prediction_tokens": prediction_tokens,
            "mean_loss": float(np.mean(step_losses)),
            "max_loss": max(step_losses),
            "final_loss": step_losses[-1],
            "max_preclip_gradient_norm": max(gradient_norms),
            "elapsed_seconds": sum(step_seconds),
            "tokens_per_second": prediction_tokens / sum(step_seconds),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "nonfinite": False,
        },
    }
    del optimizer, controller, targets, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def decide(results: dict[str, Any], integrity_valid: bool) -> dict[str, Any]:
    if set(results) != set(VALID_ARMS):
        return {"complete": False, "missing_arms": sorted(set(VALID_ARMS) - set(results))}
    terminal, early = "1525", "305"
    learned = results["learned_metric"]
    references = {
        name: results[name] for name in ("baseline", "lr_matched", "fixed_metric")
    }
    losses = {name: result["evaluations"][terminal]["loss"] for name, result in results.items()}
    intervals = {
        name: paired_loss_interval(learned, reference, terminal)
        for name, reference in references.items()
    }
    gains = {
        name: (losses[name] - losses["learned_metric"]) / losses[name]
        for name in references
    }
    early_baseline_loss = results["baseline"]["evaluations"][early]["loss"]
    early_learned_loss = learned["evaluations"][early]["loss"]
    early_gain = (early_baseline_loss - early_learned_loss) / early_baseline_loss
    initial_losses = [result["evaluations"]["0"]["loss"] for result in results.values()]
    gates = {
        "integrity_protocol_valid": integrity_valid,
        "identical_model_parameter_counts": len({r["model_parameters"] for r in results.values()}) == 1,
        "exact_initial_loss_endpoint": max(initial_losses) - min(initial_losses) <= 1e-7,
        "all_training_finite": all(not r["train"]["nonfinite"] for r in results.values()),
        "bounded_training": all(r["train"]["max_loss"] <= 20.0 and r["train"]["max_preclip_gradient_norm"] <= 100.0 for r in results.values()),
        "learned_beats_baseline_by_0p1_percent": gains["baseline"] >= 0.001,
        "learned_beats_lr_control_by_0p1_percent": gains["lr_matched"] >= 0.001,
        "learned_beats_fixed_metric_by_0p05_percent": gains["fixed_metric"] >= 0.0005,
        "all_paired_intervals_favor_learned": all(interval["upper_95"] < 0.0 for interval in intervals.values()),
        "baseline_edge_not_shrinking_after_10m": gains["baseline"] >= early_gain - 0.00025,
        "learned_metric_actually_adapts": learned["metric_update_diagnostics"][terminal]["median_a_change_norm"] > 1e-4,
        "plain_checkpoint_reload_is_exact": all(
            r["served_reload_check"]["logits_bitwise_equal"]
            and r["served_reload_check"]["loss_absolute_difference"] == 0.0
            and not r["served_reload_check"]["state_dict_contains_metric_factors"]
            and r["served_reload_check"]["disk_serialization_checked"]
            and r["served_reload_check"]["reloaded_parameter_count"] == r["model_parameters"]
            for r in results.values()
        ),
    }
    return {
        "complete": True,
        "terminal_losses": losses,
        "learned_relative_gains": gains,
        "learned_early_gain_vs_baseline": early_gain,
        "paired_terminal_intervals": intervals,
        "gates": gates,
        "advance_to_second_seed": all(gates.values()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    precision = configure_matmul_precision()
    if args.probe_only:
        payload = probe_first_step(args, device)
        write_payload(args.probe_output, payload)
        return payload
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    if arms != VALID_ARMS:
        raise ValueError(f"strict screen requires {VALID_ARMS}")
    train_file = TokenFile(args.train_file, args.sequence_length)
    validation_file = TokenFile(args.validation_file, args.sequence_length)
    data_ledger = validate_data_ledger(args.data_manifest, args.train_file, args.validation_file, args.sequence_length)
    protocol = validate_protocol(args, arms)
    probe = json.loads(args.probe_file.read_text())
    lr_multipliers = {name: float(value) for name, value in probe["lr_multipliers"].items()}
    identity_model = build_plain_model(torch.device("cpu"))
    expected_names = set(targeted_weights(identity_model))
    del identity_model
    if set(lr_multipliers) != expected_names:
        raise ValueError("probe target names do not match model")
    expected_probe_identity = {
        "source_sha256": sha256_file(Path(__file__)),
        "data_manifest_sha256": sha256_file(args.data_manifest),
        "train_file_sha256": data_ledger["files"]["train"]["sha256"],
        "validation_file_sha256": data_ledger["files"]["validation"]["sha256"],
        "seed": args.seed,
        "factor_seed": FACTOR_SEED,
        "rank": RANK,
        "scale": SCALE,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "sequence_length": args.sequence_length,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate_at_probe": args.learning_rate
        * lr_multiplier(0, args.steps, args.warmup_steps),
        "precision": precision,
    }
    invalid_probe_identity = {
        key: {"expected": value, "actual": probe.get(key)}
        for key, value in expected_probe_identity.items()
        if probe.get(key) != value
    }
    if invalid_probe_identity:
        raise ValueError(
            f"probe identity mismatch: {json.dumps(invalid_probe_identity, sort_keys=True)}"
        )
    integrity = json.loads(INTEGRITY_MANIFEST.read_text())
    integrity_checks = {
        "source": integrity.get("source_sha256") == sha256_file(Path(__file__)),
        "test": integrity.get("test_sha256") == sha256_file(TEST_SOURCE),
        "preregistration": integrity.get("preregistration_sha256") == sha256_file(PREREGISTRATION),
        "algebra_source": integrity.get("algebra_source_sha256") == sha256_file(ALGEBRA_SOURCE),
        "algebra_test": integrity.get("algebra_test_sha256") == sha256_file(ALGEBRA_TEST),
        "algebra_result": integrity.get("algebra_result_sha256") == sha256_file(ALGEBRA_RESULT),
        "probe": integrity.get("probe_sha256") == sha256_file(args.probe_file),
        "data_manifest": integrity.get("data_manifest_sha256")
        == sha256_file(args.data_manifest),
    }
    if not all(integrity_checks.values()):
        raise ValueError(f"invalid integrity manifest: {integrity_checks}")
    payload: dict[str, Any] = {
        "candidate": "folded learned-metric optimizer",
        "scope": "one-seed 50M-token scratch kill-screen",
        "source_sha256": sha256_file(Path(__file__)),
        "test_sha256": sha256_file(TEST_SOURCE),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "integrity_manifest_sha256": sha256_file(INTEGRITY_MANIFEST),
        "integrity_checks": integrity_checks,
        "data_manifest_sha256": sha256_file(args.data_manifest),
        "data_ledger": data_ledger,
        "probe_sha256": sha256_file(args.probe_file),
        "probe": probe,
        "protocol": protocol,
        "base_config": MODEL,
        "base_config_revision": MODEL_REVISION,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "precision": precision,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "arms": {},
    }
    for arm in arms:
        print(json.dumps({"starting_arm": arm}), flush=True)
        payload["arms"][arm] = train_arm(arm, args, train_file, validation_file, device, lr_multipliers)
        payload["decision"] = decide(payload["arms"], data_ledger["valid"] and protocol["valid"] and all(integrity_checks.values()))
        write_payload(args.output, payload)
    return payload


def parse_steps(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("data/block-algebra-scratch/train.uint16.bin"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/block-algebra-scratch/validation.uint16.bin"))
    parser.add_argument("--data-manifest", type=Path, default=Path("results/block-algebra-scratch-data-manifest.json"))
    parser.add_argument("--probe-file", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--probe-output", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/folded-metric-lm-screen.json"))
    parser.add_argument("--arms", default=",".join(VALID_ARMS))
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1525)
    parser.add_argument("--eval-steps", type=parse_steps, default=parse_steps("305,1525"))
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=223)
    parser.add_argument("--strict-protocol", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    payload = run(args)
    if not args.probe_only:
        print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
