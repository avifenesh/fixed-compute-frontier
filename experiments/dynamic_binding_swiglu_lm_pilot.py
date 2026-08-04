#!/usr/bin/env python3
"""10M-token development screen for margin-damped dynamic binding SwiGLU."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

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
from experiments.triangular_microdepth_lm_screen import (
    BASELINE_INTERMEDIATE_SIZE,
    causal_loss,
    scratch_config,
)


ARMS = (
    "raw_baseline",
    "identity_binding",
    "fixed_shift",
    "amplitude_only",
    "fixed_margin_normalized",
    "margin_dynamic_unnormalized",
    "margin_dynamic_normalized",
)
GROUP_SIZE = 8
GROUPS = BASELINE_INTERMEDIATE_SIZE // GROUP_SIZE
OUTPUT = Path("results/dynamic-binding-swiglu-lm-pilot-development.json")
ALGEBRA_SOURCE = Path("experiments/dynamic_binding_swiglu.py")


class BindingMLP(nn.Module):
    def __init__(self, source: nn.Module, arm: str) -> None:
        super().__init__()
        if arm not in ARMS[1:]:
            raise ValueError(f"invalid binding arm: {arm}")
        self.arm = arm
        self.gate_proj = source.gate_proj
        self.up_proj = source.up_proj
        self.down_proj = source.down_proj
        if arm == "fixed_shift":
            with torch.no_grad():
                weight = self.up_proj.weight.reshape(
                    GROUPS, GROUP_SIZE, self.up_proj.weight.shape[-1]
                )
                weight.copy_(torch.roll(weight.clone(), shifts=1, dims=1))
        self.ablation_mode = "full"
        self.record_diagnostics = False
        self.last_diagnostics: dict[str, float] | None = None

    @staticmethod
    def bind(grouped_up: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
        base = torch.arange(GROUP_SIZE, device=grouped_up.device)
        indices = (base + shifts.unsqueeze(-1)) % GROUP_SIZE
        return torch.take_along_dim(grouped_up, indices, dim=-1)

    def activation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        grouped_gate = gate.unflatten(-1, (GROUPS, GROUP_SIZE))
        grouped_up = up.unflatten(-1, (GROUPS, GROUP_SIZE))
        if self.arm == "identity_binding":
            shifts = torch.zeros_like(grouped_gate[..., 0], dtype=torch.long)
            strengths = torch.zeros_like(grouped_gate[..., 0])
        elif self.arm == "fixed_shift":
            shifts = torch.ones_like(grouped_gate[..., 0], dtype=torch.long)
            strengths = torch.ones_like(grouped_gate[..., 0])
        else:
            top = torch.topk(grouped_gate, k=2, dim=-1)
            shifts = top.indices[..., 0]
            margins = top.values[..., 0] - top.values[..., 1]
            strengths = torch.tanh(margins).square()
            if self.arm == "fixed_margin_normalized":
                shifts = torch.ones_like(shifts)
        if self.ablation_mode == "identity":
            strengths = torch.zeros_like(strengths)
        elif self.ablation_mode == "hard":
            strengths = torch.ones_like(strengths)
        elif self.ablation_mode == "shifted_route":
            shifts = (shifts + 1) % GROUP_SIZE
        elif self.ablation_mode == "token_misroute":
            if shifts.ndim < 3:
                raise RuntimeError("token misroute requires batched sequences")
            shifts = torch.roll(shifts, shifts=1, dims=-2)
            strengths = torch.roll(strengths, shifts=1, dims=-2)
        bound = self.bind(grouped_up, shifts)
        interpolated = grouped_up + strengths.unsqueeze(-1) * (bound - grouped_up)
        independent_scale = torch.sqrt(
            (1.0 - strengths).square() + strengths.square()
        )
        variance_scale = torch.where(
            shifts == 0, torch.ones_like(independent_scale), independent_scale
        )
        if self.arm == "identity_binding" or self.ablation_mode == "identity":
            mixed_up = grouped_up
        elif self.arm == "fixed_shift":
            # Avoid a BF16 subtract/add round trip at the exact endpoint.
            mixed_up = bound
        elif self.arm == "amplitude_only":
            mixed_up = variance_scale.unsqueeze(-1) * grouped_up
        elif (
            self.arm in {"fixed_margin_normalized", "margin_dynamic_normalized"}
            and self.ablation_mode != "unnormalized"
        ):
            mixed_up = interpolated / variance_scale.clamp_min(0.5).unsqueeze(-1)
        else:
            mixed_up = interpolated
        activation = F.silu(gate) * mixed_up.flatten(-2)
        if self.record_diagnostics:
            with torch.no_grad():
                histogram = torch.bincount(shifts.reshape(-1), minlength=GROUP_SIZE).float()
                probabilities = histogram / histogram.sum()
                entropy = -torch.sum(
                    torch.where(probabilities > 0, probabilities * probabilities.log(), probabilities)
                ) / math.log(GROUP_SIZE)
                ordinary = F.silu(gate) * up
                correction = activation - ordinary
                ordinary_rms = ordinary.float().square().mean().sqrt()
                self.last_diagnostics = {
                    "activation_rms": float(activation.float().square().mean().sqrt()),
                    "activation_abs_max": float(activation.float().abs().max()),
                    "routing_strength_mean": float(strengths.float().mean()),
                    "routing_strength_rms": float(strengths.float().square().mean().sqrt()),
                    "identity_route_fraction": float(probabilities[0]),
                    "route_entropy_fraction": float(entropy),
                    "correction_relative_rms": float(
                        correction.float().square().mean().sqrt() / ordinary_rms.clamp_min(1e-8)
                    ),
                }
        return activation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(hidden_states))


def build_model(device: torch.device, arm: str):
    if arm not in ARMS:
        raise ValueError(f"invalid arm: {arm}")
    model = AutoModelForCausalLM.from_config(
        scratch_config(), attn_implementation="sdpa"
    ).to(device)
    model.config.use_cache = False
    modules: list[BindingMLP] = []
    if arm != "raw_baseline":
        for layer in model.model.layers:
            replacement = BindingMLP(layer.mlp, arm).to(device)
            layer.mlp = replacement
            modules.append(replacement)
    return model, modules


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    return sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


@torch.no_grad()
def activation_diagnostics(
    model: nn.Module,
    modules: list[BindingMLP],
    validation_file: TokenFile,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    inputs, _ = validation_file.batch(0, 2, device)
    if not modules:
        captured: list[torch.Tensor] = []
        handles = [
            layer.mlp.down_proj.register_forward_pre_hook(
                lambda _module, args: captured.append(args[0].detach())
            )
            for layer in model.model.layers
        ]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(input_ids=inputs, use_cache=False)
        for handle in handles:
            handle.remove()
        return {
            "activation_rms_layer_median": float(np.median([
                float(value.float().square().mean().sqrt()) for value in captured
            ])),
            "activation_abs_max_layer_median": float(np.median([
                float(value.float().abs().max()) for value in captured
            ])),
            "routing_strength_mean_layer_median": 0.0,
            "routing_strength_rms_layer_median": 0.0,
            "identity_route_fraction_layer_median": 1.0,
            "route_entropy_fraction_layer_median": 0.0,
            "correction_relative_rms_layer_median": 0.0,
        }
    for module in modules:
        module.record_diagnostics = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=inputs, use_cache=False)
    records = [module.last_diagnostics for module in modules]
    for module in modules:
        module.record_diagnostics = False
    if any(record is None for record in records):
        raise RuntimeError("missing dynamic-binding diagnostics")
    typed = [record for record in records if record is not None]
    return {
        key + "_layer_median": float(np.median([record[key] for record in typed]))
        for key in typed[0]
    }


def evaluate_ablation(
    model: nn.Module,
    modules: list[BindingMLP],
    validation_file: TokenFile,
    args: argparse.Namespace,
    device: torch.device,
    mode: str,
) -> dict[str, Any]:
    for module in modules:
        module.ablation_mode = mode
    result = evaluate(model, validation_file, args.eval_batches, args.eval_batch_size, device)
    for module in modules:
        module.ablation_mode = "full"
    return result


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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        betas=(0.9, 0.95), eps=1e-8, fused=True,
    )
    for group in optimizer.param_groups:
        group["base_lr"] = args.learning_rate
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    parameter_tensors = len(list(model.parameters()))
    diagnostics = {"initial": activation_diagnostics(model, modules, validation_file, device)}
    evaluations = {"0": evaluate(model, validation_file, args.eval_batches, args.eval_batch_size, device)}
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    losses: list[float] = []
    norms: list[float] = []
    durations: list[float] = []
    for step in range(args.steps):
        model.train()
        multiplier = lr_multiplier(step, args.steps, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = float(group["base_lr"]) * multiplier
        started = time.perf_counter()
        accumulated = 0.0
        for micro in range(args.gradient_accumulation):
            index = step * args.gradient_accumulation + micro
            inputs, targets = train_file.batch(index, args.micro_batch_size, device)
            loss = causal_loss(model, inputs, targets)
            (loss / args.gradient_accumulation).backward()
            accumulated += float(loss.detach()) / args.gradient_accumulation
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip))
        if not math.isfinite(accumulated) or not math.isfinite(norm):
            raise RuntimeError(f"non-finite {arm} step {step + 1}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        losses.append(accumulated)
        norms.append(norm)
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
                "gradient_norm": norm,
            }), flush=True)
    diagnostics["terminal"] = activation_diagnostics(model, modules, validation_file, device)
    ablations: dict[str, Any] = {}
    if arm == "margin_dynamic_normalized":
        for mode in ("identity", "unnormalized", "hard", "shifted_route", "token_misroute"):
            ablations[mode] = evaluate_ablation(
                model, modules, validation_file, args, device, mode
            )
    prediction_tokens = args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length
    result = {
        "arm": arm,
        "total_parameters": total_parameters,
        "parameter_tensors": parameter_tensors,
        "optimizer_state_bytes": optimizer_state_bytes(optimizer),
        "evaluations": evaluations,
        "activation_diagnostics": diagnostics,
        "terminal_ablations": ablations,
        "train": {
            "prediction_tokens": prediction_tokens,
            "mean_loss": float(np.mean(losses)),
            "final_loss": losses[-1],
            "max_loss": max(losses),
            "max_preclip_gradient_norm": max(norms),
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
    losses = {arm: result["evaluations"][terminal]["loss"] for arm, result in results.items()}
    candidate = results["margin_dynamic_normalized"]
    controls = ARMS[:-1]
    best_control = min(controls, key=lambda arm: losses[arm])
    intervals = {
        "candidate_vs_" + arm: paired_loss_interval(candidate, results[arm], terminal)
        for arm in controls
    }
    full = {"evaluations": {terminal: candidate["evaluations"][terminal]}}
    ablation_losses = {}
    for mode, evaluation in candidate["terminal_ablations"].items():
        ablation_losses[mode] = evaluation["loss"]
        intervals["full_vs_" + mode] = paired_loss_interval(
            full, {"evaluations": {terminal: evaluation}}, terminal
        )
    initial_losses = {arm: result["evaluations"]["0"]["loss"] for arm, result in results.items()}
    diagnostic = candidate["activation_diagnostics"]["terminal"]
    numeric = [
        value for result in results.values() for phase in ("initial", "terminal")
        for value in result["activation_diagnostics"][phase].values()
    ]
    gates = {
        "equal_parameter_counts": len({r["total_parameters"] for r in results.values()}) == 1,
        "equal_parameter_tensor_counts": len({r["parameter_tensors"] for r in results.values()}) == 1,
        "equal_optimizer_state_bytes": len({r["optimizer_state_bytes"] for r in results.values()}) == 1,
        "identity_initial_exact_to_raw": abs(initial_losses["identity_binding"] - initial_losses["raw_baseline"]) <= 1e-7,
        "fixed_shift_initial_exact_to_raw": abs(initial_losses["fixed_shift"] - initial_losses["raw_baseline"]) <= 1e-7,
        "identity_terminal_noninferior_0p05_percent": (losses["identity_binding"] - losses["raw_baseline"]) / losses["raw_baseline"] <= 0.0005,
        "all_training_and_diagnostics_finite": all(not r["train"]["nonfinite"] for r in results.values()) and all(math.isfinite(v) for v in numeric),
        "candidate_improves_best_control_0p05_percent": (losses[best_control] - losses["margin_dynamic_normalized"]) / losses[best_control] >= 0.0005,
        "paired_interval_favors_candidate_vs_best": intervals["candidate_vs_" + best_control]["upper_95"] < 0.0,
        "candidate_beats_amplitude_only": losses["margin_dynamic_normalized"] < losses["amplitude_only"],
        "candidate_beats_fixed_margin_relation": losses["margin_dynamic_normalized"] < losses["fixed_margin_normalized"],
        "candidate_beats_unnormalized_binding": losses["margin_dynamic_normalized"] < losses["margin_dynamic_unnormalized"],
        "identity_ablation_hurts_0p02_percent": (ablation_losses["identity"] - losses["margin_dynamic_normalized"]) / losses["margin_dynamic_normalized"] >= 0.0002,
        "paired_interval_favors_full_vs_identity": intervals["full_vs_identity"]["upper_95"] < 0.0,
        "token_misroute_hurts_0p02_percent": (ablation_losses["token_misroute"] - losses["margin_dynamic_normalized"]) / losses["margin_dynamic_normalized"] >= 0.0002,
        "paired_interval_favors_full_vs_token_misroute": intervals["full_vs_token_misroute"]["upper_95"] < 0.0,
        "routing_uses_multiple_shifts": diagnostic["route_entropy_fraction_layer_median"] >= 0.75,
        "routing_correction_is_live": diagnostic["correction_relative_rms_layer_median"] >= 0.05,
    }
    return {
        "complete": True,
        "terminal_losses": losses,
        "initial_losses": initial_losses,
        "best_control": best_control,
        "terminal_ablation_losses": ablation_losses,
        "paired_intervals": intervals,
        "terminal_candidate_diagnostics": diagnostic,
        "relative_candidate_improvements": {
            arm: (losses[arm] - losses["margin_dynamic_normalized"]) / losses[arm]
            for arm in controls
        },
        "gates": gates,
        "development_pass": all(gates.values()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 is required")
    train_file = TokenFile(args.train_file, args.sequence_length)
    validation_file = TokenFile(args.validation_file, args.sequence_length)
    data_ledger = validate_data_ledger(
        args.data_manifest, args.train_file, args.validation_file, args.sequence_length
    )
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    payload: dict[str, Any] = {
        "schema": "dynamic-binding-swiglu-lm-development-v1",
        "candidate": "group-8 margin-damped token-conditioned binding SwiGLU",
        "scope": "one-seed 10M-token development falsification; no serving admission",
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "source_sha256": sha256_file(Path(__file__)),
        "algebra_source_sha256": sha256_file(ALGEBRA_SOURCE),
        "data_ledger": data_ledger,
        "device": torch.cuda.get_device_name(0),
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda},
        "candidate_shape": {"group_size": GROUP_SIZE, "groups_per_layer": GROUPS},
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "arms": {},
    }
    for arm in ARMS:
        print(json.dumps({"starting_arm": arm}), flush=True)
        payload["arms"][arm] = train_arm(arm, args, train_file, validation_file, device)
        payload["decision"] = decide(payload["arms"], args)
        write_payload(args.output, payload)
    return payload


def parse_steps(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("data/block-algebra-scratch/train.uint16.bin"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/block-algebra-scratch/validation.uint16.bin"))
    parser.add_argument("--data-manifest", type=Path, default=Path("results/block-algebra-scratch-data-manifest.json"))
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
    parser.add_argument("--seed", type=int, default=2861)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
