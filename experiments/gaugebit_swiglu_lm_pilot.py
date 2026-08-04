#!/usr/bin/env python3
"""Matched 10M-token scratch-LM pilot for GaugeBit SwiGLU.

This is a development falsification, not a frozen formal result.  The forward
pass always uses hard activation choices.  A straight-through estimator is a
training-only search rule; no soft mixture is admitted to the served model.
"""

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
    HIDDEN_SIZE,
    causal_loss,
    scratch_config,
)


ARMS = (
    "raw_baseline",
    "canonical_null",
    "fixed_half",
    "all_even",
    "learned_gaugebit",
)
OUTPUT = Path("results/gaugebit-swiglu-lm-pilot-development.json")
ALGEBRA_SOURCE = Path("experiments/gaugebit_swiglu.py")
GROUP_SIZE = 32
GROUPS = BASELINE_INTERMEDIATE_SIZE // GROUP_SIZE
# Exact BF16 representation of 1/sqrt(384), shared with prior gauge screens.
CHART = 0.051025390625
BETA_INITIAL = 0.02
STE_TEMPERATURE = 0.05


class GaugeBitMLP(nn.Module):
    def __init__(self, source: nn.Module, arm: str) -> None:
        super().__init__()
        if arm not in ARMS[1:]:
            raise ValueError(f"invalid GaugeBit arm: {arm}")
        self.arm = arm
        self.gate_proj = source.gate_proj
        self.down_proj = source.down_proj
        self.up_packed = nn.Parameter(source.up_proj.weight.detach().clone())
        rows = torch.arange(0, BASELINE_INTERMEDIATE_SIZE, GROUP_SIZE)
        pivots = torch.arange(GROUPS) % HIDDEN_SIZE
        self.register_buffer("representative_rows", rows, persistent=False)
        self.register_buffer("pivot_columns", pivots, persistent=False)
        fixed_half = (torch.arange(GROUPS) % 2).float()
        self.register_buffer("fixed_half_bits", fixed_half, persistent=False)
        with torch.no_grad():
            self.up_packed[rows, pivots] = BETA_INITIAL
        self.ablation_mode = "full"
        self.record_diagnostics = False
        self.last_diagnostics: dict[str, float] | None = None

    def betas(self) -> torch.Tensor:
        return self.up_packed[self.representative_rows, self.pivot_columns]

    def effective_up_weight(self) -> torch.Tensor:
        weight = self.up_packed.clone()
        weight[self.representative_rows, self.pivot_columns] = CHART
        return weight

    def selector_probabilities(self) -> torch.Tensor:
        return torch.sigmoid(-self.betas() / STE_TEMPERATURE)

    def hard_bits(self) -> torch.Tensor:
        return self.betas() < 0.0

    def group_selectors(self, dtype: torch.dtype) -> torch.Tensor:
        soft = self.selector_probabilities()
        if self.arm == "learned_gaugebit":
            hard = self.hard_bits().to(soft.dtype)
            selectors = hard.detach() - soft.detach() + soft
        elif self.arm == "fixed_half":
            selectors = self.fixed_half_bits + 0.0 * soft
        elif self.arm == "all_even":
            selectors = torch.ones_like(soft) + 0.0 * soft
        else:
            selectors = 0.0 * soft
        if self.ablation_mode == "all_silu":
            selectors = 0.0 * soft
        elif self.ablation_mode == "all_even":
            selectors = torch.ones_like(soft) + 0.0 * soft
        elif self.ablation_mode == "fixed_half":
            selectors = self.fixed_half_bits + 0.0 * soft
        elif self.ablation_mode == "shuffled":
            selectors = torch.roll(selectors, shifts=1, dims=0)
        elif self.ablation_mode == "complement":
            selectors = 1.0 - selectors
        return selectors.to(dtype)

    def activation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = F.linear(hidden_states, self.effective_up_weight())
        selectors = self.group_selectors(gate.dtype).repeat_interleave(GROUP_SIZE)
        ordinary = F.silu(gate)
        even = gate * torch.tanh(gate)
        activated_gate = torch.lerp(ordinary, even, selectors)
        activation = activated_gate * up
        if self.record_diagnostics:
            with torch.no_grad():
                betas = self.betas().float()
                hard = self.hard_bits().float()
                probabilities = self.selector_probabilities().float()
                self.last_diagnostics = {
                    "activation_rms": float(activation.float().square().mean().sqrt()),
                    "activation_abs_max": float(activation.float().abs().max()),
                    "beta_mean": float(betas.mean()),
                    "beta_abs_mean": float(betas.abs().mean()),
                    "beta_abs_min": float(betas.abs().min()),
                    "hard_even_fraction": float(hard.mean()),
                    "soft_even_fraction": float(probabilities.mean()),
                }
        return activation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(hidden_states))


def initialize_chart_pivots(model: nn.Module) -> None:
    rows = torch.arange(0, BASELINE_INTERMEDIATE_SIZE, GROUP_SIZE)
    pivots = torch.arange(GROUPS) % HIDDEN_SIZE
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.up_proj.weight[rows, pivots] = CHART


def build_model(
    device: torch.device, arm: str
) -> tuple[nn.Module, list[GaugeBitMLP]]:
    if arm not in ARMS:
        raise ValueError(f"invalid arm: {arm}")
    model = AutoModelForCausalLM.from_config(
        scratch_config(), attn_implementation="sdpa"
    ).to(device)
    model.config.use_cache = False
    initialize_chart_pivots(model)
    modules: list[GaugeBitMLP] = []
    if arm != "raw_baseline":
        for layer in model.model.layers:
            replacement = GaugeBitMLP(layer.mlp, arm).to(device)
            layer.mlp = replacement
            modules.append(replacement)
    return model, modules


def optimizer_for(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True,
    )
    for group in optimizer.param_groups:
        group["base_lr"] = args.learning_rate
    return optimizer


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
    modules: list[GaugeBitMLP],
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
            "beta_mean_layer_median": BETA_INITIAL,
            "beta_abs_mean_layer_median": BETA_INITIAL,
            "beta_abs_min_layer_min": BETA_INITIAL,
            "hard_even_fraction_layer_mean": 0.0,
            "soft_even_fraction_layer_mean": float(
                1.0 / (1.0 + math.exp(BETA_INITIAL / STE_TEMPERATURE))
            ),
        }
    for module in modules:
        module.record_diagnostics = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=inputs, use_cache=False)
    records = [module.last_diagnostics for module in modules]
    for module in modules:
        module.record_diagnostics = False
    if any(record is None for record in records):
        raise RuntimeError("missing GaugeBit diagnostics")
    typed = [record for record in records if record is not None]
    return {
        "activation_rms_layer_median": float(np.median([r["activation_rms"] for r in typed])),
        "activation_abs_max_layer_median": float(np.median([r["activation_abs_max"] for r in typed])),
        "beta_mean_layer_median": float(np.median([r["beta_mean"] for r in typed])),
        "beta_abs_mean_layer_median": float(np.median([r["beta_abs_mean"] for r in typed])),
        "beta_abs_min_layer_min": float(min(r["beta_abs_min"] for r in typed)),
        "hard_even_fraction_layer_mean": float(np.mean([r["hard_even_fraction"] for r in typed])),
        "soft_even_fraction_layer_mean": float(np.mean([r["soft_even_fraction"] for r in typed])),
    }


def evaluate_ablation(
    model: nn.Module,
    modules: list[GaugeBitMLP],
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
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    parameter_tensors = len(list(model.parameters()))
    optimizer = optimizer_for(model, args)
    diagnostics = {
        "initial": activation_diagnostics(model, modules, validation_file, device)
    }
    evaluations = {
        "0": evaluate(model, validation_file, args.eval_batches, args.eval_batch_size, device)
    }
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    losses: list[float] = []
    gradient_norms: list[float] = []
    durations: list[float] = []
    selector_flip_steps: list[dict[str, int]] = []
    previous_bits = None
    if arm == "learned_gaugebit":
        previous_bits = torch.cat([module.hard_bits() for module in modules]).detach().clone()
    for step in range(args.steps):
        model.train()
        multiplier = lr_multiplier(step, args.steps, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = float(group["base_lr"]) * multiplier
        started = time.perf_counter()
        accumulated = 0.0
        for micro in range(args.gradient_accumulation):
            batch_index = step * args.gradient_accumulation + micro
            inputs, targets = train_file.batch(batch_index, args.micro_batch_size, device)
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
        if previous_bits is not None:
            current_bits = torch.cat([module.hard_bits() for module in modules]).detach()
            changed = int((current_bits != previous_bits).sum().item())
            if changed:
                selector_flip_steps.append({"step": step + 1, "changed_bits": changed})
            previous_bits = current_bits.clone()
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
    diagnostics["terminal"] = activation_diagnostics(
        model, modules, validation_file, device
    )
    ablations: dict[str, Any] = {}
    if arm == "learned_gaugebit":
        for mode in ("all_silu", "all_even", "fixed_half", "shuffled", "complement"):
            ablations[mode] = evaluate_ablation(
                model, modules, validation_file, args, device, mode
            )
    selector_bits = [
        module.hard_bits().detach().cpu().to(torch.int8).tolist()
        for module in modules
    ]
    prediction_tokens = (
        args.steps
        * args.gradient_accumulation
        * args.micro_batch_size
        * args.sequence_length
    )
    result = {
        "arm": arm,
        "total_parameters": total_parameters,
        "parameter_tensors": parameter_tensors,
        "optimizer_state_bytes": optimizer_state_bytes(optimizer),
        "evaluations": evaluations,
        "activation_diagnostics": diagnostics,
        "terminal_ablations": ablations,
        "terminal_hard_bits_by_layer": selector_bits,
        "selector_training": {
            "flip_steps": selector_flip_steps,
            "total_bit_flips": sum(record["changed_bits"] for record in selector_flip_steps),
            "last_flip_step": 0 if not selector_flip_steps else selector_flip_steps[-1]["step"],
        },
        "train": {
            "steps": args.steps,
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
    candidate = results["learned_gaugebit"]
    static_arms = ("raw_baseline", "canonical_null", "fixed_half", "all_even")
    best_static = min(static_arms, key=lambda arm: losses[arm])
    intervals = {
        "candidate_vs_" + arm: paired_loss_interval(candidate, results[arm], terminal)
        for arm in static_arms
    }
    full = {"evaluations": {terminal: candidate["evaluations"][terminal]}}
    for mode, evaluation in candidate["terminal_ablations"].items():
        intervals["full_vs_" + mode] = paired_loss_interval(
            full, {"evaluations": {terminal: evaluation}}, terminal
        )
    ablation_losses = {
        mode: evaluation["loss"]
        for mode, evaluation in candidate["terminal_ablations"].items()
    }
    initial_losses = {
        arm: result["evaluations"]["0"]["loss"] for arm, result in results.items()
    }
    terminal_diag = candidate["activation_diagnostics"]["terminal"]
    all_numeric_diagnostics = [
        value
        for result in results.values()
        for phase in ("initial", "terminal")
        for value in result["activation_diagnostics"][phase].values()
    ]
    gates = {
        "equal_learned_parameter_counts": len({r["total_parameters"] for r in results.values()}) == 1,
        "equal_parameter_tensor_counts": len({r["parameter_tensors"] for r in results.values()}) == 1,
        "equal_optimizer_state_bytes": len({r["optimizer_state_bytes"] for r in results.values()}) == 1,
        "all_training_and_diagnostics_finite": (
            all(not r["train"]["nonfinite"] for r in results.values())
            and all(math.isfinite(value) for value in all_numeric_diagnostics)
        ),
        "canonical_null_initial_exact_to_raw": abs(
            initial_losses["canonical_null"] - initial_losses["raw_baseline"]
        ) <= 1e-7,
        "candidate_initial_exact_to_raw": abs(
            initial_losses["learned_gaugebit"] - initial_losses["raw_baseline"]
        ) <= 1e-7,
        "canonical_null_terminal_noninferior_0p05_percent": (
            losses["canonical_null"] - losses["raw_baseline"]
        ) / losses["raw_baseline"] <= 0.0005,
        "candidate_improves_raw_0p05_percent": (
            losses["raw_baseline"] - losses["learned_gaugebit"]
        ) / losses["raw_baseline"] >= 0.0005,
        "candidate_improves_canonical_null_0p05_percent": (
            losses["canonical_null"] - losses["learned_gaugebit"]
        ) / losses["canonical_null"] >= 0.0005,
        "candidate_improves_best_static_0p025_percent": (
            losses[best_static] - losses["learned_gaugebit"]
        ) / losses[best_static] >= 0.00025,
        "paired_interval_favors_candidate_vs_raw": intervals["candidate_vs_raw_baseline"]["upper_95"] < 0.0,
        "paired_interval_favors_candidate_vs_null": intervals["candidate_vs_canonical_null"]["upper_95"] < 0.0,
        "paired_interval_favors_candidate_vs_best_static": intervals["candidate_vs_" + best_static]["upper_95"] < 0.0,
        "all_silu_ablation_hurts_0p02_percent": (
            ablation_losses["all_silu"] - losses["learned_gaugebit"]
        ) / losses["learned_gaugebit"] >= 0.0002,
        "paired_interval_favors_full_vs_all_silu": intervals["full_vs_all_silu"]["upper_95"] < 0.0,
        "selector_is_live": terminal_diag["hard_even_fraction_layer_mean"] >= 0.025,
        "selector_not_numerically_on_boundary": terminal_diag["beta_abs_min_layer_min"] >= 1e-5,
        "selector_stable_final_20_percent": candidate["selector_training"]["last_flip_step"] <= math.floor(0.8 * args.steps),
    }
    return {
        "complete": True,
        "terminal_losses": losses,
        "initial_losses": initial_losses,
        "best_static_arm": best_static,
        "terminal_ablation_losses": ablation_losses,
        "paired_intervals": intervals,
        "terminal_candidate_diagnostics": terminal_diag,
        "relative_candidate_improvements": {
            arm: (losses[arm] - losses["learned_gaugebit"]) / losses[arm]
            for arm in static_arms
        },
        "gates": gates,
        "development_pass": all(gates.values()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps not in args.eval_steps:
        raise ValueError("terminal step must be included in eval steps")
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
        "schema": "gaugebit-swiglu-lm-pilot-development-v1",
        "candidate": "group-32 hard GaugeBit SwiGLU",
        "scope": "matched one-seed 10M-token scratch-LM falsification; no runtime admission",
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "algebra_source_sha256": sha256_file(ALGEBRA_SOURCE),
        "source_sha256": sha256_file(Path(__file__)),
        "data_ledger": data_ledger,
        "device": torch.cuda.get_device_name(0),
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda},
        "candidate_shape": {
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": BASELINE_INTERMEDIATE_SIZE,
            "group_size": GROUP_SIZE,
            "groups_per_layer": GROUPS,
            "chart": CHART,
            "beta_initial": BETA_INITIAL,
            "ste_temperature": STE_TEMPERATURE,
            "hard_forward_only": True,
            "served_selector_storage_condition": "signed existing per-row quantization scales",
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
    parser.add_argument("--seed", type=int, default=2719)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
