#!/usr/bin/env python3
"""Matched 50M-token LM screen for gauge-exposed partner-feedback SwiGLU."""

from __future__ import annotations

import argparse
import gc
import json
import math
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


VALID_ARMS = (
    "raw_baseline",
    "carrier_null",
    "self_feedback",
    "gauge_null",
    "partner_feedback",
)
PREREGISTRATION = Path("results/partner-feedback-swiglu-lm-preregistration.md")
INTEGRITY_MANIFEST = Path("results/partner-feedback-swiglu-lm-integrity-manifest.json")
TEST_SOURCE = Path("tests/test_partner_feedback_swiglu_lm_screen.py")
ALGEBRA_SOURCE = Path("experiments/partner_feedback_swiglu_gate.py")
ALGEBRA_TEST = Path("tests/test_partner_feedback_swiglu_gate.py")
ALGEBRA_RESULT = Path("results/partner-feedback-swiglu-stage0.json")
FUSED_SOURCE = Path("experiments/partner_feedback_swiglu_h100.py")
FUSED_TEST = Path("tests/test_partner_feedback_swiglu_h100.py")
FUSED_RESULT = Path("results/partner-feedback-swiglu-h100.json")

# Exact BF16 rounding of 1/sqrt(384).  Using the unrounded float32 value makes
# a BF16 pivot decode a small nonzero coefficient at the claimed endpoint.
CHART = 0.051025390625
CARRIER_GAIN = 4.0
LAMBDA_BOUND = 0.1


class PartnerFeedbackMLP(nn.Module):
    def __init__(self, source: nn.Module, arm: str) -> None:
        super().__init__()
        if arm not in {"carrier_null", "gauge_null", "self_feedback", "partner_feedback"}:
            raise ValueError(f"invalid partner-feedback arm: {arm}")
        self.arm = arm
        self.gate_proj = source.gate_proj
        self.down_proj = source.down_proj
        self.up_row0_tail = nn.Parameter(source.up_proj.weight[0, 1:].detach().clone())
        self.up_other_rows = nn.Parameter(source.up_proj.weight[1:].detach().clone())
        self.carrier_beta = nn.Parameter(torch.zeros((), device=source.up_proj.weight.device))
        self.ablation_mode = "full"
        self.record_diagnostics = False
        self.last_diagnostics: dict[str, float] | None = None

    def lambda_value(self, dtype: torch.dtype) -> torch.Tensor:
        value = LAMBDA_BOUND * torch.tanh(self.carrier_beta / LAMBDA_BOUND)
        return value.to(dtype)

    def canonical_up_weight(self) -> torch.Tensor:
        chart = self.up_row0_tail.new_tensor([CHART])
        row0 = torch.cat((chart, self.up_row0_tail), dim=0)
        return torch.cat((row0.unsqueeze(0), self.up_other_rows), dim=0)

    @staticmethod
    def partner_values(z0: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "self":
            return z0
        paired = z0.unflatten(-1, (BASELINE_INTERMEDIATE_SIZE // 2, 2))
        if mode == "shifted":
            # Shift pair sources by one pair, then swap within the source pair.
            paired = torch.roll(paired, shifts=1, dims=-2)
        return paired.flip(-1).flatten(-2)

    def activation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        canonical_up = F.linear(hidden_states, self.canonical_up_weight())
        coefficient = self.lambda_value(gate.dtype)
        scale = 1.0 + CARRIER_GAIN * coefficient
        up = torch.cat((canonical_up[..., :1] * scale, canonical_up[..., 1:]), dim=-1)
        z0 = F.silu(gate) * up
        if self.arm == "carrier_null" or self.ablation_mode == "zero":
            activation = z0
        else:
            if self.arm == "self_feedback" or self.ablation_mode == "self":
                source = z0
            elif self.arm == "gauge_null":
                source = self.partner_values(F.silu(gate), "partner")
            else:
                mode = "shifted" if self.ablation_mode == "shifted" else "partner"
                source = self.partner_values(z0, mode)
            activation = F.silu(gate + coefficient * source.clamp(-1.0, 1.0)) * up
        activation = torch.cat((activation[..., :1] / scale, activation[..., 1:]), dim=-1)
        if self.record_diagnostics:
            with torch.no_grad():
                float_scale = scale.float()
                condition = torch.maximum(float_scale.abs(), float_scale.abs().reciprocal())
                self.last_diagnostics = {
                    "activation_rms": float(activation.float().square().mean().sqrt()),
                    "activation_abs_max": float(activation.float().abs().max()),
                    "lambda": float(coefficient.float()),
                    "carrier_scale": float(float_scale),
                    "carrier_condition": float(condition),
                    "clip_fraction": float((z0.float().abs() > 1.0).float().mean()),
                }
        return activation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(hidden_states))


def build_model(device: torch.device, arm: str):
    config = scratch_config()
    model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa").to(device)
    model.config.use_cache = False
    modules: list[PartnerFeedbackMLP] = []
    if arm == "raw_baseline":
        with torch.no_grad():
            for layer in model.model.layers:
                layer.mlp.up_proj.weight[0, 0] = CHART
        return model, modules
    for layer in model.model.layers:
        replacement = PartnerFeedbackMLP(layer.mlp, arm).to(device)
        layer.mlp = replacement
        modules.append(replacement)
    return model, modules


@torch.no_grad()
def activation_diagnostics(model, modules, validation_file, device):
    model.eval()
    inputs, _ = validation_file.batch(0, 2, device)
    if not modules:
        captured = []
        handle = model.model.layers[0].mlp.down_proj.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].detach())
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(input_ids=inputs, use_cache=False)
        handle.remove()
        activation = captured[0]
        return {
            "activation_rms_layer_median": float(
                activation.float().square().mean().sqrt()
            ),
            "activation_abs_max_layer_median": float(
                activation.float().abs().max()
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
        raise RuntimeError("missing partner feedback diagnostics")
    typed = [record for record in records if record is not None]
    return {key + "_layer_median": float(np.median([r[key] for r in typed])) for key in typed[0]} | {
        "lambda_abs_layer_mean": float(np.mean([abs(r["lambda"]) for r in typed])),
        "carrier_condition_layer_max": float(max(r["carrier_condition"] for r in typed)),
        "carrier_scale_layer_min": float(min(r["carrier_scale"] for r in typed)),
    }


def validate_protocol(args, arms):
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
        "seed": 809,
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
        "prediction_tokens": args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length,
    }
    checks = {
        key: (expected_value in actual[key] if key == "device_contains" else actual[key] == expected_value)
        for key, expected_value in expected.items()
    }
    if args.strict_protocol and not all(checks.values()):
        failed = {k: {"expected": expected[k], "actual": actual[k]} for k, v in checks.items() if not v}
        raise ValueError(f"invalid experiment protocol: {json.dumps(failed, sort_keys=True)}")
    return {"valid": all(checks.values()), "checks": checks, "expected": expected, "actual": actual}


def train_arm(arm, args, train_file, validation_file, device):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model, modules = build_model(device, arm)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    carrier_parameters = [module.carrier_beta for module in modules]
    carrier_ids = {id(parameter) for parameter in carrier_parameters}
    ordinary_parameters = [parameter for parameter in model.parameters() if id(parameter) not in carrier_ids]
    parameter_groups = [
        {"params": ordinary_parameters, "weight_decay": args.weight_decay}
    ]
    if carrier_parameters:
        parameter_groups.append(
            {"params": carrier_parameters, "weight_decay": 0.0}
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True,
    )
    for group in optimizer.param_groups:
        group["base_lr"] = args.learning_rate
    diagnostics = {"initial": activation_diagnostics(model, modules, validation_file, device)}
    evaluations = {"0": evaluate(model, validation_file, args.eval_batches, args.eval_batch_size, device)}
    torch.cuda.reset_peak_memory_stats(); optimizer.zero_grad(set_to_none=True)
    losses, durations, norms = [], [], []
    for step in range(args.steps):
        model.train()
        multiplier = lr_multiplier(step, args.steps, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = float(group["base_lr"]) * multiplier
        started = time.perf_counter(); accumulated = 0.0
        for micro in range(args.gradient_accumulation):
            index = step * args.gradient_accumulation + micro
            inputs, targets = train_file.batch(index, args.micro_batch_size, device)
            loss = causal_loss(model, inputs, targets)
            (loss / args.gradient_accumulation).backward()
            accumulated += float(loss.detach()) / args.gradient_accumulation
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip))
        if not math.isfinite(norm) or not math.isfinite(accumulated):
            raise RuntimeError(f"non-finite training state in {arm} at step {step + 1}")
        optimizer.step(); optimizer.zero_grad(set_to_none=True); torch.cuda.synchronize()
        losses.append(accumulated); durations.append(time.perf_counter() - started); norms.append(norm)
        if step + 1 in args.eval_steps:
            evaluations[str(step + 1)] = evaluate(model, validation_file, args.eval_batches, args.eval_batch_size, device)
            print(json.dumps({"arm": arm, "step": step + 1, "train_loss": accumulated,
                              "validation_loss": evaluations[str(step + 1)]["loss"], "gradient_norm": norm}), flush=True)
    diagnostics["terminal"] = activation_diagnostics(model, modules, validation_file, device)
    ablations = {}
    if arm == "partner_feedback":
        for mode in ("zero", "self", "shifted"):
            for module in modules:
                module.ablation_mode = mode
            ablations[mode] = evaluate(model, validation_file, args.eval_batches, args.eval_batch_size, device)
        for module in modules:
            module.ablation_mode = "full"
    prediction_tokens = args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length
    result = {
        "arm": arm, "total_parameters": total, "trainable_parameters": trainable,
        "evaluations": evaluations, "activation_diagnostics": diagnostics,
        "terminal_ablations": ablations,
        "train": {"steps": args.steps, "prediction_tokens": prediction_tokens,
                  "mean_loss": float(np.mean(losses)), "max_loss": max(losses), "final_loss": losses[-1],
                  "max_preclip_gradient_norm": max(norms), "elapsed_seconds": sum(durations),
                  "tokens_per_second": prediction_tokens / sum(durations),
                  "median_step_seconds_after_five": float(np.median(durations[5:])),
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "nonfinite": False},
    }
    del optimizer, modules, model; gc.collect(); torch.cuda.empty_cache()
    return result


def decide(results, integrity_valid):
    if set(results) != set(VALID_ARMS):
        return {"complete": False, "missing_arms": sorted(set(VALID_ARMS) - set(results))}
    terminal, early = "1525", "305"
    losses = {name: r["evaluations"][terminal]["loss"] for name, r in results.items()}
    early_losses = {name: r["evaluations"][early]["loss"] for name, r in results.items()}
    candidate = results["partner_feedback"]
    intervals = {
        "partner_vs_" + name: paired_loss_interval(candidate, results[name], terminal)
        for name in ("raw_baseline", "carrier_null", "gauge_null", "self_feedback")
    }
    full_eval = candidate["evaluations"][terminal]
    full_wrapper = {"evaluations": {terminal: full_eval}}
    for mode, evaluation in candidate["terminal_ablations"].items():
        intervals["full_vs_" + mode] = paired_loss_interval(
            full_wrapper, {"evaluations": {terminal: evaluation}}, terminal
        )
    terminal_diag = candidate["activation_diagnostics"]["terminal"]
    ablation_losses = {"full": full_eval["loss"]} | {
        mode: evaluation["loss"] for mode, evaluation in candidate["terminal_ablations"].items()
    }
    baseline_rms = results["raw_baseline"]["activation_diagnostics"]["initial"]["activation_rms_layer_median"]
    rms_ratios = {name: r["activation_diagnostics"]["initial"]["activation_rms_layer_median"] / baseline_rms
                  for name, r in results.items()}
    controls = ("raw_baseline", "carrier_null", "self_feedback", "gauge_null")
    better_activation_control = min(
        ("carrier_null", "self_feedback", "gauge_null"),
        key=lambda name: losses[name],
    )
    gates = {
        "integrity_protocol_valid": integrity_valid,
        "identical_parameter_counts": len({r["total_parameters"] for r in results.values()}) == 1,
        "identical_trainable_parameter_counts": len({r["trainable_parameters"] for r in results.values()}) == 1,
        "initial_activation_rms_matched_within_25_percent": all(0.75 <= v <= 1.25 for v in rms_ratios.values()),
        "all_training_finite": all(not r["train"]["nonfinite"] for r in results.values()),
        "bounded_training": all(r["train"]["max_preclip_gradient_norm"] <= 100 and r["train"]["max_loss"] <= 20 for r in results.values()),
        "candidate_not_worse_at_10m": all(early_losses["partner_feedback"] <= early_losses[c] for c in controls),
        "candidate_0p05_percent_better_raw_baseline": (losses["raw_baseline"] - losses["partner_feedback"]) / losses["raw_baseline"] >= 0.0005,
        "candidate_0p025_percent_better_best_activation_control": (losses[better_activation_control] - losses["partner_feedback"]) / losses[better_activation_control] >= 0.00025,
        "paired_interval_favors_candidate_vs_raw_baseline": intervals["partner_vs_raw_baseline"]["upper_95"] < 0,
        "paired_interval_favors_candidate_vs_best_activation_control": intervals["partner_vs_" + better_activation_control]["upper_95"] < 0,
        "lambda_live": terminal_diag["lambda_abs_layer_mean"] >= 0.002,
        "carrier_stable": terminal_diag["carrier_scale_layer_min"] > 0 and terminal_diag["carrier_condition_layer_max"] < 2,
        "zero_ablation_hurts_0p05_percent": (ablation_losses["zero"] - ablation_losses["full"]) / ablation_losses["full"] >= 0.0005,
        "zero_interval_favors_full": intervals["full_vs_zero"]["upper_95"] < 0,
        "self_ablation_hurts_0p02_percent": (ablation_losses["self"] - ablation_losses["full"]) / ablation_losses["full"] >= 0.0002,
        "self_interval_favors_full": intervals["full_vs_self"]["upper_95"] < 0,
    }
    return {"complete": True, "terminal_step": 1525, "terminal_losses": losses,
            "early_losses": early_losses, "terminal_ablation_losses": ablation_losses,
            "paired_terminal_intervals": intervals, "terminal_candidate_diagnostics": terminal_diag,
            "initial_activation_rms_ratios_to_baseline": rms_ratios, "gates": gates,
            "best_activation_control": better_activation_control,
            "advance_to_second_matching": all(gates.values())}


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    arms = tuple(v.strip() for v in args.arms.split(",") if v.strip())
    if arms != VALID_ARMS:
        raise ValueError(f"strict screen requires arms in order {VALID_ARMS}")
    train_file = TokenFile(args.train_file, args.sequence_length)
    validation_file = TokenFile(args.validation_file, args.sequence_length)
    data_ledger = validate_data_ledger(args.data_manifest, args.train_file, args.validation_file, args.sequence_length)
    protocol = validate_protocol(args, arms)
    if train_file.sequence_count < args.steps * args.gradient_accumulation * args.micro_batch_size:
        raise ValueError("training token file too short")
    if validation_file.sequence_count < args.eval_batches * args.eval_batch_size:
        raise ValueError("validation token file too short")
    manifest = json.loads(INTEGRITY_MANIFEST.read_text())
    paths = {"source": Path(__file__), "preregistration": PREREGISTRATION, "test": TEST_SOURCE,
             "algebra_source": ALGEBRA_SOURCE, "algebra_test": ALGEBRA_TEST, "algebra_result": ALGEBRA_RESULT,
             "fused_source": FUSED_SOURCE, "fused_test": FUSED_TEST, "fused_result": FUSED_RESULT}
    integrity_checks = {name: manifest.get(name + "_sha256") == sha256_file(path) for name, path in paths.items()}
    if not all(integrity_checks.values()):
        raise ValueError(f"invalid frozen integrity manifest: {integrity_checks}")
    torch.set_float32_matmul_precision("high"); torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    payload = {
        "candidate": "gauge-exposed partner-feedback SwiGLU", "scope": "matched one-seed 50M-token scratch screen",
        "source_sha256": sha256_file(Path(__file__)), "preregistration_sha256": sha256_file(PREREGISTRATION),
        "integrity_manifest_sha256": sha256_file(INTEGRITY_MANIFEST), "integrity_checks": integrity_checks,
        "data_manifest_sha256": sha256_file(args.data_manifest), "data_ledger": data_ledger,
        "experiment_protocol": protocol, "base_config": MODEL, "base_config_revision": MODEL_REVISION,
        "candidate_shape": {"hidden_size": HIDDEN_SIZE, "intermediate_size": BASELINE_INTERMEDIATE_SIZE,
                            "matching": "adjacent involution", "chart": CHART, "carrier_gain": CARRIER_GAIN,
                            "lambda_bound": LAMBDA_BOUND, "training_parameterization": "one removed U pivot replaced by beta"},
        "device": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "arms": {},
    }
    for arm in arms:
        print(json.dumps({"starting_arm": arm}), flush=True)
        payload["arms"][arm] = train_arm(arm, args, train_file, validation_file, device)
        payload["decision"] = decide(payload["arms"], data_ledger["valid"] and protocol["valid"] and all(integrity_checks.values()))
        write_payload(args.output, payload)
    return payload


def parse_steps(text):
    return [int(v) for v in text.split(",") if v]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("data/block-algebra-scratch/train.uint16.bin"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/block-algebra-scratch/validation.uint16.bin"))
    parser.add_argument("--data-manifest", type=Path, default=Path("results/block-algebra-scratch-data-manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/partner-feedback-swiglu-lm-screen.json"))
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
    parser.add_argument("--seed", type=int, default=809)
    parser.add_argument("--strict-protocol", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(); payload = run(args)
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
